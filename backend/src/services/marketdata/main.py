"""
Market Data Service - Real-time market data streaming and processing.

This service subscribes to market data for symbols in the watchlist table,
processes tick data, writes to the database, and handles dynamic subscription updates.
Uses event-driven streaming following TWS best practices.
"""

import asyncio
import signal
import sys
import time
import math
from datetime import datetime, timezone, time as dt_time
from typing import Dict, List, Optional, Set
from decimal import Decimal
from zoneinfo import ZoneInfo

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
from ib_insync import IB, Stock, Ticker, RealTimeBarList
from collections import deque

# Import our common modules
# Add the src directory to Python path for Docker compatibility
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from common.config import get_settings
from common.db import get_db_session, execute_with_retry, initialize_database
from common.models import WatchlistEntry, Tick, Symbol, HealthStatus, Candle
from common.logging import configure_service_logging, log_market_data_event, log_system_event
from common.notify import listen_for_watchlist_updates, get_notification_manager, initialize_notifications
from common.sync_status import update_sync_status
from sqlalchemy.dialects.postgresql import insert
from tws_bridge.ib_client import create_ib_client
from tws_bridge.client_ids import allocate_service_client_id, heartbeat_service_client_id, release_service_client_id


class MarketDataService:
    """Market data streaming service with event-driven architecture."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = configure_service_logging("marketdata")
        self.app = FastAPI(title="Market Data Service")
        
        # TWS connection
        self.client_id = None
        self.ib_client = None
        self.connected = False
        
        # Subscription management
        self.subscribed_symbols: Dict[str, Ticker] = {}
        self.watchlist_symbols: Set[str] = set()
        self.max_subscriptions = self.settings.market_data.max_subscriptions
        
        # Real-time 5-second bar subscriptions
        self.realtime_bar_subscriptions: Dict[str, RealTimeBarList] = {}
        self.bar_cache: Dict[str, deque] = {}  # In-memory cache of recent bars per symbol
        self.bar_cache_size = 1000  # Keep last 1000 bars (~83 minutes at 5-sec bars)
        
        # WebSocket connections for real-time updates
        self.websocket_connections: List[WebSocket] = []
        self.bar_websocket_connections: List[WebSocket] = []  # Websockets for bar streaming
        
        # Service state
        self.running = False
        self.collection_enabled = True  # Market data collection state
        self.last_heartbeat = time.time()
        
        # Setup FastAPI routes
        self._setup_routes()
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _is_market_hours(self, timestamp: datetime) -> bool:
        """
        Check if timestamp falls within configured market hours.
        
        Args:
            timestamp: Timestamp to check (will be converted to market timezone)
            
        Returns:
            True if during market hours, False otherwise or if filtering disabled
        """
        # If filtering disabled, always return True (save all bars)
        if not self.settings.market_hours.enabled:
            return True
        
        try:
            # Get market timezone
            market_tz = ZoneInfo(self.settings.market_hours.timezone)
            
            # Convert timestamp to market timezone
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            market_time = timestamp.astimezone(market_tz)
            
            # Skip weekends (Saturday = 5, Sunday = 6)
            if market_time.weekday() >= 5:
                return False
            
            # Check time of day
            current_time = market_time.time()
            market_open = dt_time(
                self.settings.market_hours.market_open_hour,
                self.settings.market_hours.market_open_minute
            )
            market_close = dt_time(
                self.settings.market_hours.market_close_hour,
                self.settings.market_hours.market_close_minute
            )
            
            return market_open <= current_time < market_close
            
        except Exception as e:
            self.logger.error(f"Error checking market hours: {e}")
            # On error, default to saving the bar (fail-open)
            return True
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/healthz")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy" if self.connected else "unhealthy",
                "service": "marketdata",
                "client_id": self.client_id,
                "connected": self.connected,
                "collection_enabled": self.collection_enabled,
                "subscriptions": len(self.subscribed_symbols),
                "max_subscriptions": self.max_subscriptions,
                "watchlist_size": len(self.watchlist_symbols),
                "last_heartbeat": self.last_heartbeat
            }
        
        @self.app.get("/subscriptions")
        async def get_subscriptions():
            """Get current market data subscriptions."""
            subscriptions = []
            for symbol, ticker in self.subscribed_symbols.items():
                subscriptions.append({
                    "symbol": symbol,
                    "bid": float(ticker.bid) if ticker.bid and ticker.bid > 0 else None,
                    "ask": float(ticker.ask) if ticker.ask and ticker.ask > 0 else None,
                    "last": float(ticker.last) if ticker.last and ticker.last > 0 else None,
                    "bid_size": ticker.bidSize if ticker.bidSize else None,
                    "ask_size": ticker.askSize if ticker.askSize else None,
                    "last_size": ticker.lastSize if ticker.lastSize else None
                })
            return {
                "subscriptions": subscriptions,
                "count": len(subscriptions),
                "max_allowed": self.max_subscriptions
            }
        
        @self.app.post("/collection/enable")
        async def enable_collection():
            """Enable market data collection."""
            return await self.enable_collection()
        
        @self.app.post("/collection/disable")
        async def disable_collection():
            """Disable market data collection."""
            return await self.disable_collection()
        
        @self.app.get("/collection/status")
        async def get_collection_status():
            """Get market data collection status."""
            return await self.get_collection_status()
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time market data."""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                # Send current subscription data
                await self._send_websocket_update(websocket, {
                    "type": "subscription_status",
                    "data": {
                        "subscribed_symbols": list(self.subscribed_symbols.keys()),
                        "count": len(self.subscribed_symbols)
                    }
                })
                
                # Keep connection alive
                while True:
                    try:
                        await websocket.receive_text()
                    except WebSocketDisconnect:
                        break
                        
            except WebSocketDisconnect:
                pass
            finally:
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)
        
        @self.app.post("/realtime-bars/subscribe")
        async def subscribe_realtime_bars():
            """Subscribe to real-time 5-second bars for watchlist symbols."""
            try:
                subscribed = await self._subscribe_realtime_bars()
                return {
                    "status": "success",
                    "subscribed_symbols": subscribed,
                    "count": len(subscribed)
                }
            except Exception as e:
                self.logger.error(f"Failed to subscribe to real-time bars: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "message": str(e)}
                )
        
        @self.app.get("/realtime-bars/subscriptions")
        async def get_realtime_bar_subscriptions():
            """Get active real-time bar subscriptions."""
            return {
                "subscriptions": list(self.realtime_bar_subscriptions.keys()),
                "count": len(self.realtime_bar_subscriptions)
            }
        
        @self.app.get("/realtime-bars/cache/{symbol}")
        async def get_cached_bars(symbol: str, limit: int = 100):
            """Get cached bars for a symbol."""
            if symbol not in self.bar_cache:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"No cached bars for {symbol}"}
                )
            
            bars = list(self.bar_cache[symbol])[-limit:]
            return {
                "symbol": symbol,
                "count": len(bars),
                "bars": [
                    {
                        "timestamp": bar["timestamp"].isoformat(),
                        "open": float(bar["open"]),
                        "high": float(bar["high"]),
                        "low": float(bar["low"]),
                        "close": float(bar["close"]),
                        "volume": bar["volume"]
                    }
                    for bar in bars
                ]
            }
        
        @self.app.websocket("/ws/bars")
        async def websocket_bars(websocket: WebSocket):
            """WebSocket endpoint for real-time 5-second bar streaming."""
            await websocket.accept()
            self.bar_websocket_connections.append(websocket)
            self.logger.info(f"Bar WebSocket connected. Total: {len(self.bar_websocket_connections)}")
            
            try:
                # Send initial status
                await websocket.send_json({
                    "type": "status",
                    "data": {
                        "subscribed_symbols": list(self.realtime_bar_subscriptions.keys()),
                        "message": "Connected to real-time bar stream"
                    }
                })
                
                # Keep connection alive
                while True:
                    try:
                        await websocket.receive_text()
                    except WebSocketDisconnect:
                        break
                        
            except WebSocketDisconnect:
                pass
            finally:
                if websocket in self.bar_websocket_connections:
                    self.bar_websocket_connections.remove(websocket)
                self.logger.info(f"Bar WebSocket disconnected. Remaining: {len(self.bar_websocket_connections)}")
    
    async def start(self):
        """Start the market data service."""
        try:
            log_system_event(self.logger, "service_start", service="marketdata")
            
            # Initialize database
            initialize_database()
            
            # Initialize notifications
            initialize_notifications()
            
            # Allocate client ID
            self.client_id = allocate_service_client_id("marketdata")
            self.logger.info(f"Allocated TWS client ID: {self.client_id}")
            
            # Create IB client
            self.ib_client = create_ib_client(self.client_id)
            
            # Setup event handlers (following notes.md best practices)
            self._setup_event_handlers()
            
            # Connect to TWS
            await self._connect_to_tws()
            
            # Load watchlist and start subscriptions
            await self._load_watchlist()
            await self._subscribe_to_watchlist()
            
            # Subscribe to real-time 5-second bars
            try:
                subscribed = await self._subscribe_realtime_bars()
                self.logger.info(f"Subscribed to real-time bars for {len(subscribed)} symbols")
            except Exception as e:
                self.logger.error(f"Failed to subscribe to real-time bars: {e}")
            
            # Setup watchlist update notifications
            self._setup_watchlist_notifications()
            
            # Update health status
            await self._update_health_status("healthy")
            
            # Start heartbeat task
            asyncio.create_task(self._heartbeat_loop())
            
            self.running = True
            self.logger.info("Market data service started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start market data service: {e}")
            await self._update_health_status("unhealthy")
            raise
    
    async def stop(self):
        """Stop the market data service."""
        try:
            log_system_event(self.logger, "service_stop", service="marketdata")
            
            self.running = False
            
            # Cancel all subscriptions
            await self._unsubscribe_all()
            
            # Disconnect from TWS
            if self.ib_client:
                await self.ib_client.disconnect()
                self.connected = False
            
            # Release client ID
            if self.client_id:
                release_service_client_id(self.client_id, "marketdata")
            
            # Update health status
            await self._update_health_status("stopping")
            
            self.logger.info("Market data service stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping market data service: {e}")
    
    def _setup_event_handlers(self):
        """Setup TWS event handlers following notes.md best practices."""
        
        # Store the ticker update handler for later use
        def on_ticker_update(ticker: Ticker):
            """Handle real-time ticker updates (event-driven streaming)."""
            try:
                symbol = ticker.contract.symbol
                
                # Log market data event
                log_market_data_event(
                    self.logger, "tick_update", symbol,
                    bid=ticker.bid, ask=ticker.ask, last=ticker.last,
                    bid_size=ticker.bidSize, ask_size=ticker.askSize, last_size=ticker.lastSize
                )
                
                # Process tick data asynchronously
                asyncio.create_task(self._process_tick_data(ticker))
                
            except Exception as e:
                self.logger.error(f"Error in ticker update handler: {e}")
        
        # Store the handler for use when subscribing to individual tickers
        self._ticker_update_handler = on_ticker_update
        
        def on_error(req_id, error_code, error_string, contract):
            """Handle TWS errors."""
            if error_code == 200:  # No security definition found
                symbol = contract.symbol if contract else "unknown"
                self.logger.warning(f"No security definition for {symbol}")
            elif error_code in [162, 300]:  # Historical data errors
                self.logger.warning(f"Market data error {error_code}: {error_string}")
            elif error_code in [2104, 2106, 2158]:  # Connection OK messages
                self.logger.info(f"TWS connection info {error_code}: {error_string}")
            elif error_code >= 2000:  # Other informational messages
                self.logger.info(f"TWS info {error_code}: {error_string}")
            else:
                self.logger.error(f"TWS error {error_code}: {error_string}")
        
        def on_disconnect():
            """Handle TWS disconnection."""
            self.connected = False
            self.logger.warning("TWS connection lost")
            # Reconnection will be handled by heartbeat loop
        
        def on_connect():
            """Handle TWS connection."""
            self.connected = True
            self.logger.info("TWS connection established")
        
        # Connect event handlers
        self.ib_client.add_error_handler(on_error)
        self.ib_client.add_disconnection_handler(on_disconnect)
        self.ib_client.add_connection_handler(on_connect)
    
    async def _connect_to_tws(self):
        """Connect to TWS with retry logic."""
        connected = await self.ib_client.connect()
        if not connected:
            raise ConnectionError("Failed to connect to TWS")
        
        self.connected = True
        self.logger.info(f"Connected to TWS on {self.settings.tws.host}:{self.settings.tws.port}")
    
    async def _load_watchlist(self):
        """Load symbols from watchlist table."""
        def _get_watchlist(session):
            entries = session.query(WatchlistEntry).all()
            return [entry.symbol for entry in entries]
        
        try:
            symbols = execute_with_retry(_get_watchlist)
            self.watchlist_symbols = set(symbols)
            self.logger.info(f"Loaded {len(symbols)} symbols from watchlist: {symbols}")
        except Exception as e:
            self.logger.error(f"Failed to load watchlist: {e}")
            self.watchlist_symbols = set()
    
    async def _subscribe_to_watchlist(self):
        """Subscribe to market data for all watchlist symbols."""
        if not self.connected:
            self.logger.warning("Not connected to TWS, skipping subscriptions")
            return
        
        # Enforce subscription limit
        symbols_to_subscribe = list(self.watchlist_symbols)[:self.max_subscriptions]
        
        if len(self.watchlist_symbols) > self.max_subscriptions:
            self.logger.warning(
                f"Watchlist has {len(self.watchlist_symbols)} symbols, "
                f"limiting to {self.max_subscriptions} due to MAX_SUBSCRIPTIONS"
            )
        
        for symbol in symbols_to_subscribe:
            try:
                await self._subscribe_to_symbol(symbol)
            except Exception as e:
                self.logger.error(f"Failed to subscribe to {symbol}: {e}")
        
        self.logger.info(f"Subscribed to {len(self.subscribed_symbols)} symbols")
    
    async def _subscribe_to_symbol(self, symbol: str):
        """Subscribe to market data for a single symbol."""
        if symbol in self.subscribed_symbols:
            self.logger.debug(f"Already subscribed to {symbol}")
            return
        
        if len(self.subscribed_symbols) >= self.max_subscriptions:
            self.logger.warning(f"Max subscriptions ({self.max_subscriptions}) reached")
            return
        
        try:
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Request market data (event-driven streaming)
            ticker = await self.ib_client.req_market_data_with_retry(
                contract=contract,
                generic_tick_list="",
                snapshot=False,  # Streaming, not snapshot
                regulatory_snapshot=False
                # Note: market_data_type parameter removed as it's not supported by ib-insync
            )
            
            # Attach event handler to this ticker
            ticker.updateEvent += self._ticker_update_handler
            
            self.subscribed_symbols[symbol] = ticker
            
            log_market_data_event(self.logger, "subscription_added", symbol)
            
            # Broadcast to WebSocket clients
            await self._broadcast_websocket_update({
                "type": "subscription_added",
                "data": {"symbol": symbol}
            })
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to {symbol}: {e}")
            raise
    
    async def _unsubscribe_from_symbol(self, symbol: str):
        """Unsubscribe from market data for a single symbol."""
        if symbol not in self.subscribed_symbols:
            return
        
        try:
            ticker = self.subscribed_symbols[symbol]
            contract = ticker.contract
            
            # Remove event handler
            ticker.updateEvent -= self._ticker_update_handler
            
            await self.ib_client.cancel_market_data(contract)
            
            del self.subscribed_symbols[symbol]
            
            log_market_data_event(self.logger, "subscription_removed", symbol)
            
            # Broadcast to WebSocket clients
            await self._broadcast_websocket_update({
                "type": "subscription_removed",
                "data": {"symbol": symbol}
            })
            
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from {symbol}: {e}")
    
    async def _unsubscribe_all(self):
        """Unsubscribe from all market data."""
        symbols = list(self.subscribed_symbols.keys())
        for symbol in symbols:
            await self._unsubscribe_from_symbol(symbol)
        
        self.logger.info("Unsubscribed from all market data")
    
    def _is_valid_number(self, value) -> bool:
        """Check if a value is a valid number (not None, not NaN, not inf)."""
        if value is None:
            return False
        try:
            return not (math.isnan(value) or math.isinf(value))
        except (TypeError, ValueError):
            return False
    
    def _safe_int(self, value) -> Optional[int]:
        """Safely convert a value to int, handling NaN, None, and inf."""
        if not self._is_valid_number(value):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    
    async def _process_tick_data(self, ticker: Ticker):
        """Process and store tick data in database."""
        try:
            # Check if collection is enabled
            if not self.collection_enabled:
                return
                
            symbol = ticker.contract.symbol
            
            # Extract tick data with proper NaN handling
            tick_data = {
                'symbol': symbol,
                'ts': datetime.now(timezone.utc),
                'bid': Decimal(str(ticker.bid)) if self._is_valid_number(ticker.bid) and ticker.bid > 0 else None,
                'ask': Decimal(str(ticker.ask)) if self._is_valid_number(ticker.ask) and ticker.ask > 0 else None,
                'last': Decimal(str(ticker.last)) if self._is_valid_number(ticker.last) and ticker.last > 0 else None,
                'bid_size': self._safe_int(ticker.bidSize),
                'ask_size': self._safe_int(ticker.askSize),
                'last_size': self._safe_int(ticker.lastSize)
            }
            
            # Only store if we have meaningful data
            if not any([tick_data['bid'], tick_data['ask'], tick_data['last']]):
                return
            
            # Store in database
            def _store_tick(session):
                tick = Tick(**tick_data)
                session.add(tick)
                session.commit()
                return tick
            
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: execute_with_retry(_store_tick)
            )
            
            # Record sync timing (source + database)
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: update_sync_status(
                    "market_ticks",
                    source_ts=tick_data['ts'],
                    db_ts=datetime.now(timezone.utc)
                )
            )
            
            # Broadcast to WebSocket clients
            await self._broadcast_websocket_update({
                "type": "tick_update",
                "data": {
                    "symbol": symbol,
                    "bid": float(tick_data['bid']) if tick_data['bid'] else None,
                    "ask": float(tick_data['ask']) if tick_data['ask'] else None,
                    "last": float(tick_data['last']) if tick_data['last'] else None,
                    "bid_size": tick_data['bid_size'],
                    "ask_size": tick_data['ask_size'],
                    "last_size": tick_data['last_size'],
                    "timestamp": tick_data['ts'].isoformat()
                }
            })
            
        except Exception as e:
            self.logger.error(f"Error processing tick data for {ticker.contract.symbol}: {e}")
    
    async def _subscribe_realtime_bars(self) -> List[str]:
        """Subscribe to real-time 5-second bars for all watchlist symbols."""
        if not self.connected or not self.ib_client:
            raise Exception("Not connected to TWS")
        
        subscribed = []
        
        for symbol in self.watchlist_symbols:
            if symbol in self.realtime_bar_subscriptions:
                continue  # Already subscribed
            
            try:
                contract = Stock(symbol, 'SMART', 'USD')
                
                # Subscribe to real-time 5-second bars (use underlying IB client)
                bars = self.ib_client.ib.reqRealTimeBars(
                    contract, 
                    barSize=5,  # 5-second bars
                    whatToShow='TRADES',
                    useRTH=False  # Include extended hours
                )
                
                # Store subscription
                self.realtime_bar_subscriptions[symbol] = bars
                
                # Initialize bar cache for this symbol
                if symbol not in self.bar_cache:
                    self.bar_cache[symbol] = deque(maxlen=self.bar_cache_size)
                
                # Set up event handler for this symbol's bars
                # Use a closure to properly capture the symbol
                def make_handler(sym):
                    def handler(bars_list, hasGaps):
                        asyncio.create_task(self._on_realtime_bar_update(sym, bars_list))
                    return handler
                
                bars.updateEvent += make_handler(symbol)
                
                subscribed.append(symbol)
                self.logger.info(f"Subscribed to real-time bars for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Failed to subscribe to real-time bars for {symbol}: {e}")
        
        return subscribed
    
    async def _on_realtime_bar_update(self, symbol: str, bars_list: RealTimeBarList):
        """Handle incoming real-time bar updates."""
        try:
            if not bars_list or len(bars_list) == 0:
                return
            
            # Get the latest bar
            bar = bars_list[-1]
            
            # Create bar dict
            bar_data = {
                "timestamp": bar.time,
                "open": Decimal(str(bar.open_)),
                "high": Decimal(str(bar.high)),
                "low": Decimal(str(bar.low)),
                "close": Decimal(str(bar.close)),
                "volume": int(bar.volume)
            }
            
            # Add to in-memory cache (always, for websocket clients)
            if symbol in self.bar_cache:
                self.bar_cache[symbol].append(bar_data)
            
            # Check market hours before saving to database
            if self._is_market_hours(bar_data["timestamp"]):
                # Store to database asynchronously
                asyncio.create_task(self._store_bar_to_db(symbol, bar_data))
            else:
                # Log skipped bar (at debug level to avoid spam)
                self.logger.debug(
                    f"Skipped after-hours bar for {symbol} at {bar_data['timestamp']}"
                )
            
            # Broadcast to WebSocket clients (always send, regardless of market hours)
            await self._broadcast_bar_update(symbol, bar_data)
            
            # Only log at info level if saved to DB
            if self._is_market_hours(bar_data["timestamp"]):
                log_market_data_event(
                    self.logger, "bar_update", symbol,
                    open=float(bar_data["open"]),
                    high=float(bar_data["high"]),
                    low=float(bar_data["low"]),
                    close=float(bar_data["close"]),
                    volume=bar_data["volume"]
                )
            
        except Exception as e:
            self.logger.error(f"Error processing real-time bar for {symbol}: {e}")
    
    async def _store_bar_to_db(self, symbol: str, bar_data: dict):
        """Store a bar to the database."""
        try:
            def _store_candle(session):
                stmt = insert(Candle).values(
                    symbol=symbol,
                    tf="5 secs",
                    ts=bar_data["timestamp"],
                    open=bar_data["open"],
                    high=bar_data["high"],
                    low=bar_data["low"],
                    close=bar_data["close"],
                    volume=bar_data["volume"]
                )
                update_cols = {
                    "open": bar_data["open"],
                    "high": bar_data["high"],
                    "low": bar_data["low"],
                    "close": bar_data["close"],
                    "volume": bar_data["volume"]
                }
                on_conflict_stmt = stmt.on_conflict_do_update(
                    index_elements=['symbol', 'tf', 'ts'],
                    set_=update_cols
                )
                session.execute(on_conflict_stmt)
                session.commit()
            
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: execute_with_retry(_store_candle)
            )
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: update_sync_status(
                    "market_bars",
                    source_ts=bar_data["timestamp"],
                    db_ts=datetime.now(timezone.utc)
                )
            )
            
        except Exception as e:
            self.logger.error(f"Error storing bar to database for {symbol}: {e}")
    
    async def _broadcast_bar_update(self, symbol: str, bar_data: dict):
        """Broadcast bar update to all connected WebSocket clients."""
        if not self.bar_websocket_connections:
            return
        
        message = {
            "type": "bar_update",
            "data": {
                "symbol": symbol,
                "timestamp": bar_data["timestamp"].isoformat(),
                "open": float(bar_data["open"]),
                "high": float(bar_data["high"]),
                "low": float(bar_data["low"]),
                "close": float(bar_data["close"]),
                "volume": bar_data["volume"]
            }
        }
        
        disconnected = []
        for websocket in self.bar_websocket_connections:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(websocket)
        
        # Remove disconnected websockets
        for websocket in disconnected:
            if websocket in self.bar_websocket_connections:
                self.bar_websocket_connections.remove(websocket)
    
    def _setup_watchlist_notifications(self):
        """Setup listener for watchlist update notifications."""
        def on_watchlist_update(channel: str, data: dict):
            """Handle watchlist update notifications."""
            try:
                action = data.get('action')
                symbol = data.get('symbol')
                
                if not action or not symbol:
                    return
                
                self.logger.info(f"Watchlist update: {action} {symbol}")
                
                # Schedule the update in the event loop
                asyncio.create_task(self._handle_watchlist_update(action, symbol))
                
            except Exception as e:
                self.logger.error(f"Error handling watchlist update: {e}")
        
        # Listen for watchlist updates
        listen_for_watchlist_updates(on_watchlist_update)
    
    async def _handle_watchlist_update(self, action: str, symbol: str):
        """Handle dynamic watchlist updates."""
        try:
            if action == "add":
                if symbol not in self.watchlist_symbols:
                    self.watchlist_symbols.add(symbol)
                    await self._subscribe_to_symbol(symbol)
                    self.logger.info(f"Added subscription for {symbol}")
                    
            elif action == "remove":
                if symbol in self.watchlist_symbols:
                    self.watchlist_symbols.discard(symbol)
                    await self._unsubscribe_from_symbol(symbol)
                    self.logger.info(f"Removed subscription for {symbol}")
                    
        except Exception as e:
            self.logger.error(f"Error handling watchlist update {action} {symbol}: {e}")
    
    async def _heartbeat_loop(self):
        """Heartbeat loop for health monitoring and reconnection."""
        while self.running:
            try:
                # Send heartbeat
                if self.client_id:
                    heartbeat_service_client_id(self.client_id)
                
                self.last_heartbeat = time.time()
                
                # Check TWS connection
                if not self.connected:
                    self.logger.warning("TWS connection lost, attempting reconnection")
                    try:
                        await self._connect_to_tws()
                        # Resubscribe to watchlist after reconnection
                        await self._subscribe_to_watchlist()
                    except Exception as e:
                        self.logger.error(f"Reconnection failed: {e}")
                
                # Update health status
                status = "healthy" if self.connected else "unhealthy"
                await self._update_health_status(status)
                
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(30)
    
    async def _update_health_status(self, status: str):
        """Update service health status in database."""
        try:
            def _update_health(session):
                health_record = session.query(HealthStatus).filter(
                    HealthStatus.service == "marketdata"
                ).first()
                
                if health_record:
                    health_record.status = status
                    health_record.updated_at = datetime.now(timezone.utc)
                else:
                    health_record = HealthStatus(
                        service="marketdata",
                        status=status
                    )
                    session.add(health_record)
                
                session.commit()
                return health_record
            
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: execute_with_retry(_update_health)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update health status: {e}")
    
    async def _broadcast_websocket_update(self, message: dict):
        """Broadcast update to all WebSocket connections."""
        if not self.websocket_connections:
            return
        
        disconnected = []
        for websocket in self.websocket_connections:
            try:
                await self._send_websocket_update(websocket, message)
            except Exception:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            self.websocket_connections.remove(websocket)
    
    async def _send_websocket_update(self, websocket: WebSocket, message: dict):
        """Send update to a single WebSocket connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            self.logger.debug(f"Failed to send WebSocket message: {e}")
            raise
    
    async def enable_collection(self):
        """Enable market data collection."""
        if not self.collection_enabled:
            self.collection_enabled = True
            self.logger.info("Market data collection enabled")
            
            # Resume subscriptions if we have a watchlist
            if self.watchlist_symbols and self.connected:
                await self._subscribe_to_watchlist()
                await self._subscribe_realtime_bars()
            
            # Notify WebSocket clients
            await self._broadcast_collection_status()
            
            return {"status": "enabled", "message": "Market data collection enabled"}
        else:
            return {"status": "already_enabled", "message": "Market data collection is already enabled"}
    
    async def disable_collection(self):
        """Disable market data collection."""
        if self.collection_enabled:
            self.collection_enabled = False
            self.logger.info("Market data collection disabled")
            
            # Unsubscribe from all market data
            await self._unsubscribe_all()
            
            # Notify WebSocket clients
            await self._broadcast_collection_status()
            
            return {"status": "disabled", "message": "Market data collection disabled"}
        else:
            return {"status": "already_disabled", "message": "Market data collection is already disabled"}
    
    async def get_collection_status(self):
        """Get current collection status."""
        return {
            "enabled": self.collection_enabled,
            "connected": self.connected,
            "subscriptions": len(self.subscribed_symbols),
            "watchlist_size": len(self.watchlist_symbols)
        }
    
    async def _broadcast_collection_status(self):
        """Broadcast collection status to all WebSocket clients."""
        message = {
            "type": "collection_status",
            "data": {
                "enabled": self.collection_enabled,
                "connected": self.connected,
                "subscriptions": len(self.subscribed_symbols)
            }
        }
        
        # Send to tick WebSocket clients
        disconnected = []
        for websocket in self.websocket_connections:
            try:
                await self._send_websocket_update(websocket, message)
            except Exception:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            self.websocket_connections.remove(websocket)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(self.stop())


async def main():
    """Main entry point for the market data service."""
    service = MarketDataService()
    
    try:
        # Start the service
        await service.start()
        
        # Start FastAPI server
        config = uvicorn.Config(
            service.app,
            host="0.0.0.0",
            port=8002,
            log_level="info",
            access_log=True
        )
        server = uvicorn.Server(config)
        
        # Run server
        await server.serve()
        
    except KeyboardInterrupt:
        service.logger.info("Received keyboard interrupt")
    except Exception as e:
        service.logger.error(f"Service error: {e}")
        raise
    finally:
        await service.stop()


if __name__ == "__main__":
    from src.common.debug import enable_remote_debugging
    enable_remote_debugging(5680)
    asyncio.run(main())
