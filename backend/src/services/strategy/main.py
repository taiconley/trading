#!/usr/bin/env python3
"""
Strategy Service - Live Strategy Execution

This service manages the execution of trading strategies in real-time.
It loads strategies from the database, executes them on bar-driven events,
generates signals, and places orders through the Trader service.
"""

import asyncio
import logging
import signal
import sys
import json
import aiohttp
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from decimal import Decimal
from collections import deque

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func

# Add the backend src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.config import get_settings
from common.logging import setup_logging
from common.db import get_db_session
from common.models import Strategy, Candle, Signal, Order, HealthStatus
from common.schemas import HealthCheckResponse, SignalType
from tws_bridge.client_ids import ClientIdManager
from tws_bridge.ib_client import EnhancedIBClient
from strategy_lib import (
    get_strategy_registry, 
    load_strategies_from_directory,
    create_strategy_from_db_config,
    BaseStrategy,
    StrategyState,
    validate_bars_dataframe
)

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manage WebSocket connections for real-time strategy updates."""
    
    def __init__(self):
        self.connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.connections)}")
    
    async def broadcast(self, data: dict):
        if not self.connections:
            return
        
        disconnected = set()
        for connection in self.connections:
            try:
                await connection.send_json(data)
            except Exception:
                disconnected.add(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)


class StrategyRunner:
    """Manages individual strategy execution."""
    
    def __init__(self, strategy: BaseStrategy, db_config: Dict[str, Any]):
        self.strategy = strategy
        self.db_config = db_config
        self.last_bar_time: Dict[str, datetime] = {}
        self.is_running = False
        
    async def start(self, instruments: Dict[str, Any]):
        """Start the strategy."""
        try:
            await self.strategy.on_start(instruments)
            self.is_running = True
            logger.info(f"Started strategy: {self.strategy.config.strategy_id}")
        except Exception as e:
            logger.error(f"Failed to start strategy {self.strategy.config.strategy_id}: {e}")
            raise
    
    async def stop(self):
        """Stop the strategy."""
        try:
            if self.is_running:
                await self.strategy.on_stop()
                self.is_running = False
                logger.info(f"Stopped strategy: {self.strategy.config.strategy_id}")
        except Exception as e:
            logger.error(f"Error stopping strategy {self.strategy.config.strategy_id}: {e}")
    
    async def process_bars(self, symbol: str, timeframe: str, bars_df: pd.DataFrame) -> List[Any]:
        """Process new bar data through the strategy."""
        if not self.is_running:
            return []
        
        try:
            # Check if this is new data
            if bars_df.empty:
                return []
            
            latest_time = bars_df['timestamp'].iloc[-1]
            if isinstance(latest_time, str):
                latest_time = pd.to_datetime(latest_time)
            
            last_processed = self.last_bar_time.get(symbol)
            if last_processed and latest_time <= last_processed:
                return []  # No new data
            
            # Update last processed time
            self.last_bar_time[symbol] = latest_time
            
            # Validate bars dataframe
            if not validate_bars_dataframe(bars_df):
                logger.warning(f"Invalid bars dataframe for {symbol}")
                return []
            
            # Call strategy
            signals = await self.strategy.on_bar(symbol, timeframe, bars_df)
            
            return signals or []
            
        except Exception as e:
            logger.error(f"Error processing bars for strategy {self.strategy.config.strategy_id}, symbol {symbol}: {e}")
            return []


class StrategyService:
    """Main strategy service class."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client_id_manager = ClientIdManager()
        self.client_id = None
        self.ib_client = None
        self.db_session_factory = get_db_session
        self.websocket_manager = WebSocketManager()
        
        # Strategy management
        self.strategy_runners: Dict[str, StrategyRunner] = {}
        self.last_strategy_reload = datetime.now(timezone.utc)
        self.shutdown_event = asyncio.Event()
        
        # FastAPI app
        self.app = FastAPI(title="Strategy Service", version="1.0.0")
        self._setup_routes()
        
        # Execution control
        self.running = False
        self.bar_processing_interval = 5  # seconds - check strategies every 5 seconds
        
        # Real-time bar cache (in-memory)
        self.bar_cache: Dict[str, deque] = {}  # symbol -> deque of bar dicts
        self.bar_cache_size = 2000  # Keep last 2000 bars per symbol in memory
        self.marketdata_ws_client = None  # WebSocket connection to MarketData service
        self.marketdata_reconnect_delay = 5  # seconds
        
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/healthz")
        async def health_check():
            """Health check endpoint."""
            try:
                health_status = "healthy" if self.running else "starting"
                
                strategy_info = {}
                for strategy_id, runner in self.strategy_runners.items():
                    strategy_info[strategy_id] = {
                        "running": runner.is_running,
                        "state": runner.strategy.get_state().value,
                        "metrics": runner.strategy.get_metrics().dict()
                    }
                
                return HealthCheckResponse(
                    service="strategy",
                    status=health_status,
                    timestamp=datetime.now(timezone.utc),
                    details={
                        "tws_connected": self.ib_client.state.connected if self.ib_client else False,
                        "client_id": self.client_id,
                        "active_strategies": len(self.strategy_runners),
                        "strategies": strategy_info
                    }
                )
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return HealthCheckResponse(
                    service="strategy",
                    status="unhealthy",
                    timestamp=datetime.now(timezone.utc),
                    details={"error": str(e)}
                )
        
        @self.app.get("/strategies")
        async def list_strategies():
            """List active strategies."""
            try:
                strategies = []
                for strategy_id, runner in self.strategy_runners.items():
                    strategy_info = {
                        "strategy_id": strategy_id,
                        "name": runner.strategy.config.name,
                        "symbols": runner.strategy.config.symbols,
                        "enabled": runner.strategy.config.enabled,
                        "running": runner.is_running,
                        "state": runner.strategy.get_state().value,
                        "metrics": runner.strategy.get_metrics().dict()
                    }
                    # Add detailed state if strategy supports it
                    if hasattr(runner.strategy, 'get_strategy_state'):
                        try:
                            strategy_info["state_details"] = runner.strategy.get_strategy_state()
                        except Exception as e:
                            logger.warning(f"Failed to get strategy state for {strategy_id}: {e}")
                    strategies.append(strategy_info)
                
                return {"strategies": strategies, "total": len(strategies)}
            except Exception as e:
                logger.error(f"Failed to list strategies: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/strategies/{strategy_id}/reload")
        async def reload_strategy(strategy_id: str):
            """Reload a specific strategy."""
            try:
                await self._reload_strategy(strategy_id)
                return {"message": f"Strategy {strategy_id} reloaded successfully"}
            except Exception as e:
                logger.error(f"Failed to reload strategy {strategy_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/strategies/reload-all")
        async def reload_all_strategies():
            """Reload all strategies from database."""
            try:
                await self._reload_all_strategies()
                return {"message": "All strategies reloaded successfully"}
            except Exception as e:
                logger.error(f"Failed to reload all strategies: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time strategy updates."""
            await self.websocket_manager.connect(websocket)
            try:
                while True:
                    # Send periodic updates
                    await asyncio.sleep(5)
                    
                    update_data = {
                        "type": "strategy_update",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "data": {
                            "active_strategies": len(self.strategy_runners),
                            "service_status": "healthy" if self.running else "starting"
                        }
                    }
                    
                    await websocket.send_json(update_data)
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)
    
    async def initialize(self):
        """Initialize the strategy service."""
        try:
            logger.info("Initializing strategy service...")
            
            # Load strategies into registry
            registry = get_strategy_registry()
            loaded_count = load_strategies_from_directory()
            logger.info(f"Loaded {loaded_count} strategies into registry")
            
            # Get client ID (strategies can use range 15-29)
            self.client_id = self.client_id_manager.get_service_client_id("strategy")
            logger.info(f"Allocated client ID: {self.client_id}")
            
            # Initialize IB client (optional - some strategies may not need TWS)
            if self.settings.tws.host:
                self.ib_client = EnhancedIBClient(
                    client_id=self.client_id,
                    host=self.settings.tws.host,
                    port=self.settings.tws.port
                )
                
                try:
                    await self.ib_client.connect()
                    logger.info("Connected to TWS successfully")
                except Exception as e:
                    logger.warning(f"Failed to connect to TWS: {e}. Strategies will run without TWS connection.")
            
            # Load strategies from database
            await self._reload_all_strategies()
            
            # Update health status
            await self._update_health_status("healthy")
            
            logger.info("Strategy service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize strategy service: {e}")
            await self._update_health_status("unhealthy")
            raise
    
    async def start(self):
        """Start the strategy service."""
        try:
            print("DEBUG start(): Calling initialize()...", flush=True)
            await self.initialize()
            print("DEBUG start(): initialize() completed", flush=True)
            
            # Backfill bar cache with historical data
            print("DEBUG start(): Calling _backfill_bar_cache()...", flush=True)
            await self._backfill_bar_cache()
            print("DEBUG start(): _backfill_bar_cache() completed", flush=True)
            
            # Start main execution loop
            self.running = True
            
            # Start background tasks
            print("DEBUG start(): Starting background tasks...", flush=True)
            asyncio.create_task(self._strategy_execution_loop())
            asyncio.create_task(self._strategy_reload_loop())
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._connect_to_marketdata_stream())  # Real-time bars
            print("DEBUG start(): Background tasks started", flush=True)
            
            logger.info("Strategy service started successfully")
            print("DEBUG start(): All done!", flush=True)
            
        except Exception as e:
            print(f"DEBUG start(): EXCEPTION: {e}", flush=True)
            logger.error(f"Failed to start strategy service: {e}")
            await self._update_health_status("unhealthy")
            raise
    
    async def stop(self):
        """Stop the strategy service."""
        try:
            logger.info("Stopping strategy service...")
            self.running = False
            self.shutdown_event.set()
            
            # Stop all strategies
            for runner in self.strategy_runners.values():
                await runner.stop()
            
            # Disconnect from TWS
            if self.ib_client:
                await self.ib_client.disconnect()
            
            # Release client ID
            if self.client_id:
                self.client_id_manager.release_service_client_id("strategy", self.client_id)
            
            await self._update_health_status("stopping")
            logger.info("Strategy service stopped")
            
        except Exception as e:
            logger.error(f"Error stopping strategy service: {e}")
    
    async def _strategy_execution_loop(self):
        """Main strategy execution loop."""
        print("DEBUG exec_loop: Starting strategy execution loop", flush=True)
        logger.info("Starting strategy execution loop")
        iteration = 0
        
        while self.running and not self.shutdown_event.is_set():
            try:
                iteration += 1
                if iteration % 6 == 1:  # Log every ~30 seconds
                    print(f"DEBUG exec_loop: Iteration {iteration}, processing {len(self.strategy_runners)} strategies", flush=True)
                
                # Process each active strategy
                for strategy_id, runner in list(self.strategy_runners.items()):
                    if not runner.is_running or not runner.strategy.config.enabled:
                        continue
                    
                    # Check if strategy supports multi-symbol mode
                    if hasattr(runner.strategy, 'supports_multi_symbol') and runner.strategy.supports_multi_symbol:
                        # Multi-symbol strategy: process all symbols together
                        await self._process_multi_symbol_strategy(runner)
                    else:
                        # Single-symbol strategy: process each symbol individually
                        for symbol in runner.strategy.config.symbols:
                            await self._process_symbol_for_strategy(runner, symbol)
                
                # Wait before next iteration
                await asyncio.sleep(self.bar_processing_interval)
                
            except Exception as e:
                logger.error(f"Error in strategy execution loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
        
        logger.info("Strategy execution loop stopped")
    
    async def _process_symbol_for_strategy(self, runner: StrategyRunner, symbol: str):
        """Process a symbol for a specific strategy."""
        try:
            # Get latest bars from in-memory cache (real-time bars)
            timeframe = runner.strategy.config.bar_timeframe
            lookback = runner.strategy.config.lookback_periods
            
            # Use in-memory cache instead of database for real-time data
            if symbol in self.bar_cache and len(self.bar_cache[symbol]) > 0:
                print(f"DEBUG process_symbol: Using {len(self.bar_cache[symbol])} cached bars for {symbol}", flush=True)
            
            bars_df = await self._get_latest_bars(symbol, timeframe, lookback)
            
            if bars_df.empty:
                return
            
            # Process through strategy
            signals = await runner.process_bars(symbol, timeframe, bars_df)
            
            # Handle any generated signals
            if signals:
                await self._handle_strategy_signals(runner, signals)
                
        except Exception as e:
            logger.error(f"Error processing symbol {symbol} for strategy {runner.strategy.config.strategy_id}: {e}")
    
    async def _process_multi_symbol_strategy(self, runner: StrategyRunner):
        """Process a multi-symbol strategy (e.g., pairs trading)."""
        try:
            timeframe = runner.strategy.config.bar_timeframe
            lookback = runner.strategy.config.lookback_periods
            symbols = runner.strategy.config.symbols
            
            print(f"DEBUG multi_symbol: Processing {len(symbols)} symbols for {runner.strategy.config.strategy_id}", flush=True)
            
            # Fetch bars for all symbols
            bars_data = {}
            for symbol in symbols:
                bars_df = await self._get_latest_bars(symbol, timeframe, lookback)
                if not bars_df.empty:
                    bars_data[symbol] = bars_df
            
            if len(bars_data) < 2:
                # Need at least 2 symbols for pairs trading
                logger.debug(f"Insufficient data for multi-symbol strategy: {len(bars_data)}/{len(symbols)} symbols")
                return
            
            # Log bar counts for debugging
            bar_counts_str = ', '.join([f"{sym}:{len(df)}" for sym, df in list(bars_data.items())[:5]])
            print(f"DEBUG multi_symbol: Calling on_bar_multi() with {len(bars_data)} symbols, bar counts: {bar_counts_str}...", flush=True)
            
            # Call strategy's multi-symbol handler
            signals = await runner.strategy.on_bar_multi(symbols, timeframe, bars_data)
            
            signal_count = len(signals) if signals else 0
            print(f"DEBUG multi_symbol: Received {signal_count} signals", flush=True)
            if signal_count > 0:
                logger.info(
                    f"Strategy {runner.strategy.config.strategy_id} generated {signal_count} signals: "
                    f"{[f'{s.signal_type} {s.symbol}' for s in signals]}"
                )
            
            # Handle any generated signals
            if signals:
                await self._handle_strategy_signals(runner, signals)
                
        except Exception as e:
            logger.error(f"Error processing multi-symbol strategy {runner.strategy.config.strategy_id}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    async def _get_latest_bars(self, symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
        """Get latest bars from in-memory cache (real-time) or database (fallback)."""
        try:
            # Try to get from in-memory cache first (real-time)
            if symbol in self.bar_cache and len(self.bar_cache[symbol]) > 0:
                cached_bars = list(self.bar_cache[symbol])
                
                # Get last N bars from cache
                recent_bars = cached_bars[-lookback:] if len(cached_bars) >= lookback else cached_bars
                
                if recent_bars:
                    logger.debug(f"Using {len(recent_bars)} cached bars for {symbol}")
                    return pd.DataFrame(recent_bars)
            
            # Fallback to database if cache is empty or insufficient
            logger.debug(f"Fetching bars from database for {symbol} (cache miss)")
            with self.db_session_factory() as db:
                # Query latest candles
                candles = db.query(Candle).filter(
                    and_(
                        Candle.symbol == symbol,
                        Candle.tf == timeframe
                    )
                ).order_by(desc(Candle.ts)).limit(lookback).all()
                
                if not candles:
                    return pd.DataFrame()
                
                # Convert to DataFrame (reverse to get chronological order)
                candles.reverse()
                
                data = []
                for candle in candles:
                    data.append({
                        'timestamp': candle.ts,
                        'open': float(candle.open),
                        'high': float(candle.high),
                        'low': float(candle.low),
                        'close': float(candle.close),
                        'volume': int(candle.volume)
                    })
                
                return pd.DataFrame(data)
                
        except Exception as e:
            logger.error(f"Failed to get latest bars for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _connect_to_marketdata_stream(self):
        """Connect to MarketData service WebSocket for real-time bars."""
        print("DEBUG _connect_to_marketdata_stream: Method called!", flush=True)
        # MarketData service WebSocket endpoint (hardcoded as per Docker Compose setup)
        ws_url = "ws://backend-marketdata:8002/ws/bars"
        print(f"DEBUG _connect_to_marketdata_stream: Will connect to {ws_url}", flush=True)
        
        while self.running and not self.shutdown_event.is_set():
            try:
                print(f"DEBUG _connect_to_marketdata_stream: Attempting connection...", flush=True)
                logger.info(f"Connecting to MarketData WebSocket at {ws_url}...")
                
                async with aiohttp.ClientSession() as session:
                    print(f"DEBUG _connect: Created ClientSession", flush=True)
                    async with session.ws_connect(ws_url) as ws:
                        print(f"DEBUG _connect: WebSocket connected successfully!", flush=True)
                        self.marketdata_ws_client = ws
                        logger.info("Connected to MarketData real-time bar stream")
                        
                        # Listen for incoming bar updates
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    data = json.loads(msg.data)
                                    await self._handle_bar_update(data)
                                except json.JSONDecodeError as e:
                                    logger.error(f"Failed to parse WebSocket message: {e}")
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                logger.error(f"WebSocket error: {ws.exception()}")
                                break
                            elif msg.type == aiohttp.WSMsgType.CLOSED:
                                logger.warning("WebSocket connection closed by server")
                                break
                        
            except aiohttp.ClientError as e:
                print(f"DEBUG _connect: ClientError: {e}", flush=True)
                logger.error(f"MarketData WebSocket connection error: {e}")
            except Exception as e:
                print(f"DEBUG _connect: Exception: {e}", flush=True)
                logger.error(f"Unexpected error in MarketData WebSocket: {e}")
            finally:
                self.marketdata_ws_client = None
                
                if self.running and not self.shutdown_event.is_set():
                    logger.info(f"Reconnecting to MarketData in {self.marketdata_reconnect_delay}s...")
                    await asyncio.sleep(self.marketdata_reconnect_delay)
    
    async def _handle_bar_update(self, data: Dict[str, Any]):
        """Handle incoming bar update from MarketData service."""
        try:
            if data.get("type") != "bar_update":
                return
            
            bar_data = data.get("data", {})
            symbol = bar_data.get("symbol")
            
            if not symbol:
                return
            
            # Initialize cache for symbol if needed
            if symbol not in self.bar_cache:
                self.bar_cache[symbol] = deque(maxlen=self.bar_cache_size)
                logger.info(f"Initialized bar cache for {symbol}")
            
            # Add bar to cache
            bar = {
                'timestamp': pd.to_datetime(bar_data['timestamp']),
                'open': float(bar_data['open']),
                'high': float(bar_data['high']),
                'low': float(bar_data['low']),
                'close': float(bar_data['close']),
                'volume': int(bar_data['volume'])
            }
            
            self.bar_cache[symbol].append(bar)
            logger.debug(f"Cached new bar for {symbol} at {bar['timestamp']} (cache size: {len(self.bar_cache[symbol])})")
            
        except Exception as e:
            logger.error(f"Error handling bar update: {e}")
    
    async def _backfill_bar_cache(self):
        """Backfill in-memory cache with recent historical bars on startup."""
        try:
            print(f"DEBUG _backfill: strategy_runners count = {len(self.strategy_runners)}", flush=True)
            # Get all symbols from active strategies
            all_symbols = set()
            for runner in self.strategy_runners.values():
                if runner.strategy.config.enabled:
                    all_symbols.update(runner.strategy.config.symbols)
            
            print(f"DEBUG _backfill: all_symbols = {all_symbols}", flush=True)
            if not all_symbols:
                logger.info("No active strategies, skipping bar cache backfill")
                print("DEBUG _backfill: EARLY RETURN - no symbols", flush=True)
                return
            
            logger.info(f"Backfilling bar cache for {len(all_symbols)} symbols...")
            print(f"DEBUG _backfill: About to query database for {len(all_symbols)} symbols", flush=True)
            
            with self.db_session_factory() as db:
                for symbol in all_symbols:
                    # Fetch recent bars (enough for max lookback + buffer)
                    candles = db.query(Candle).filter(
                        and_(
                            Candle.symbol == symbol,
                            Candle.tf == "5 secs"  # Match MarketData service format
                        )
                    ).order_by(desc(Candle.ts)).limit(self.bar_cache_size).all()
                    
                    if candles:
                        # Initialize cache
                        self.bar_cache[symbol] = deque(maxlen=self.bar_cache_size)
                        
                        # Add bars in chronological order
                        candles.reverse()
                        for candle in candles:
                            self.bar_cache[symbol].append({
                                'timestamp': candle.ts,
                                'open': float(candle.open),
                                'high': float(candle.high),
                                'low': float(candle.low),
                                'close': float(candle.close),
                                'volume': int(candle.volume)
                            })
                        
                        logger.info(f"Backfilled {len(self.bar_cache[symbol])} bars for {symbol}")
                        print(f"DEBUG _backfill: Backfilled {len(self.bar_cache[symbol])} bars for {symbol}", flush=True)
            
            logger.info("Bar cache backfill complete")
            print(f"DEBUG _backfill: Bar cache backfill complete! Cache has {len(self.bar_cache)} symbols", flush=True)
            
        except Exception as e:
            logger.error(f"Failed to backfill bar cache: {e}")
            print(f"DEBUG _backfill: EXCEPTION: {e}", flush=True)
    
    async def _backfill_symbols(self, symbols: List[str]):
        """Backfill bar cache for specific symbols (used when strategies are reloaded)."""
        try:
            if not symbols:
                return
            
            logger.info(f"Backfilling bar cache for {len(symbols)} symbols...")
            
            with self.db_session_factory() as db:
                for symbol in symbols:
                    # Skip if already cached
                    if symbol in self.bar_cache and len(self.bar_cache[symbol]) > 0:
                        continue
                    
                    # Fetch recent bars
                    candles = db.query(Candle).filter(
                        and_(
                            Candle.symbol == symbol,
                            Candle.tf == "5 secs"
                        )
                    ).order_by(desc(Candle.ts)).limit(self.bar_cache_size).all()
                    
                    if candles:
                        # Initialize cache
                        self.bar_cache[symbol] = deque(maxlen=self.bar_cache_size)
                        
                        # Add bars in chronological order
                        candles.reverse()
                        for candle in candles:
                            self.bar_cache[symbol].append({
                                'timestamp': candle.ts,
                                'open': float(candle.open),
                                'high': float(candle.high),
                                'low': float(candle.low),
                                'close': float(candle.close),
                                'volume': int(candle.volume)
                            })
                        
                        logger.info(f"Backfilled {len(self.bar_cache[symbol])} bars for {symbol}")
            
        except Exception as e:
            logger.error(f"Error backfilling symbols: {e}")
    
    async def _handle_strategy_signals(self, runner: StrategyRunner, signals: List[Any]):
        """Handle signals generated by a strategy."""
        if signals:
            logger.info(f"Received {len(signals)} signals from strategy {runner.strategy.config.strategy_id}")
        
        for signal in signals:
            try:
                logger.info(
                    f"Processing signal: {signal.signal_type} {signal.symbol} "
                    f"qty={signal.quantity} price={signal.price} "
                    f"strategy={runner.strategy.config.strategy_id}"
                )
                
                # Store signal in database
                await self._store_signal(signal, runner.strategy.config.strategy_id)
                logger.debug(f"Stored signal in database: {signal.symbol} {signal.signal_type}")
                
                # Check if strategy is enabled and risk limits allow trading
                if runner.strategy.config.enabled:
                    should_execute = await self._should_execute_signal(signal)
                    logger.info(
                        f"Signal execution check: enabled={runner.strategy.config.enabled}, "
                        f"should_execute={should_execute}, signal_type={signal.signal_type}"
                    )
                    
                    if should_execute:
                        # Place order through Trader service
                        logger.info(f"Placing order for signal: {signal.symbol} {signal.signal_type}")
                        await self._place_order_for_signal(signal, runner.strategy.config.strategy_id)
                    else:
                        logger.warning(
                            f"Signal not executed: enabled={runner.strategy.config.enabled}, "
                            f"should_execute={should_execute}"
                        )
                else:
                    logger.warning(f"Strategy {runner.strategy.config.strategy_id} is disabled, skipping order placement")
                
                # Broadcast signal via WebSocket
                await self._broadcast_signal(signal, runner.strategy.config.strategy_id)
                
            except Exception as e:
                logger.error(f"Error handling signal: {e}", exc_info=True)
    
    async def _store_signal(self, signal: Any, strategy_id: str):
        """Store signal in database."""
        try:
            with self.db_session_factory() as db:
                # Handle both Enum and string signal_type
                signal_type_str = signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type)
                db_signal = Signal(
                    strategy_id=strategy_id,
                    symbol=signal.symbol,
                    signal_type=signal_type_str,
                    strength=signal.strength,
                    ts=signal.timestamp,
                    meta_json=signal.metadata
                )
                db.add(db_signal)
                db.commit()
                
        except Exception as e:
            logger.error(f"Failed to store signal in database: {e}")
    
    async def _should_execute_signal(self, signal: Any) -> bool:
        """Check if signal should be executed based on risk limits."""
        # This is a simplified check - in practice would integrate with risk management
        return signal.signal_type in [SignalType.BUY, SignalType.SELL]
    
    async def _place_order_for_signal(self, signal: Any, strategy_id: str):
        """Place order through Trader service REST API.
        
        This follows the exact same order placement logic as tested in test_order_placement.sh.
        The order request format matches exactly: symbol, side, quantity, order_type, 
        limit_price, time_in_force, strategy_id.
        """
        try:
            # Determine order side - must be "BUY" or "SELL" string
            order_side = "BUY" if signal.signal_type == SignalType.BUY else "SELL"
            
            # Determine order type - use signal's execution_type if provided, otherwise default logic
            has_price = signal.price is not None and float(signal.price) > 0
            
            if signal.execution_type:
                # Use execution type from signal (e.g., "ADAPTIVE", "PEG BEST", "PEG MID")
                order_type = signal.execution_type
            else:
                # Default: use LIMIT if price provided, otherwise MARKET
                # Note: pairs_trading always provides price, so orders will be LIMIT by default
                order_type = "LMT" if has_price else "MKT"
            
            # Build order request - format matches exactly what we tested
            order_request = {
                "symbol": signal.symbol,
                "side": order_side,
                "quantity": signal.quantity or 100,
                "order_type": order_type,
                "limit_price": float(signal.price) if has_price else None,
                "time_in_force": "DAY",
                "strategy_id": strategy_id
            }
            
            # Add algorithm parameters if provided
            if signal.algo_strategy:
                order_request["algo_strategy"] = signal.algo_strategy
            if signal.algo_params:
                order_request["algo_params"] = signal.algo_params
            
            # Send to Trader service (same endpoint as test)
            trader_url = f"http://backend-trader:8004/orders"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(trader_url, json=order_request) as response:
                    if response.status == 200:
                        order_data = await response.json()
                        # Response uses 'id' field, not 'order_id'
                        order_id = order_data.get('id') or order_data.get('order_id')
                        logger.info(
                            f"Order placed successfully: ID={order_id}, "
                            f"symbol={signal.symbol}, side={order_side}, "
                            f"qty={order_request['quantity']}, type={order_type}"
                        )
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Failed to place order: HTTP {response.status} - {error_text}. "
                            f"Request: symbol={signal.symbol}, side={order_side}, "
                            f"quantity={order_request['quantity']}, order_type={order_type}"
                        )
                        
        except Exception as e:
            logger.error(
                f"Error placing order for signal {signal.symbol}: {e}",
                exc_info=True
            )
    
    async def _broadcast_signal(self, signal: Any, strategy_id: str):
        """Broadcast signal via WebSocket."""
        try:
            # Handle both Enum and string signal_type
            signal_type_str = signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type)
            signal_data = {
                "type": "signal_generated",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "strategy_id": strategy_id,
                    "symbol": signal.symbol,
                    "signal_type": signal_type_str,
                    "strength": float(signal.strength) if signal.strength else None,
                    "price": float(signal.price) if signal.price else None,
                    "quantity": signal.quantity,
                    "metadata": signal.metadata
                }
            }
            
            await self.websocket_manager.broadcast(signal_data)
            
        except Exception as e:
            logger.error(f"Error broadcasting signal: {e}")
    
    async def _strategy_reload_loop(self):
        """Periodically check for strategy configuration changes."""
        while self.running and not self.shutdown_event.is_set():
            try:
                # Check every 5 minutes
                await asyncio.sleep(300)
                
                # Check if strategies need reloading
                if await self._check_strategies_need_reload():
                    logger.info("Strategy configurations changed, reloading...")
                    await self._reload_all_strategies()
                    
            except Exception as e:
                logger.error(f"Error in strategy reload loop: {e}")
    
    async def _check_strategies_need_reload(self) -> bool:
        """Check if strategies need reloading based on database changes."""
        try:
            with self.db_session_factory() as db:
                # Check for any strategy changes (created_at or enabled status)
                # We need to check both because enabling/disabling doesn't change created_at
                latest_created = db.query(func.max(Strategy.created_at)).scalar()
                
                # Also check if any strategy's enabled status has changed
                # by comparing current enabled strategies with what we have loaded
                current_enabled_strategies = set()
                for runner in self.strategy_runners.values():
                    if runner.strategy.config.enabled:
                        current_enabled_strategies.add(runner.strategy.config.strategy_id)
                
                db_enabled_strategies = set()
                db_strategies = db.query(Strategy).filter(Strategy.enabled == True).all()
                for strategy in db_strategies:
                    db_enabled_strategies.add(strategy.strategy_id)
                
                # Check if there are differences in enabled strategies
                if current_enabled_strategies != db_enabled_strategies:
                    logger.info(f"Strategy enabled status changed. Current: {current_enabled_strategies}, DB: {db_enabled_strategies}")
                    return True
                
                # Also check for new strategies (created_at changes)
                if latest_created and latest_created > self.last_strategy_reload:
                    return True
                    
                return False
                
        except Exception as e:
            logger.error(f"Error checking strategy reload need: {e}")
            return False
    
    async def _reload_all_strategies(self):
        """Reload all strategies from database."""
        try:
            # Stop existing strategies
            for runner in self.strategy_runners.values():
                await runner.stop()
            
            self.strategy_runners.clear()
            
            # Load strategies from database
            with self.db_session_factory() as db:
                db_strategies = db.query(Strategy).filter(Strategy.enabled == True).all()
                
                for db_strategy in db_strategies:
                    try:
                        # Convert to dict
                        strategy_config = {
                            'strategy_id': db_strategy.strategy_id,
                            'name': db_strategy.name,
                            'enabled': db_strategy.enabled,
                            'symbols': [],  # Will be populated from params_json
                            'params_json': db_strategy.params_json or {}
                        }
                        
                        # Extract symbols from params_json
                        if isinstance(db_strategy.params_json, dict):
                            strategy_config['symbols'] = db_strategy.params_json.get('symbols', [])
                            strategy_config.update(db_strategy.params_json)
                        
                        # Create strategy instance
                        strategy_instance = create_strategy_from_db_config(strategy_config)
                        
                        # Create runner
                        runner = StrategyRunner(strategy_instance, strategy_config)
                        
                        # Start strategy
                        instruments = {}  # Could be populated with contract info if needed
                        await runner.start(instruments)
                        
                        # Store runner
                        self.strategy_runners[db_strategy.strategy_id] = runner
                        
                        # Backfill bar cache for new strategy symbols if service is running
                        if self.running and strategy_config.get('symbols'):
                            await self._backfill_symbols(strategy_config['symbols'])
                        
                        logger.info(f"Loaded strategy: {db_strategy.strategy_id}")
                        
                    except Exception as e:
                        logger.error(f"Failed to load strategy {db_strategy.strategy_id}: {e}")
            
            self.last_strategy_reload = datetime.now(timezone.utc)
            logger.info(f"Loaded {len(self.strategy_runners)} strategies")
            
        except Exception as e:
            logger.error(f"Failed to reload strategies: {e}")
            raise
    
    async def _reload_strategy(self, strategy_id: str):
        """Reload a specific strategy."""
        try:
            # Stop existing strategy if running
            if strategy_id in self.strategy_runners:
                await self.strategy_runners[strategy_id].stop()
                del self.strategy_runners[strategy_id]
            
            # Load from database
            with self.db_session_factory() as db:
                db_strategy = db.query(Strategy).filter(Strategy.strategy_id == strategy_id).first()
                
                if not db_strategy:
                    raise ValueError(f"Strategy {strategy_id} not found in database")
                
                if not db_strategy.enabled:
                    logger.info(f"Strategy {strategy_id} is disabled, not loading")
                    return
                
                # Create strategy instance (same logic as _reload_all_strategies)
                strategy_config = {
                    'strategy_id': db_strategy.strategy_id,
                    'name': db_strategy.name,
                    'enabled': db_strategy.enabled,
                    'symbols': [],
                    'params_json': db_strategy.params_json or {}
                }
                
                if isinstance(db_strategy.params_json, dict):
                    strategy_config['symbols'] = db_strategy.params_json.get('symbols', [])
                    strategy_config.update(db_strategy.params_json)
                
                strategy_instance = create_strategy_from_db_config(strategy_config)
                runner = StrategyRunner(strategy_instance, strategy_config)
                
                instruments = {}
                await runner.start(instruments)
                
                self.strategy_runners[strategy_id] = runner
                logger.info(f"Reloaded strategy: {strategy_id}")
                
        except Exception as e:
            logger.error(f"Failed to reload strategy {strategy_id}: {e}")
            raise
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat updates."""
        while self.running and not self.shutdown_event.is_set():
            try:
                await self._update_health_status("healthy")
                await asyncio.sleep(30)  # Every 30 seconds
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)
    
    async def _update_health_status(self, status: str):
        """Update health status in database."""
        try:
            with self.db_session_factory() as db:
                health_record = db.query(HealthStatus).filter(
                    HealthStatus.service == "strategy"
                ).first()
                
                if health_record:
                    health_record.status = status
                    health_record.updated_at = datetime.now(timezone.utc)
                else:
                    health_record = HealthStatus(
                        service="strategy",
                        status=status,
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(health_record)
                
                db.commit()
                
        except Exception as e:
            logger.error(f"Failed to update health status: {e}")


async def main():
    """Main entry point."""
    # Setup logging first
    print("=" * 80, flush=True)
    print("DEBUG: main() function called!", flush=True)
    print("=" * 80, flush=True)
    setup_logging("strategy")
    logger.info("Starting Strategy Service...")
    print("DEBUG: After setup_logging and logger.info", flush=True)
    
    # Create service instance
    print("DEBUG: Creating StrategyService instance...", flush=True)
    strategy_service = StrategyService()
    print("DEBUG: StrategyService instance created", flush=True)
    
    # Setup signal handlers
    shutdown_event = asyncio.Event()
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        shutdown_event.set()
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start service
        print("DEBUG: About to call strategy_service.start()...", flush=True)
        await strategy_service.start()
        print("DEBUG: strategy_service.start() completed!", flush=True)
        
        # Start FastAPI server
        config = uvicorn.Config(
            app=strategy_service.app,
            host="0.0.0.0",
            port=8005,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        # Create server task
        server_task = asyncio.create_task(server.serve())
        
        # Wait for either server completion or shutdown signal
        done, pending = await asyncio.wait(
            [server_task, asyncio.create_task(shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
        
        # Stop the server
        server.should_exit = True
        
    except Exception as e:
        logger.error(f"Strategy service failed: {e}")
        raise
    finally:
        await strategy_service.stop()


if __name__ == "__main__":
    asyncio.run(main())