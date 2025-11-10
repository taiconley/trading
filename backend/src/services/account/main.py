#!/usr/bin/env python3
"""
Account Service - Real-time TWS Account Data Streaming
Streams account values, positions, and P&L data to database and WebSocket clients.
"""
import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Set, Dict, Any, Optional
from decimal import Decimal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from ib_insync import IB
from sqlalchemy import text, and_, func

# Add the backend src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.config import get_settings
from common.logging import setup_logging
from common.db import get_db_session
from common.models import AccountSummary, Position, Account, Symbol

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections for real-time data broadcasting"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, data: dict):
        """Broadcast data to all connected WebSocket clients"""
        if not self.active_connections:
            return
        
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

class AccountService:
    """Account Service for streaming TWS account data"""
    
    def __init__(self):
        self.settings = get_settings()
        self.websocket_manager = WebSocketManager()
        self.ib = IB()
        self.account = None
        self.running = False
        
        # Current account state (in-memory cache)
        self.account_values: Dict[str, Any] = {}
        self.current_positions: Dict[str, Dict[str, Any]] = {}
        self.pnl_data: Dict[str, Any] = {}
        
        # Counters for monitoring
        self.stats = {
            'account_updates': 0,
            'position_updates': 0,
            'pnl_updates': 0,
            'summary_updates': 0,
            'last_update': None
        }
    
    async def start_tws_connection(self):
        """Start TWS connection in background"""
        try:
            logger.info("🚀 Starting TWS connection...")
            
            # Connect to TWS
            logger.info(f"🔄 Connecting to TWS at {self.settings.tws.host}:{self.settings.tws.port}")
            await self.ib.connectAsync(
                host=self.settings.tws.host,
                port=self.settings.tws.port,
                clientId=11  # Account service client ID
            )
            logger.info("✅ Connected to TWS successfully!")
            
            # Get managed accounts
            accounts = self.ib.managedAccounts()
            if not accounts:
                raise RuntimeError("No managed accounts found")
            
            self.account = accounts[0]
            logger.info(f"✅ Using account: {self.account}")
            
            # Set up event handlers for streaming data
            self.setup_event_handlers()
            
            # Start streaming subscriptions
            await self.start_streaming()
            
            self.running = True
            logger.info("🎯 Account Service TWS connection fully operational!")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to TWS: {e}")
            # Don't stop the service, just log the error
            logger.info("🌐 FastAPI server will continue running without TWS connection")
    
    def setup_event_handlers(self):
        """Set up TWS event handlers for real-time data streaming"""
        logger.info("📡 Setting up TWS event handlers...")
        
        # Account value updates (balance, margin, etc.)
        def on_account_value(account_value):
            if account_value.account == self.account:
                asyncio.create_task(self.handle_account_value_update(account_value))
        
        # Position updates
        def on_position_update(position):
            logger.info(f"📍 Position event received: {position.contract.symbol} for account {position.account} (our account: {self.account})")
            if position.account == self.account:
                asyncio.create_task(self.handle_position_update(position))
            else:
                logger.warning(f"⚠️ Position for different account: {position.account} != {self.account}")
        
        # P&L updates
        def on_pnl_update(pnl):
            if hasattr(pnl, 'account') and pnl.account == self.account:
                asyncio.create_task(self.handle_pnl_update(pnl))
        
        # Account summary updates
        def on_account_summary_update(account_summary):
            if account_summary.account == self.account:
                asyncio.create_task(self.handle_account_summary_update(account_summary))
        
        # Connect event handlers
        self.ib.accountValueEvent += on_account_value
        self.ib.positionEvent += on_position_update
        self.ib.pnlEvent += on_pnl_update
        self.ib.accountSummaryEvent += on_account_summary_update
        
        logger.info("✅ Event handlers connected")
    
    async def start_streaming(self):
        """Start all streaming subscriptions"""
        logger.info(f"📡 Starting streaming subscriptions for account: {self.account}")
        
        try:
            # Start account value updates
            logger.info("   📊 Starting account value updates...")
            await self.ib.reqAccountUpdatesAsync(self.account)
            logger.info("   ✅ Account value updates started")
            
            # Start position updates
            logger.info("   📍 Starting position updates...")
            positions = await self.ib.reqPositionsAsync()
            logger.info(f"   ✅ Position updates started - received {len(positions) if positions else 0} initial positions")
            
            # Track which symbols TWS reports as having positions (non-zero quantities only)
            tws_position_symbols = set()
            if positions:
                for pos in positions:
                    logger.info(f"      📍 {pos.contract.symbol}: {pos.position} shares @ ${pos.avgCost:.2f} (account: {pos.account})")
                    # Process initial positions
                    if pos.account == self.account:
                        # Only track non-zero positions for sync purposes
                        # Zero-quantity positions will be deleted by handle_position_update
                        if pos.position != 0:
                            tws_position_symbols.add(pos.contract.symbol)
                        await self.handle_position_update(pos)
            else:
                logger.info("      No positions found")
            
            # Sync database with TWS: remove any positions in DB that TWS doesn't report
            # This handles the case where positions were closed but TWS didn't send zero-quantity updates
            await self._sync_positions_with_tws(tws_position_symbols)
            
            # Start P&L updates
            logger.info("   💰 Starting P&L updates...")
            await self.ib.reqPnLAsync(self.account, '')
            logger.info("   ✅ P&L updates started")
            
            # Start account summary updates
            logger.info("   📈 Starting account summary updates...")
            await self.ib.reqAccountSummaryAsync()
            logger.info("   ✅ Account summary updates started")
            
            # Wait a moment for initial data to flow in
            await asyncio.sleep(1)
            
            # Log what we've received
            net_liq = self.account_values.get('NetLiquidation', {}).get('value', 'N/A')
            avail_funds = self.account_values.get('AvailableFunds', {}).get('value', 'N/A')
            buying_power = self.account_values.get('BuyingPower', {}).get('value', 'N/A')
            
            logger.info("🎯 All streaming subscriptions active")
            logger.info(f"   💰 Net Liquidation: ${net_liq}")
            logger.info(f"   💵 Available Funds: ${avail_funds}")
            logger.info(f"   🔄 Buying Power: ${buying_power}")
            logger.info(f"   📊 Positions: {len(self.current_positions)}")
            
        except Exception as e:
            logger.error(f"❌ Error starting subscriptions: {e}")
            raise
    
    async def handle_account_value_update(self, account_value):
        """Handle account value updates (balance, margin, etc.)"""
        try:
            self.stats['account_updates'] += 1
            self.stats['last_update'] = datetime.now(timezone.utc)
            
            # Store in memory cache
            self.account_values[account_value.tag] = {
                'value': account_value.value,
                'currency': account_value.currency,
                'timestamp': self.stats['last_update']
            }
            
            # Store to database
            await self._store_account_summary(
                tag=account_value.tag,
                value=account_value.value,
                currency=account_value.currency
            )
            
            # Broadcast to WebSocket clients
            await self.websocket_manager.broadcast({
                'type': 'account_value',
                'data': {
                    'account': account_value.account,
                    'tag': account_value.tag,
                    'value': account_value.value,
                    'currency': account_value.currency,
                    'timestamp': self.stats['last_update'].isoformat()
                }
            })
            
            # Log important updates
            if account_value.tag in ['NetLiquidation', 'TotalCashValue', 'BuyingPower', 'AvailableFunds']:
                logger.info(f"💰 {account_value.tag}: {account_value.value} {account_value.currency}")
                
        except Exception as e:
            logger.error(f"Error handling account value update: {e}")
    
    async def handle_position_update(self, position):
        """Handle position updates"""
        try:
            self.stats['position_updates'] += 1
            self.stats['last_update'] = datetime.now(timezone.utc)
            
            # Store in memory cache
            symbol = position.contract.symbol
            position_qty = position.position
            
            # ib-insync Position objects don't always have marketPrice/marketValue/unrealizedPNL
            # These come separately or need to be calculated
            market_price = getattr(position, 'marketPrice', None)
            market_value = getattr(position, 'marketValue', None)
            unrealized_pnl = getattr(position, 'unrealizedPNL', None)
            
            # Remove from in-memory cache if quantity is zero (for display purposes)
            # But still store qty=0 in database for audit trail
            if position_qty == 0:
                if symbol in self.current_positions:
                    del self.current_positions[symbol]
                    logger.info(f"📍 Position closed: {symbol} (qty=0, removed from cache, kept in DB for audit)")
            else:
                # Update in-memory cache
                self.current_positions[symbol] = {
                    'symbol': symbol,
                    'qty': position_qty,
                    'avg_price': position.avgCost,
                    'market_price': market_price,
                    'market_value': market_value,
                    'unrealized_pnl': unrealized_pnl,
                    'conid': position.contract.conId,
                    'timestamp': self.stats['last_update']
                }
            
            # Always store position to database (including qty=0 for audit trail)
            # Frontend and queries filter out qty=0 positions
            await self._store_position(
                symbol=symbol,
                conid=position.contract.conId,
                qty=position_qty,
                avg_price=position.avgCost,
                market_price=market_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl
            )
            
            # Broadcast to WebSocket clients
            await self.websocket_manager.broadcast({
                'type': 'position',
                'data': {
                    'account': position.account,
                    'symbol': symbol,
                    'position': position_qty,
                    'avg_cost': position.avgCost if position_qty != 0 else 0,
                    'market_price': market_price,
                    'market_value': market_value,
                    'unrealized_pnl': unrealized_pnl,
                    'timestamp': self.stats['last_update'].isoformat()
                }
            })
            
            if position_qty != 0:
                logger.info(f"📍 Position: {symbol} = {position_qty} shares @ ${position.avgCost}")
            
        except Exception as e:
            logger.error(f"Error handling position update: {e}")
    
    async def handle_pnl_update(self, pnl):
        """Handle P&L updates"""
        try:
            self.stats['pnl_updates'] += 1
            self.stats['last_update'] = datetime.now(timezone.utc)
            
            # Store in memory cache
            self.pnl_data = {
                'daily_pnl': pnl.dailyPnL,
                'unrealized_pnl': pnl.unrealizedPnL,
                'realized_pnl': pnl.realizedPnL,
                'timestamp': self.stats['last_update']
            }
            
            # Broadcast to WebSocket clients
            await self.websocket_manager.broadcast({
                'type': 'pnl',
                'data': {
                    'account': getattr(pnl, 'account', self.account),
                    'daily_pnl': pnl.dailyPnL,
                    'unrealized_pnl': pnl.unrealizedPnL,
                    'realized_pnl': pnl.realizedPnL,
                    'timestamp': self.stats['last_update'].isoformat()
                }
            })
            
            logger.info(f"📊 P&L: Daily={pnl.dailyPnL}, Unrealized={pnl.unrealizedPnL}")
            
        except Exception as e:
            logger.error(f"Error handling P&L update: {e}")
    
    async def handle_account_summary_update(self, account_summary):
        """Handle account summary updates"""
        try:
            self.stats['summary_updates'] += 1
            self.stats['last_update'] = datetime.now(timezone.utc)
            
            # Broadcast to WebSocket clients
            await self.websocket_manager.broadcast({
                'type': 'account_summary',
                'data': {
                    'account': account_summary.account,
                    'tag': account_summary.tag,
                    'value': account_summary.value,
                    'currency': account_summary.currency,
                    'timestamp': self.stats['last_update'].isoformat()
                }
            })
            
        except Exception as e:
            logger.error(f"Error handling account summary update: {e}")
    
    async def _store_account_summary(self, tag: str, value: str, currency: str):
        """Store account summary value to database"""
        try:
            # Handle TWS "BASE" currency (4 chars) - map to USD for database storage
            if currency == "BASE" or len(currency) > 3:
                currency = "USD"
            
            with get_db_session() as session:
                # Ensure account exists
                account = session.query(Account).filter_by(account_id=self.account).first()
                if not account:
                    account = Account(account_id=self.account, currency=currency)
                    session.add(account)
                    session.flush()
                
                # Insert account summary record
                summary = AccountSummary(
                    account_id=self.account,
                    tag=tag,
                    value=str(value),
                    currency=currency,
                    ts=datetime.now(timezone.utc)
                )
                session.add(summary)
                session.commit()
        except Exception as e:
            logger.error(f"Error storing account summary to database: {e}")
    
    async def _store_position(self, symbol: str, conid: int, qty: float, 
                             avg_price: float, market_price: Optional[float] = None,
                             market_value: Optional[float] = None, 
                             unrealized_pnl: Optional[float] = None):
        """Store position to database (including qty=0 for audit trail)"""
        try:
            with get_db_session() as session:
                # Ensure account exists
                account = session.query(Account).filter_by(account_id=self.account).first()
                if not account:
                    account = Account(account_id=self.account, currency='USD')
                    session.add(account)
                    session.flush()
                
                # Ensure symbol exists (auto-create if missing)
                db_symbol = session.query(Symbol).filter_by(symbol=symbol).first()
                if not db_symbol:
                    logger.info(f"Auto-creating symbol: {symbol} (conId: {conid})")
                    db_symbol = Symbol(
                        symbol=symbol,
                        conid=conid,
                        primary_exchange='SMART',  # Default exchange
                        currency='USD',
                        active=True
                    )
                    session.add(db_symbol)
                    session.flush()
                
                # Delete old position for this symbol (we'll insert new one)
                session.execute(
                    text("DELETE FROM positions WHERE account_id = :account_id AND symbol = :symbol"),
                    {'account_id': self.account, 'symbol': symbol}
                )
                
                # Insert new position
                position = Position(
                    account_id=self.account,
                    symbol=symbol,
                    conid=conid,
                    qty=Decimal(str(qty)),
                    avg_price=Decimal(str(avg_price)),
                    ts=datetime.now(timezone.utc)
                )
                session.add(position)
                session.commit()
        except Exception as e:
            logger.error(f"Error storing position to database: {e}")
    
    async def _sync_positions_with_tws(self, tws_position_symbols: set):
        """Sync database positions with TWS: set qty=0 for positions that TWS doesn't report"""
        try:
            with get_db_session() as session:
                # Get all positions in database for this account (including zero-quantity)
                db_positions = session.query(Position).filter(
                    Position.account_id == self.account
                ).all()
                
                stale_count = 0
                for db_pos in db_positions:
                    if db_pos.symbol not in tws_position_symbols and db_pos.qty != 0:
                        logger.info(f"   🔄 Closing stale position: {db_pos.symbol} (qty={db_pos.qty} -> 0, not in TWS)")
                        db_pos.qty = Decimal('0')
                        db_pos.ts = datetime.now(timezone.utc)  # Update timestamp to show when it was closed
                        stale_count += 1
                        
                        # Also remove from in-memory cache if present
                        if db_pos.symbol in self.current_positions:
                            del self.current_positions[db_pos.symbol]
                
                if stale_count > 0:
                    session.commit()
                    logger.info(f"   ✅ Closed {stale_count} stale position(s) (set qty=0 for audit trail)")
                else:
                    logger.info(f"   ✅ Database positions are in sync with TWS")
        except Exception as e:
            logger.error(f"Error syncing positions with TWS: {e}")
    
    def get_current_account_state(self) -> Dict[str, Any]:
        """Get current account state from in-memory cache"""
        # Extract key values
        net_liquidation = None
        available_funds = None
        buying_power = None
        
        if 'NetLiquidation' in self.account_values:
            try:
                net_liquidation = float(self.account_values['NetLiquidation']['value'])
            except (ValueError, TypeError):
                pass
        
        if 'AvailableFunds' in self.account_values:
            try:
                available_funds = float(self.account_values['AvailableFunds']['value'])
            except (ValueError, TypeError):
                pass
        
        if 'BuyingPower' in self.account_values:
            try:
                buying_power = float(self.account_values['BuyingPower']['value'])
            except (ValueError, TypeError):
                pass
        
        # Convert positions to list format
        positions = []
        for symbol, pos_data in self.current_positions.items():
            positions.append({
                'symbol': symbol,
                'qty': pos_data['qty'],
                'avg_price': pos_data['avg_price'],
                'market_price': pos_data.get('market_price'),
                'market_value': pos_data.get('market_value'),
                'unrealized_pnl': pos_data.get('unrealized_pnl')
            })
        
        return {
            'account_id': self.account,
            'net_liquidation': net_liquidation,
            'available_funds': available_funds,
            'buying_power': buying_power,
            'positions': positions,
            'pnl': self.pnl_data,
            'last_update': self.stats['last_update'].isoformat() if self.stats['last_update'] else None
        }
    
    async def stop(self):
        """Stop the account service"""
        logger.info("🛑 Stopping Account Service...")
        
        self.running = False
        
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("✅ Disconnected from TWS")
        
        logger.info("👋 Account Service stopped")
    
    async def startup_tws(self):
        """Startup routine for TWS connection"""
        logger.info("🚀 Starting TWS connection in background...")
        # Start TWS connection in background (don't block server startup)
        asyncio.create_task(self._connect_with_retry())
        logger.info("✅ TWS connection task created")
    
    async def _connect_with_retry(self):
        """Connect to TWS with retries"""
        try:
            await self.start_tws_connection()
            # Wait a moment for initial data
            await asyncio.sleep(2)
            logger.info(f"🎯 TWS ready - {self.stats['account_updates']} account updates, {self.stats['position_updates']} positions, {len(self.current_positions)} in memory")
            
            # Ensure positions are synced after connection (in case sync didn't run during startup)
            # Get current positions from TWS and sync with database
            try:
                positions = await self.ib.reqPositionsAsync()
                tws_position_symbols = set()
                if positions:
                    for pos in positions:
                        if pos.account == self.account and pos.position != 0:
                            tws_position_symbols.add(pos.contract.symbol)
                await self._sync_positions_with_tws(tws_position_symbols)
            except Exception as sync_error:
                logger.warning(f"Position sync after TWS connection failed: {sync_error}")
        except Exception as e:
            logger.error(f"❌ TWS connection failed: {e}")
            # Retry after 5 seconds
            await asyncio.sleep(5)
            logger.info("🔄 Retrying TWS connection...")
            asyncio.create_task(self._connect_with_retry())
    
    def create_app(self) -> FastAPI:
        """Create FastAPI application"""
        from contextlib import asynccontextmanager
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info("🚀 FastAPI lifespan startup - initializing TWS...")
            await self.startup_tws()
            yield
            # Shutdown
            logger.info("🛑 FastAPI lifespan shutdown - stopping account service...")
            await self.stop()
        
        app = FastAPI(
            title="Account Service",
            description="Real-time TWS account data streaming service",
            version="1.0.0",
            lifespan=lifespan
        )
        
        @app.get("/healthz")
        async def health_check():
            """Health check endpoint"""
            is_connected = self.ib and self.ib.isConnected()
            
            return {
                "status": "healthy",  # API is always healthy if responding
                "service": "account",
                "tws_connected": is_connected,
                "account": self.account,
                "stats": self.stats,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        @app.post("/account/sync-positions")
        async def sync_positions():
            """Manually trigger position sync with TWS"""
            if not self.ib or not self.ib.isConnected():
                return {"error": "TWS not connected", "tws_connected": False}
            
            try:
                # Get current positions from TWS
                positions = await self.ib.reqPositionsAsync()
                tws_position_symbols = set()
                
                if positions:
                    for pos in positions:
                        if pos.account == self.account and pos.position != 0:
                            tws_position_symbols.add(pos.contract.symbol)
                
                # Sync with database
                await self._sync_positions_with_tws(tws_position_symbols)
                
                return {
                    "success": True,
                    "tws_positions": list(tws_position_symbols),
                    "message": "Position sync completed"
                }
            except Exception as e:
                logger.error(f"Error in manual position sync: {e}")
                return {"error": str(e), "success": False}
        
        @app.get("/account/stats")
        async def get_account_stats():
            """Get current account stats from database"""
            try:
                with get_db_session() as session:
                    # Get most recent account summary values
                    # Query for key account values (most recent per tag)
                    subq = session.query(
                        AccountSummary.tag,
                        func.max(AccountSummary.ts).label('max_ts')
                    ).filter(
                        AccountSummary.account_id == self.account
                    ).group_by(AccountSummary.tag).subquery()
                    
                    recent_summaries = session.query(AccountSummary).join(
                        subq,
                        and_(
                            AccountSummary.tag == subq.c.tag,
                            AccountSummary.ts == subq.c.max_ts
                        )
                    ).all()
                    
                    # Extract key values
                    values_dict = {s.tag: s.value for s in recent_summaries}
                    net_liquidation = float(values_dict.get('NetLiquidation', 0)) if 'NetLiquidation' in values_dict else None
                    available_funds = float(values_dict.get('AvailableFunds', 0)) if 'AvailableFunds' in values_dict else None
                    buying_power = float(values_dict.get('BuyingPower', 0)) if 'BuyingPower' in values_dict else None
                    
                    # Extract P&L values from database (from accountValueEvent)
                    # TWS sends these via accountValueEvent with tags like "RealizedPnL", "UnrealizedPnL"
                    realized_pnl = float(values_dict.get('RealizedPnL', 0)) if 'RealizedPnL' in values_dict else None
                    unrealized_pnl = float(values_dict.get('UnrealizedPnL', 0)) if 'UnrealizedPnL' in values_dict else None
                    
                    # Calculate Daily P&L from Realized + Unrealized (or use memory fallback)
                    daily_pnl = None
                    if realized_pnl is not None and unrealized_pnl is not None:
                        daily_pnl = realized_pnl + unrealized_pnl
                    elif self.pnl_data and self.pnl_data.get('daily_pnl') is not None:
                        daily_pnl = self.pnl_data.get('daily_pnl')
                    
                    # Fallback to memory if database values not available (for backward compatibility)
                    pnl_data = {
                        'daily_pnl': daily_pnl,
                        'unrealized_pnl': unrealized_pnl if unrealized_pnl is not None else (self.pnl_data.get('unrealized_pnl') if self.pnl_data else None),
                        'realized_pnl': realized_pnl if realized_pnl is not None else (self.pnl_data.get('realized_pnl') if self.pnl_data else None),
                        'timestamp': self.stats['last_update'].isoformat() if self.stats['last_update'] else None
                    }
                    
                    # Additional useful metrics for risk management
                    total_cash_value = float(values_dict.get('TotalCashValue', 0)) if 'TotalCashValue' in values_dict else None
                    cash_balance = float(values_dict.get('CashBalance', 0)) if 'CashBalance' in values_dict else None
                    equity_with_loan = float(values_dict.get('EquityWithLoanValue', 0)) if 'EquityWithLoanValue' in values_dict else None
                    cushion = float(values_dict.get('Cushion', 0)) if 'Cushion' in values_dict else None
                    excess_liquidity = float(values_dict.get('ExcessLiquidity', 0)) if 'ExcessLiquidity' in values_dict else None
                    maint_margin_req = float(values_dict.get('MaintMarginReq', 0)) if 'MaintMarginReq' in values_dict else None
                    gross_position_value = float(values_dict.get('GrossPositionValue', 0)) if 'GrossPositionValue' in values_dict else None
                    
                    # Get current positions from database (exclude zero quantity positions)
                    db_positions = session.query(Position).filter(
                        and_(
                            Position.account_id == self.account,
                            Position.qty != 0  # Only show positions with non-zero quantity
                        )
                    ).all()
                    
                    positions = [{
                        'symbol': p.symbol,
                        'qty': float(p.qty),
                        'avg_price': float(p.avg_price),
                        'market_price': None,  # Not stored in positions table
                        'market_value': None,
                        'unrealized_pnl': None
                    } for p in db_positions]
                    
                    return {
                        'account_id': self.account,
                        'net_liquidation': net_liquidation,
                        'available_funds': available_funds,
                        'buying_power': buying_power,
                        'positions': positions,
                        'pnl': pnl_data,  # Now from database with fallback to memory
                        'total_cash_value': total_cash_value,
                        'cash_balance': cash_balance,
                        'equity_with_loan': equity_with_loan,
                        'cushion': cushion,
                        'excess_liquidity': excess_liquidity,
                        'maint_margin_req': maint_margin_req,
                        'gross_position_value': gross_position_value,
                        'last_update': self.stats['last_update'].isoformat() if self.stats['last_update'] else None,
                        'tws_connected': self.ib.isConnected() if self.ib else False,
                        'stats': self.stats
                    }
                    
            except Exception as e:
                logger.error(f"Error getting account stats from database: {e}")
                # Fallback to in-memory cache
                account_state = self.get_current_account_state()
                account_state['tws_connected'] = self.ib.isConnected() if self.ib else False
                account_state['stats'] = self.stats
                return account_state
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time data"""
            await self.websocket_manager.connect(websocket)
            try:
                while self.running:
                    # Keep connection alive
                    await asyncio.sleep(1)
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)
        
        return app

# Global service instance
service = AccountService()

def main():
    """Main entry point"""
    setup_logging("account")
    
    logger.info("🌐 Starting Account Service on port 8001...")
    # Use string import path instead of app object to ensure lifespan works
    uvicorn.run(
        "src.services.account.main:app",
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )

# Create app at module level for uvicorn to import
app = service.create_app()

if __name__ == "__main__":
    main()