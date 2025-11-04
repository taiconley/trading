#!/usr/bin/env python3
"""
Trader Service - Order Management and Execution
Handles order placement, cancellation, risk management, and execution tracking.
"""
import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set
from decimal import Decimal
import json

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse
import uvicorn
from ib_insync import IB, Stock, Order as IBOrder, Trade
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from pydantic import BaseModel

# Add the backend src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.config import get_settings
from common.logging import setup_logging
from common.db import get_db_session
from common.models import Order, Execution, Account, Symbol, RiskLimit, HealthStatus
from common.schemas import (
    OrderRequest, OrderResponse, ExecutionSchema, 
    HealthCheckResponse, OrderStatus, OrderSide, OrderType, TimeInForce
)
from tws_bridge.client_ids import ClientIdManager
from tws_bridge.ib_client import EnhancedIBClient

# Define missing schema
class OrderListResponse(BaseModel):
    """Order list response schema"""
    orders: List[OrderResponse]
    total: int

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections for real-time order updates"""
    
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
                logger.warning(f"Failed to send WebSocket message: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.active_connections.discard(connection)

class RiskManager:
    """Enhanced risk management with violation logging and emergency stop"""
    
    def __init__(self, db_session_factory):
        self.db_session_factory = db_session_factory
        self.risk_limits_cache = {}
        self.last_cache_update = None
        self.cache_ttl = 60  # seconds
        self.emergency_stop_active = False
    
    async def check_emergency_stop(self) -> tuple[bool, str]:
        """
        Check if emergency stop is active.
        
        Returns:
            (is_stopped, reason)
        """
        try:
            with self.db_session_factory() as session:
                from common.models import RiskLimit
                stop_limit = session.query(RiskLimit).filter(
                    RiskLimit.key == 'emergency_stop'
                ).first()
                
                if stop_limit and stop_limit.value_json == True:
                    self.emergency_stop_active = True
                    return True, "Emergency stop is active - all trading halted"
                    
            self.emergency_stop_active = False
            return False, ""
        except Exception as e:
            logger.error(f"Error checking emergency stop: {e}")
            return False, ""
    
    async def log_violation(self, violation_type: str, account_id: str, symbol: str,
                           limit_key: str, limit_value: float, actual_value: float,
                           message: str, severity: str = 'warning', 
                           strategy_id: int = None, order_id: int = None,
                           action_taken: str = 'rejected', metadata: dict = None):
        """Log a risk violation to the database and send alerts"""
        try:
            with self.db_session_factory() as session:
                from common.models import RiskViolation
                violation = RiskViolation(
                    violation_type=violation_type,
                    severity=severity,
                    account_id=account_id,
                    symbol=symbol,
                    strategy_id=strategy_id,
                    order_id=order_id,
                    limit_key=limit_key,
                    limit_value=limit_value,
                    actual_value=actual_value,
                    message=message,
                    metadata_json=metadata,
                    action_taken=action_taken,
                    resolved=False,
                    created_at=datetime.now(timezone.utc)
                )
                session.add(violation)
                session.commit()
                
                logger.warning(f"Risk violation logged: {violation_type} - {message}", 
                             extra={'violation_id': violation.id, 'severity': severity})
                
                # Send alert notification
                try:
                    from common.risk_alerts import send_violation_alert
                    await send_violation_alert(
                        violation_type=violation_type,
                        message=message,
                        violation_id=violation.id,
                        severity=severity,
                        metadata={
                            'account_id': account_id,
                            'symbol': symbol,
                            'limit_key': limit_key,
                            'limit_value': limit_value,
                            'actual_value': actual_value,
                            **(metadata or {})
                        }
                    )
                except Exception as alert_error:
                    logger.error(f"Failed to send violation alert: {alert_error}")
                
        except Exception as e:
            logger.error(f"Failed to log risk violation: {e}")
    
    async def validate_order(self, order_request: OrderRequest, account_id: str, 
                            strategy_id: int = None) -> tuple[bool, str]:
        """
        Validate order against risk limits with comprehensive logging.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            # Check emergency stop first
            is_stopped, stop_reason = await self.check_emergency_stop()
            if is_stopped:
                await self.log_violation(
                    violation_type='emergency_stop',
                    account_id=account_id,
                    symbol=order_request.symbol,
                    limit_key='emergency_stop',
                    limit_value=1,
                    actual_value=1,
                    message=stop_reason,
                    severity='critical',
                    strategy_id=strategy_id,
                    action_taken='emergency_stop'
                )
                return False, stop_reason
            
            # Refresh risk limits cache if needed
            await self._refresh_risk_limits()
            
            # Check individual order size
            notional = float(order_request.quantity) * (order_request.limit_price or 0)
            max_notional_per_order = self.risk_limits_cache.get('max_notional_per_order', 100000)
            
            if notional > max_notional_per_order:
                error_msg = f"Order notional ${notional:,.2f} exceeds limit ${max_notional_per_order:,.2f}"
                await self.log_violation(
                    violation_type='order_size_exceeded',
                    account_id=account_id,
                    symbol=order_request.symbol,
                    limit_key='max_notional_per_order',
                    limit_value=max_notional_per_order,
                    actual_value=notional,
                    message=error_msg,
                    severity='warning',
                    strategy_id=strategy_id,
                    action_taken='rejected',
                    metadata={'order_type': order_request.order_type, 'side': str(order_request.side)}
                )
                return False, error_msg
            
            # Check position size limits
            with self.db_session_factory() as session:
                # Get current position
                from common.models import Position
                current_position = session.query(Position).filter(
                    and_(Position.account_id == account_id, Position.symbol == order_request.symbol)
                ).first()
                
                current_qty = float(current_position.qty) if current_position else 0.0
                new_qty = current_qty
                
                # Handle both enum and string values for order side
                side_value = order_request.side.value if hasattr(order_request.side, 'value') else str(order_request.side)
                if side_value == "BUY":
                    new_qty += float(order_request.quantity)
                else:
                    new_qty -= float(order_request.quantity)
                
                max_position_size = self.risk_limits_cache.get('max_notional_per_symbol', 500000)
                position_notional = abs(new_qty) * (order_request.limit_price or 0)
                
                if position_notional > max_position_size:
                    error_msg = f"Position would exceed symbol limit: ${position_notional:,.2f} > ${max_position_size:,.2f}"
                    await self.log_violation(
                        violation_type='position_limit_exceeded',
                        account_id=account_id,
                        symbol=order_request.symbol,
                        limit_key='max_notional_per_symbol',
                        limit_value=max_position_size,
                        actual_value=position_notional,
                        message=error_msg,
                        severity='warning',
                        strategy_id=strategy_id,
                        action_taken='rejected',
                        metadata={'current_qty': current_qty, 'new_qty': new_qty}
                    )
                    return False, error_msg
            
            # Check if live trading is blocked
            block_until = self.risk_limits_cache.get('block_live_trading_until')
            if block_until and isinstance(block_until, str):
                try:
                    block_datetime = datetime.fromisoformat(block_until.replace('Z', '+00:00'))
                    if datetime.now(timezone.utc) < block_datetime:
                        error_msg = f"Live trading blocked until {block_until}"
                        await self.log_violation(
                            violation_type='trading_blocked',
                            account_id=account_id,
                            symbol=order_request.symbol,
                            limit_key='block_live_trading_until',
                            limit_value=0,
                            actual_value=1,
                            message=error_msg,
                            severity='warning',
                            strategy_id=strategy_id,
                            action_taken='rejected'
                        )
                        return False, error_msg
                except Exception as e:
                    logger.warning(f"Error parsing block_live_trading_until: {e}")
            
            # All checks passed
            logger.info(f"Order validation passed for {order_request.symbol} - {account_id}")
            return True, ""
            
        except Exception as e:
            logger.error(f"Risk validation error: {e}")
            return False, f"Risk validation failed: {str(e)}"
    
    async def _refresh_risk_limits(self):
        """Refresh risk limits from database"""
        try:
            now = datetime.now(timezone.utc)
            if (self.last_cache_update is None or 
                (now - self.last_cache_update).total_seconds() > self.cache_ttl):
                
                with self.db_session_factory() as session:
                    limits = session.query(RiskLimit).all()
                    self.risk_limits_cache = {
                        limit.key: limit.value_json for limit in limits
                    }
                    self.last_cache_update = now
                    logger.debug(f"Refreshed risk limits: {list(self.risk_limits_cache.keys())}")
                    
        except Exception as e:
            logger.error(f"Failed to refresh risk limits: {e}")

class TraderService:
    """Main trader service class"""
    
    def __init__(self):
        self.settings = get_settings()
        self.client_id_manager = ClientIdManager()
        self.client_id = None
        self.ib_client = None
        self.db_session_factory = get_db_session
        self.risk_manager = RiskManager(self.db_session_factory)
        self.websocket_manager = WebSocketManager()
        self.active_orders: Dict[int, Trade] = {}  # order_id -> Trade
        self.shutdown_event = asyncio.Event()
        
    async def initialize(self):
        print("DEBUG: initialize() ENTRY POINT")
        """Initialize the trader service"""
        print("DEBUG: INSIDE initialize() - first line")
        try:
            # Get client ID
            print("DEBUG: About to get client ID")
            self.client_id = self.client_id_manager.get_service_client_id("trader")
            print(f"DEBUG: Got client ID: {self.client_id}")
            logger.info(f"Allocated client ID: {self.client_id}")
            
            # Initialize IB client
            self.ib_client = EnhancedIBClient(
                client_id=self.client_id,
                host=self.settings.tws.host,
                port=self.settings.tws.port
            )
            
            # Connect to TWS FIRST
            print("DEBUG: About to connect to TWS")
            await self.ib_client.connect()
            print(f"DEBUG: TWS connect() completed, ib_client={self.ib_client}, ib={self.ib_client.ib if self.ib_client else None}")
            print(f"DEBUG: ib.isConnected()={self.ib_client.ib.isConnected() if self.ib_client and self.ib_client.ib else 'N/A'}")
            logger.info("Connected to TWS successfully")
            
            # Set up event handlers AFTER connection
            print("DEBUG: Setting up event handlers...")
            
            # Test handler to verify events are flowing
            def _test_error_handler(reqId, errorCode, errorString, contract):
                print(f"🔧 TWS EVENT RECEIVED: Error {errorCode} - {errorString}")
            
            # Also add a new order event handler
            def _test_new_order_handler(trade):
                print(f"🆕 NEW ORDER EVENT: Order {trade.order.orderId} created - {trade.contract.symbol}")
                # Forward to the main handler
                self._on_order_status(trade)
            
            self.ib_client.ib.errorEvent += _test_error_handler
            self.ib_client.ib.newOrderEvent += _test_new_order_handler
            self.ib_client.ib.orderStatusEvent += self._on_order_status
            self.ib_client.ib.execDetailsEvent += self._on_execution
            print(f"DEBUG: Event handlers registered:")
            print(f"  - orderStatusEvent has {len(self.ib_client.ib.orderStatusEvent)} subscribers")
            print(f"  - newOrderEvent has {len(self.ib_client.ib.newOrderEvent)} subscribers")
            print(f"  - execDetailsEvent has {len(self.ib_client.ib.execDetailsEvent)} subscribers")
            print(f"  - errorEvent has {len(self.ib_client.ib.errorEvent)} subscribers")
            
            # Subscribe to ALL open orders (not just orders from this client ID)
            print("DEBUG: Requesting all open orders with reqAutoOpenOrders(True)...")
            self.ib_client.ib.reqAutoOpenOrders(True)
            print("DEBUG: Waiting for IB to send existing orders via events...")
            
            # Wait a moment for IB to send existing orders via events
            await asyncio.sleep(3)
            
            # Get current trades that are already loaded
            existing_trades = self.ib_client.ib.trades()
            print(f"DEBUG: Found {len(existing_trades)} existing trades in memory")
            
            # Sync existing orders to database (one-time on startup)
            for trade in existing_trades:
                self._on_order_status(trade)
            
            logger.info(f"Subscribed to all open orders, synced {len(existing_trades)} existing to database")
            print(f"DEBUG: Event-driven order updates enabled - orders will be created immediately when placed in TWS")
            
            # Request executions (non-async, just trigger the request)
            print("DEBUG: Requesting execution history...")
            try:
                from ib_insync import ExecutionFilter
                self.ib_client.ib.reqExecutions(ExecutionFilter())
                print("DEBUG: Execution history requested (will arrive via execDetailsEvent)")
            except Exception as e:
                logger.warning(f"Failed to request executions: {e}")
                print(f"DEBUG: Execution request warning (non-critical): {e}")
            
            # Update health status
            await self._update_health_status("healthy")
            print("DEBUG: Health status updated")
            
        except Exception as e:
            print(f"DEBUG: EXCEPTION in initialize(): {e}")
            import traceback
            traceback.print_exc()
            logger.error(f"Failed to initialize trader service: {e}")
            await self._update_health_status("unhealthy")
            raise
    
    def _on_order_status(self, trade: Trade):
        """Handle order status updates from TWS (MUST BE SYNC for ib_insync)"""
        print("=" * 80)
        print("🔔 ORDER STATUS EVENT TRIGGERED!")
        print("=" * 80)
        try:
            order_status = trade.orderStatus.status
            ib_order_id = trade.order.orderId
            
            # Skip invalid orders (orderId=0 means it's not a real order)
            if not ib_order_id or ib_order_id == 0:
                print(f"⚠️  SKIPPING invalid order event with orderId={ib_order_id}")
                return
            
            # Skip orders with zero quantity (junk events)
            if not trade.order.totalQuantity or trade.order.totalQuantity <= 0:
                print(f"⚠️  SKIPPING order {ib_order_id} with zero quantity")
                return
            
            print(f"🔔 ORDER EVENT: {ib_order_id} -> {order_status} (Symbol: {trade.contract.symbol})")
            print(f"   Order details: {trade.order.action} {trade.order.totalQuantity} {trade.contract.symbol}")
            print(f"   Account: {trade.order.account}")
            print(f"   ClientId: {trade.order.clientId}")
            logger.info(f"Order status update: {ib_order_id} -> {order_status}")
            
            # Update or create order in database
            with self.db_session_factory() as session:
                db_order = session.query(Order).filter(
                    Order.external_order_id == str(ib_order_id)
                ).first()
                
                if db_order:
                    # Update existing order
                    print(f"   📝 Updating existing order ID {db_order.id}: {db_order.symbol} -> {order_status}")
                    db_order.status = order_status
                    db_order.updated_at = datetime.now(timezone.utc)
                    session.commit()
                else:
                    # Auto-create external order (placed directly in TWS)
                    print(f"   ➕ Creating NEW order in database: {trade.contract.symbol} {trade.order.action} {trade.order.totalQuantity}")
                    logger.info(f"Creating external order from TWS: {ib_order_id}")
                    
                    # Ensure account exists
                    account_id = trade.order.account if trade.order.account else "DU7084660"
                    account = session.query(Account).filter_by(account_id=account_id).first()
                    if not account:
                        account = Account(account_id=account_id, currency='USD')
                        session.add(account)
                        session.flush()
                    
                    # Ensure symbol exists
                    symbol = trade.contract.symbol
                    db_symbol = session.query(Symbol).filter_by(symbol=symbol).first()
                    if not db_symbol:
                        logger.info(f"Auto-creating symbol: {symbol}")
                        db_symbol = Symbol(
                            symbol=symbol,
                            conid=trade.contract.conId,
                            primary_exchange=trade.contract.primaryExchange or 'SMART',
                            currency=trade.contract.currency or 'USD',
                            active=True
                        )
                        session.add(db_symbol)
                        session.flush()
                    
                    # Create new order
                    db_order = Order(
                        account_id=account_id,
                        strategy_id=None,  # External order, no strategy
                        symbol=symbol,
                        side=trade.order.action,  # BUY or SELL
                        qty=Decimal(str(trade.order.totalQuantity)),
                        order_type=trade.order.orderType,  # MKT, LMT, etc.
                        limit_price=Decimal(str(trade.order.lmtPrice)) if trade.order.lmtPrice else None,
                        stop_price=Decimal(str(trade.order.auxPrice)) if trade.order.auxPrice else None,
                        tif=trade.order.tif,
                        status=order_status,
                        external_order_id=str(ib_order_id),
                        placed_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc)
                    )
                    session.add(db_order)
                    session.commit()
                    print(f"   ✅ Order saved to database! ID: {db_order.id}")
                    logger.info(f"✅ Created external order: {symbol} {trade.order.action} {trade.order.totalQuantity}")
                
                # Broadcast update via WebSocket (run in background)
                asyncio.create_task(self.websocket_manager.broadcast({
                    'type': 'order_status',
                    'data': {
                        'order_id': db_order.id,
                        'external_order_id': str(ib_order_id),
                        'status': order_status,
                        'symbol': db_order.symbol,
                        'side': db_order.side,
                        'qty': float(db_order.qty),
                        'updated_at': db_order.updated_at.isoformat()
                    }
                }))
                print(f"   📡 Order ID {db_order.id} ready - frontend will see it on next poll (2s)")
                    
        except Exception as e:
            logger.error(f"Error handling order status update: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_execution(self, trade: Trade, fill):
        """Handle execution reports from TWS (MUST BE SYNC for ib_insync)"""
        try:
            execution = fill.execution
            logger.info(f"Execution: {execution.execId} - {execution.shares} @ {execution.price}")
            
            # Find the corresponding database order
            with self.db_session_factory() as session:
                db_order = session.query(Order).filter(
                    Order.external_order_id == str(trade.order.orderId)
                ).first()
                
                if db_order:
                    # Create execution record
                    db_execution = Execution(
                        order_id=db_order.id,
                        trade_id=execution.execId,
                        symbol=trade.contract.symbol,
                        qty=Decimal(str(execution.shares)),
                        price=Decimal(str(execution.price)),
                        ts=datetime.now(timezone.utc)
                    )
                    session.add(db_execution)
                    session.commit()
                    
                    # Broadcast execution via WebSocket (run in background)
                    asyncio.create_task(self.websocket_manager.broadcast({
                        'type': 'execution',
                        'data': {
                            'execution_id': db_execution.id,
                            'order_id': db_order.id,
                            'trade_id': execution.execId,
                            'symbol': trade.contract.symbol,
                            'qty': float(execution.shares),
                            'price': float(execution.price),
                            'timestamp': db_execution.ts.isoformat()
                        }
                    }))
                else:
                    # Order doesn't exist - this happens with fast-filling orders (market orders)
                    # Create the order record from the execution data
                    
                    # Skip invalid executions (orderId=0, qty=0, etc.)
                    if not trade.order.orderId or trade.order.orderId == 0:
                        logger.warning(f"Skipping invalid execution with orderId={trade.order.orderId}")
                        return
                    if not trade.order.totalQuantity or trade.order.totalQuantity <= 0:
                        logger.warning(f"Skipping execution with invalid quantity={trade.order.totalQuantity}")
                        return
                    
                    logger.info(f"Creating order from execution: {trade.order.orderId}")
                    
                    # Ensure account exists
                    account_id = trade.order.account if trade.order.account else "DU7084660"
                    account = session.query(Account).filter_by(account_id=account_id).first()
                    if not account:
                        account = Account(account_id=account_id, currency='USD')
                        session.add(account)
                        session.flush()
                    
                    # Ensure symbol exists
                    symbol = fill.contract.symbol
                    db_symbol = session.query(Symbol).filter_by(symbol=symbol).first()
                    if not db_symbol:
                        logger.info(f"Auto-creating symbol: {symbol}")
                        db_symbol = Symbol(
                            symbol=symbol,
                            conid=fill.contract.conId,
                            primary_exchange=fill.contract.primaryExchange or 'SMART',
                            currency=fill.contract.currency or 'USD',
                            active=True
                        )
                        session.add(db_symbol)
                        session.flush()
                    
                    # Get execution time (could be datetime or need conversion)
                    if hasattr(execution, 'time') and execution.time:
                        if isinstance(execution.time, datetime):
                            exec_time = execution.time if execution.time.tzinfo else execution.time.replace(tzinfo=timezone.utc)
                        else:
                            exec_time = datetime.fromtimestamp(execution.time, tz=timezone.utc)
                    else:
                        exec_time = datetime.now(timezone.utc)
                    
                    # Create order record from execution
                    db_order = Order(
                        account_id=account_id,
                        strategy_id=None,  # External order
                        symbol=symbol,
                        side=trade.order.action,  # BUY or SELL
                        qty=Decimal(str(trade.order.totalQuantity)),
                        order_type=trade.order.orderType,  # MKT, LMT, etc.
                        limit_price=Decimal(str(trade.order.lmtPrice)) if trade.order.lmtPrice else None,
                        stop_price=Decimal(str(trade.order.auxPrice)) if trade.order.auxPrice else None,
                        tif=trade.order.tif,
                        status='Filled',  # Must be filled if we're getting execution
                        external_order_id=str(trade.order.orderId),
                        placed_at=exec_time,
                        updated_at=datetime.now(timezone.utc)
                    )
                    session.add(db_order)
                    session.commit()
                    session.refresh(db_order)
                    logger.info(f"✅ Created order {db_order.id} from execution")
                    
                    # Now create the execution record
                    db_execution = Execution(
                        order_id=db_order.id,
                        trade_id=execution.execId,
                        symbol=symbol,
                        qty=Decimal(str(execution.shares)),
                        price=Decimal(str(execution.price)),
                        ts=exec_time
                    )
                    session.add(db_execution)
                    session.commit()
                    
                    # Broadcast both order and execution (run in background)
                    asyncio.create_task(self.websocket_manager.broadcast({
                        'type': 'order_created_from_execution',
                        'data': {
                            'order_id': db_order.id,
                            'execution_id': db_execution.id,
                            'symbol': symbol,
                            'side': trade.order.action,
                            'qty': float(trade.order.totalQuantity),
                            'price': float(execution.price),
                            'status': 'Filled'
                        }
                    }))
                    print(f"   ✅ Order {db_order.id} created from fast-fill execution - frontend will see it on next poll (2s)")
                    
        except Exception as e:
            logger.error(f"Error handling execution: {e}")
            import traceback
            traceback.print_exc()
    
    async def _ensure_account_exists(self, account_id: str):
        """Ensure account exists in database, create if not exists"""
        try:
            with self.db_session_factory() as session:
                account = session.query(Account).filter(Account.account_id == account_id).first()
                if not account:
                    # Create the account record
                    new_account = Account(
                        account_id=account_id,
                        currency="USD",  # Default currency
                        created_at=datetime.now(timezone.utc)
                    )
                    session.add(new_account)
                    session.commit()
                    logger.info(f"Created account record for {account_id}")
        except Exception as e:
            logger.error(f"Failed to ensure account exists: {e}")
            raise
    
    async def _ensure_symbol_exists(self, symbol: str):
        """Ensure symbol exists in database, create if not exists"""
        try:
            with self.db_session_factory() as session:
                symbol_record = session.query(Symbol).filter(Symbol.symbol == symbol).first()
                if not symbol_record:
                    # Create the symbol record
                    new_symbol = Symbol(
                        symbol=symbol,
                        currency="USD",  # Default currency
                        active=True,
                        updated_at=datetime.now(timezone.utc)
                    )
                    session.add(new_symbol)
                    session.commit()
                    logger.info(f"Created symbol record for {symbol}")
        except Exception as e:
            logger.error(f"Failed to ensure symbol exists: {e}")
            raise
    
    async def place_order(self, order_request: OrderRequest, account_id: str) -> OrderResponse:
        """Place a new order"""
        try:
            # Ensure required reference data exists in database
            await self._ensure_account_exists(account_id)
            await self._ensure_symbol_exists(order_request.symbol)
            
            # Safety checks for live trading
            if not self._validate_trading_mode():
                raise HTTPException(status_code=403, detail="Trading mode validation failed")
            
            # Risk validation
            # Note: strategy_id is a string identifier (e.g., 'pairs_trading'), not an integer
            # Pass None for risk validation since it's not used for validation logic
            is_valid, error_msg = await self.risk_manager.validate_order(
                order_request, account_id, strategy_id=None
            )
            if not is_valid:
                raise HTTPException(status_code=400, detail=f"Risk check failed: {error_msg}")
            
            # DRY_RUN mode - simulate order without sending to TWS
            if self.settings.tws.dry_run:
                return await self._simulate_order(order_request, account_id)
            
            # Create database order record
            with self.db_session_factory() as session:
                db_order = Order(
                    account_id=account_id,
                    strategy_id=order_request.strategy_id,
                    symbol=order_request.symbol,
                    side=order_request.side.value if hasattr(order_request.side, 'value') else str(order_request.side),
                    qty=Decimal(str(order_request.quantity)),
                    order_type=order_request.order_type.value if hasattr(order_request.order_type, 'value') else str(order_request.order_type),
                    limit_price=Decimal(str(order_request.limit_price)) if order_request.limit_price else None,
                    stop_price=Decimal(str(order_request.stop_price)) if order_request.stop_price else None,
                    tif=order_request.time_in_force.value if hasattr(order_request.time_in_force, 'value') else str(order_request.time_in_force),
                    status=OrderStatus.PENDING_SUBMIT.value,
                    placed_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )
                session.add(db_order)
                session.commit()
                session.refresh(db_order)
                
                # Create IB contract and order
                contract = Stock(order_request.symbol, 'SMART', 'USD')
                
                ib_order = IBOrder()
                ib_order.action = order_request.side.value if hasattr(order_request.side, 'value') else str(order_request.side)
                ib_order.totalQuantity = float(order_request.quantity)
                ib_order.orderType = order_request.order_type.value if hasattr(order_request.order_type, 'value') else str(order_request.order_type)
                ib_order.tif = order_request.time_in_force.value if hasattr(order_request.time_in_force, 'value') else str(order_request.time_in_force)
                
                if order_request.limit_price:
                    ib_order.lmtPrice = float(order_request.limit_price)
                if order_request.stop_price:
                    ib_order.auxPrice = float(order_request.stop_price)
                
                # Place order with TWS
                trade = self.ib_client.ib.placeOrder(contract, ib_order)
                
                # Update database with IB order ID
                db_order.external_order_id = str(trade.order.orderId)
                session.commit()
                
                # Store active order
                self.active_orders[db_order.id] = trade
                
                logger.info(f"Order placed: {db_order.id} -> IB:{trade.order.orderId}")
                
                return OrderResponse(
                    id=db_order.id,
                    account_id=db_order.account_id,
                    strategy_id=db_order.strategy_id,
                    symbol=db_order.symbol,
                    side=OrderSide(db_order.side),
                    qty=float(db_order.qty),
                    order_type=OrderType(db_order.order_type),
                    limit_price=float(db_order.limit_price) if db_order.limit_price else None,
                    stop_price=float(db_order.stop_price) if db_order.stop_price else None,
                    tif=TimeInForce(db_order.tif),
                    status=OrderStatus(db_order.status),
                    external_order_id=db_order.external_order_id,
                    placed_at=db_order.placed_at,
                    updated_at=db_order.updated_at
                )
                
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _simulate_order(self, order_request: OrderRequest, account_id: str) -> OrderResponse:
        """Simulate order placement in DRY_RUN mode"""
        # Ensure required reference data exists for simulation too
        await self._ensure_account_exists(account_id)
        await self._ensure_symbol_exists(order_request.symbol)
        
        with self.db_session_factory() as session:
            db_order = Order(
                account_id=account_id,
                strategy_id=order_request.strategy_id,
                symbol=order_request.symbol,
                side=order_request.side.value if hasattr(order_request.side, 'value') else str(order_request.side),
                qty=Decimal(str(order_request.quantity)),
                order_type=order_request.order_type.value if hasattr(order_request.order_type, 'value') else str(order_request.order_type),
                limit_price=Decimal(str(order_request.limit_price)) if order_request.limit_price else None,
                stop_price=Decimal(str(order_request.stop_price)) if order_request.stop_price else None,
                tif=order_request.time_in_force.value if hasattr(order_request.time_in_force, 'value') else str(order_request.time_in_force),
                status=OrderStatus.FILLED.value,  # Simulate immediate fill
                external_order_id=f"DRY_{int(datetime.now().timestamp())}",
                placed_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            session.add(db_order)
            session.commit()
            session.refresh(db_order)
            
            logger.info(f"DRY_RUN: Simulated order {db_order.id} as FILLED")
            
            return OrderResponse(
                id=db_order.id,
                account_id=db_order.account_id,
                strategy_id=db_order.strategy_id,
                symbol=db_order.symbol,
                side=OrderSide(db_order.side),
                qty=float(db_order.qty),
                order_type=OrderType(db_order.order_type),
                limit_price=float(db_order.limit_price) if db_order.limit_price else None,
                stop_price=float(db_order.stop_price) if db_order.stop_price else None,
                tif=TimeInForce(db_order.tif),
                status=OrderStatus(db_order.status),
                external_order_id=db_order.external_order_id,
                placed_at=db_order.placed_at,
                updated_at=db_order.updated_at
            )
    
    def _validate_trading_mode(self) -> bool:
        """Validate trading mode and safety switches"""
        settings = self.settings.tws
        
        # Always allow DRY_RUN mode
        if settings.dry_run:
            return True
        
        # Paper trading validation
        if settings.use_paper:
            if settings.port != 7497:
                logger.error("Paper trading requires TWS_PORT=7497")
                return False
            return True
        
        # Live trading validation - all conditions must be met
        if not settings.enable_live:
            logger.error("Live trading requires ENABLE_LIVE=1")
            return False
        
        if settings.use_paper:
            logger.error("Live trading requires USE_PAPER=0")
            return False
        
        if settings.port != 7496:
            logger.error("Live trading requires TWS_PORT=7496")
            return False
        
        if settings.dry_run:
            logger.error("Live trading requires DRY_RUN=0")
            return False
        
        logger.info("Live trading mode validated successfully")
        return True
    
    async def cancel_order(self, order_id: int) -> dict:
        """Cancel an existing order"""
        try:
            with self.db_session_factory() as session:
                db_order = session.query(Order).filter(Order.id == order_id).first()
                if not db_order:
                    raise HTTPException(status_code=404, detail="Order not found")
                
                if db_order.status in [OrderStatus.FILLED.value, OrderStatus.CANCELLED.value]:
                    raise HTTPException(status_code=400, detail=f"Cannot cancel order with status: {db_order.status}")
                
                # Cancel with TWS if not in DRY_RUN mode
                if not self.settings.tws.dry_run and db_order.external_order_id:
                    trade = self.active_orders.get(order_id)
                    if trade:
                        self.ib_client.ib.cancelOrder(trade.order)
                        logger.info(f"Cancelled order with TWS: {db_order.external_order_id}")
                
                # Update database
                db_order.status = OrderStatus.CANCELLED.value
                db_order.updated_at = datetime.now(timezone.utc)
                session.commit()
                
                return {"order_id": order_id, "status": "cancelled"}
                
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_orders(self, limit: int = 100, status: Optional[str] = None) -> OrderListResponse:
        """Get order list with filtering"""
        try:
            with self.db_session_factory() as session:
                query = session.query(Order).order_by(desc(Order.placed_at))
                
                if status:
                    query = query.filter(Order.status == status)
                
                orders = query.limit(limit).all()
                
                order_responses = []
                for order in orders:
                    order_responses.append(OrderResponse(
                        id=order.id,
                        account_id=order.account_id,
                        strategy_id=order.strategy_id,
                        symbol=order.symbol,
                        side=OrderSide(order.side),
                        qty=float(order.qty),
                        order_type=OrderType(order.order_type),
                        limit_price=float(order.limit_price) if order.limit_price else None,
                        stop_price=float(order.stop_price) if order.stop_price else None,
                        tif=TimeInForce(order.tif),
                        status=OrderStatus(order.status),
                        external_order_id=order.external_order_id,
                        placed_at=order.placed_at,
                        updated_at=order.updated_at
                    ))
                
                return OrderListResponse(orders=order_responses, total=len(order_responses))
                
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_order(self, order_id: int) -> OrderResponse:
        """Get specific order details"""
        try:
            with self.db_session_factory() as session:
                order = session.query(Order).filter(Order.id == order_id).first()
                if not order:
                    raise HTTPException(status_code=404, detail="Order not found")
                
                return OrderResponse(
                    id=order.id,
                    account_id=order.account_id,
                    strategy_id=order.strategy_id,
                    symbol=order.symbol,
                    side=OrderSide(order.side),
                    qty=float(order.qty),
                    order_type=OrderType(order.order_type),
                    limit_price=float(order.limit_price) if order.limit_price else None,
                    stop_price=float(order.stop_price) if order.stop_price else None,
                    tif=TimeInForce(order.tif),
                    status=OrderStatus(order.status),
                    external_order_id=order.external_order_id,
                    placed_at=order.placed_at,
                    updated_at=order.updated_at
                )
                
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _update_health_status(self, status: str):
        """Update service health status in database"""
        try:
            with self.db_session_factory() as session:
                health = session.query(HealthStatus).filter(HealthStatus.service == "trader").first()
                if health:
                    health.status = status
                    health.updated_at = datetime.now(timezone.utc)
                else:
                    health = HealthStatus(
                        service="trader",
                        status=status,
                        updated_at=datetime.now(timezone.utc)
                    )
                    session.add(health)
                session.commit()
            
        except Exception as e:
            logger.error(f"Failed to update health status: {e}")
    
    async def get_health(self) -> HealthCheckResponse:
        """Get service health status"""
        try:
            tws_connected = self.ib_client and self.ib_client.state.connected if self.ib_client else False
            
            return HealthCheckResponse(
                service="trader",
                status="healthy" if tws_connected else "unhealthy",
                timestamp=datetime.now(timezone.utc),
                details={
                    "client_id": self.client_id,
                    "tws_connected": tws_connected,
                    "trading_mode": "dry_run" if self.settings.tws.dry_run else ("paper" if self.settings.tws.use_paper else "live"),
                    "active_orders": len(self.active_orders)
                }
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthCheckResponse(
                service="trader",
                status="unhealthy",
                timestamp=datetime.now(timezone.utc),
                details={"error": str(e)}
            )
    
    async def _periodic_health_update(self):
        """Periodically update health status in database"""
        while not self.shutdown_event.is_set():
            try:
                # Update health status based on TWS connection
                tws_connected = self.ib_client and self.ib_client.state.connected if self.ib_client else False
                status = "healthy" if tws_connected else "unhealthy"
                await self._update_health_status(status)
                
            except Exception as e:
                logger.error(f"Error in periodic health update: {e}")
            
            # Wait 30 seconds before next update
            await asyncio.sleep(30)
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down trader service...")
        self.shutdown_event.set()
        
        try:
            if self.ib_client:
                await self.ib_client.disconnect()
            
            if self.client_id:
                self.client_id_manager.release_client_id(self.client_id, "trader")
                
            await self._update_health_status("stopping")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# FastAPI application
app = FastAPI(title="Trader Service", version="1.0.0")
trader_service = TraderService()

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    print("DEBUG: startup_event called")
    setup_logging("trader")
    print("DEBUG: setup_logging completed")
    logger.info("Starting Trader Service...")
    print("DEBUG: About to call trader_service.initialize()")
    
    try:
        await trader_service.initialize()
        print("DEBUG: trader_service.initialize() completed")
        
        # Start periodic health update in background
        asyncio.create_task(trader_service._periodic_health_update())
        
        logger.info("Trader Service started successfully")
    except Exception as e:
        print(f"DEBUG: Exception in startup: {e}")
        logger.error(f"Failed to start Trader Service: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await trader_service.shutdown()

# API Endpoints
@app.get("/healthz", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return await trader_service.get_health()

@app.post("/orders", response_model=OrderResponse)
async def place_order(order_request: OrderRequest):
    """Place a new order"""
    # For now, use a default account - in production this would come from authentication
    account_id = "DU7084660"  # Default paper account
    return await trader_service.place_order(order_request, account_id)

@app.post("/cancel/{order_id}")
async def cancel_order(order_id: int):
    """Cancel an existing order"""
    return await trader_service.cancel_order(order_id)

@app.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(order_id: int):
    """Get specific order details"""
    return await trader_service.get_order(order_id)

@app.get("/orders", response_model=OrderListResponse)
async def get_orders(limit: int = 100, status: Optional[str] = None):
    """Get order list with filtering"""
    return await trader_service.get_orders(limit, status)

# Risk Management Endpoints
@app.post("/risk/emergency-stop")
async def activate_emergency_stop(reason: str = "Manual emergency stop"):
    """
    EMERGENCY STOP - Immediately halt all trading
    
    This endpoint activates the emergency stop flag in the database,
    which will prevent ALL new orders from being placed.
    """
    try:
        with get_db_session() as session:
            from common.models import RiskLimit
            
            # Check if emergency stop limit exists
            stop_limit = session.query(RiskLimit).filter(
                RiskLimit.key == 'emergency_stop'
            ).first()
            
            if stop_limit:
                stop_limit.value_json = True
                stop_limit.updated_at = datetime.now(timezone.utc)
            else:
                stop_limit = RiskLimit(
                    key='emergency_stop',
                    value_json=True,
                    updated_at=datetime.now(timezone.utc)
                )
                session.add(stop_limit)
            
            session.commit()
            
            logger.critical(f"🚨 EMERGENCY STOP ACTIVATED: {reason}")
            
            # Send emergency alert
            try:
                from common.risk_alerts import send_emergency_alert
                await send_emergency_alert(
                    message=f"EMERGENCY STOP ACTIVATED: {reason}",
                    metadata={'reason': reason, 'activated_at': datetime.now(timezone.utc).isoformat()}
                )
            except Exception as e:
                logger.error(f"Failed to send emergency alert: {e}")
            
            return {
                "status": "emergency_stop_activated",
                "message": "All trading has been halted",
                "reason": reason,
                "activated_at": datetime.now(timezone.utc).isoformat()
            }
    except Exception as e:
        logger.error(f"Failed to activate emergency stop: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to activate emergency stop: {str(e)}")

@app.delete("/risk/emergency-stop")
async def deactivate_emergency_stop():
    """
    Deactivate emergency stop and resume trading
    
    Use with caution - only deactivate when you're certain it's safe to resume.
    """
    try:
        with get_db_session() as session:
            from common.models import RiskLimit
            
            stop_limit = session.query(RiskLimit).filter(
                RiskLimit.key == 'emergency_stop'
            ).first()
            
            if stop_limit:
                stop_limit.value_json = False
                stop_limit.updated_at = datetime.now(timezone.utc)
                session.commit()
            
            logger.warning("✅ Emergency stop deactivated - trading resumed")
            
            return {
                "status": "emergency_stop_deactivated",
                "message": "Trading has been resumed",
                "deactivated_at": datetime.now(timezone.utc).isoformat()
            }
    except Exception as e:
        logger.error(f"Failed to deactivate emergency stop: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to deactivate emergency stop: {str(e)}")

@app.get("/risk/violations")
async def get_risk_violations(
    limit: int = 50,
    severity: Optional[str] = None,
    resolved: Optional[bool] = None
):
    """Get recent risk violations with optional filtering"""
    try:
        with get_db_session() as session:
            from common.models import RiskViolation
            
            query = session.query(RiskViolation)
            
            if severity:
                query = query.filter(RiskViolation.severity == severity)
            
            if resolved is not None:
                query = query.filter(RiskViolation.resolved == resolved)
            
            violations = query.order_by(RiskViolation.created_at.desc()).limit(limit).all()
            
            return {
                "violations": [
                    {
                        "id": v.id,
                        "violation_type": v.violation_type,
                        "severity": v.severity,
                        "account_id": v.account_id,
                        "symbol": v.symbol,
                        "strategy_id": v.strategy_id,
                        "message": v.message,
                        "limit_key": v.limit_key,
                        "limit_value": float(v.limit_value) if v.limit_value else None,
                        "actual_value": float(v.actual_value) if v.actual_value else None,
                        "action_taken": v.action_taken,
                        "resolved": v.resolved,
                        "created_at": v.created_at.isoformat(),
                        "metadata": v.metadata_json
                    }
                    for v in violations
                ],
                "total": len(violations),
                "filtered": f"severity={severity}" if severity else "all"
            }
    except Exception as e:
        logger.error(f"Failed to get risk violations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get risk violations: {str(e)}")

@app.get("/risk/status")
async def get_risk_status():
    """Get current risk management status"""
    try:
        with get_db_session() as session:
            from common.models import RiskLimit, RiskViolation
            
            # Get all active risk limits
            limits = session.query(RiskLimit).all()
            limits_dict = {limit.key: limit.value_json for limit in limits}
            
            # Get recent violations (last 24 hours)
            from sqlalchemy import func
            from datetime import timedelta
            
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            recent_violations = session.query(RiskViolation).filter(
                RiskViolation.created_at >= cutoff
            ).count()
            
            critical_violations = session.query(RiskViolation).filter(
                and_(
                    RiskViolation.created_at >= cutoff,
                    RiskViolation.severity == 'critical'
                )
            ).count()
            
            unresolved_violations = session.query(RiskViolation).filter(
                RiskViolation.resolved == False
            ).count()
            
            emergency_stop = limits_dict.get('emergency_stop', False)
            
            return {
                "emergency_stop_active": emergency_stop,
                "risk_limits": limits_dict,
                "violations_last_24h": recent_violations,
                "critical_violations_last_24h": critical_violations,
                "unresolved_violations": unresolved_violations,
                "status": "critical" if emergency_stop else ("warning" if unresolved_violations > 0 else "healthy"),
                "checked_at": datetime.now(timezone.utc).isoformat()
            }
    except Exception as e:
        logger.error(f"Failed to get risk status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get risk status: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time order updates"""
    await trader_service.websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            # Echo back for testing
            await websocket.send_json({"type": "echo", "data": data})
    except WebSocketDisconnect:
        trader_service.websocket_manager.disconnect(websocket)

# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}")
    asyncio.create_task(trader_service.shutdown())

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    uvicorn.run(app, host="0.0.0.0", port=8004, log_level="info")
