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
    """Risk management and validation"""
    
    def __init__(self, db_session_factory):
        self.db_session_factory = db_session_factory
        self.risk_limits_cache = {}
        self.last_cache_update = None
        self.cache_ttl = 60  # seconds
    
    async def validate_order(self, order_request: OrderRequest, account_id: str) -> tuple[bool, str]:
        """
        Validate order against risk limits.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            # Refresh risk limits cache if needed
            await self._refresh_risk_limits()
            
            # Check individual order size
            notional = float(order_request.quantity) * (order_request.limit_price or 0)
            max_notional_per_order = self.risk_limits_cache.get('max_notional_per_order', 100000)
            
            if notional > max_notional_per_order:
                return False, f"Order notional ${notional:,.2f} exceeds limit ${max_notional_per_order:,.2f}"
            
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
                    return False, f"Position would exceed symbol limit: ${position_notional:,.2f} > ${max_position_size:,.2f}"
            
            # Check daily loss limits (simplified - would need P&L calculation)
            max_daily_loss = self.risk_limits_cache.get('max_daily_loss', 10000)
            
            # Check if live trading is blocked
            block_until = self.risk_limits_cache.get('block_live_trading_until')
            if block_until and isinstance(block_until, str):
                try:
                    block_datetime = datetime.fromisoformat(block_until.replace('Z', '+00:00'))
                    if datetime.now(timezone.utc) < block_datetime:
                        return False, f"Live trading blocked until {block_until}"
                except Exception as e:
                    logger.warning(f"Error parsing block_live_trading_until: {e}")
            
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
        """Initialize the trader service"""
        try:
            # Get client ID
            self.client_id = self.client_id_manager.get_service_client_id("trader")
            logger.info(f"Allocated client ID: {self.client_id}")
            
            # Initialize IB client
            self.ib_client = EnhancedIBClient(
                client_id=self.client_id,
                host=self.settings.tws.host,
                port=self.settings.tws.port
            )
            
            # Set up event handlers following notes.md best practices
            self.ib_client.ib.orderStatusEvent += self._on_order_status
            self.ib_client.ib.execDetailsEvent += self._on_execution
            
            # Connect to TWS
            await self.ib_client.connect()
            logger.info("Connected to TWS successfully")
            
            # Update health status
            await self._update_health_status("healthy")
            
        except Exception as e:
            logger.error(f"Failed to initialize trader service: {e}")
            await self._update_health_status("unhealthy")
            raise
    
    async def _on_order_status(self, trade: Trade):
        """Handle order status updates from TWS"""
        try:
            order_status = trade.orderStatus.status
            ib_order_id = trade.order.orderId
            
            logger.info(f"Order status update: {ib_order_id} -> {order_status}")
            
            # Update database
            with self.db_session_factory() as session:
                db_order = session.query(Order).filter(
                    Order.external_order_id == str(ib_order_id)
                ).first()
                
                if db_order:
                    db_order.status = order_status
                    db_order.updated_at = datetime.now(timezone.utc)
                    session.commit()
                    
                    # Broadcast update via WebSocket
                    await self.websocket_manager.broadcast({
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
                    })
                else:
                    logger.warning(f"Order not found in database: {ib_order_id}")
                    
        except Exception as e:
            logger.error(f"Error handling order status update: {e}")
    
    async def _on_execution(self, trade: Trade, fill):
        """Handle execution reports from TWS"""
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
                        symbol=execution.contract.symbol,
                        qty=Decimal(str(execution.shares)),
                        price=Decimal(str(execution.price)),
                        ts=datetime.now(timezone.utc)
                    )
                    session.add(db_execution)
                    session.commit()
                    
                    # Broadcast execution via WebSocket
                    await self.websocket_manager.broadcast({
                        'type': 'execution',
                        'data': {
                            'execution_id': db_execution.id,
                            'order_id': db_order.id,
                            'trade_id': execution.execId,
                            'symbol': execution.contract.symbol,
                            'qty': float(execution.shares),
                            'price': float(execution.price),
                            'timestamp': db_execution.ts.isoformat()
                        }
                    })
                else:
                    logger.warning(f"Order not found for execution: {trade.order.orderId}")
                    
        except Exception as e:
            logger.error(f"Error handling execution: {e}")
    
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
            is_valid, error_msg = await self.risk_manager.validate_order(order_request, account_id)
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
    setup_logging("trader")
    logger.info("Starting Trader Service...")
    
    try:
        await trader_service.initialize()
        logger.info("Trader Service started successfully")
    except Exception as e:
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
