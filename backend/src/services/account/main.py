#!/usr/bin/env python3
"""
Account Service - Real-time TWS Account Data Streaming
Streams account values, positions, and P&L data to database and WebSocket clients.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from tws_bridge.base_service import BaseTWSService
from common.db import get_db_connection
from common.logging import setup_logging
from common.notify import NotificationManager

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

class AccountService(BaseTWSService):
    """Account Service for streaming TWS account data"""
    
    def __init__(self):
        super().__init__(
            service_name="account",
            client_id_service="account",
            port=8001
        )
        self.websocket_manager = WebSocketManager()
        self.notification_manager = NotificationManager()
        self.account = None
        self.db_connection = None
        
        # Counters for monitoring
        self.stats = {
            'account_updates': 0,
            'position_updates': 0,
            'pnl_updates': 0,
            'summary_updates': 0,
            'last_update': None
        }
    
    async def setup_service(self):
        """Initialize service-specific setup"""
        logger.info("Setting up Account Service...")
        
        # Get database connection
        try:
            self.db_connection = await get_db_connection()
            logger.info("✅ Database connection established")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
        
        # Get managed accounts
        accounts = self.ib_client.ib.managedAccounts()
        if not accounts:
            raise RuntimeError("No managed accounts found")
        
        self.account = accounts[0]
        logger.info(f"✅ Using account: {self.account}")
        
        # Set up event handlers for streaming data
        self.setup_event_handlers()
        
        # Start streaming subscriptions
        await self.start_streaming()
        
        logger.info("🚀 Account Service setup complete")
    
    def setup_event_handlers(self):
        """Set up TWS event handlers for real-time data streaming"""
        logger.info("📡 Setting up TWS event handlers...")
        
        # Account value updates (balance, margin, etc.)
        def on_account_value(account_value):
            if account_value.account == self.account:
                asyncio.create_task(self.handle_account_value_update(account_value))
        
        # Position updates
        def on_position_update(position):
            if position.account == self.account:
                asyncio.create_task(self.handle_position_update(position))
        
        # P&L updates
        def on_pnl_update(pnl):
            if hasattr(pnl, 'account') and pnl.account == self.account:
                asyncio.create_task(self.handle_pnl_update(pnl))
        
        # Account summary updates
        def on_account_summary_update(account_summary):
            if account_summary.account == self.account:
                asyncio.create_task(self.handle_account_summary_update(account_summary))
        
        # Connect event handlers
        self.ib_client.ib.accountValueEvent += on_account_value
        self.ib_client.ib.positionEvent += on_position_update
        self.ib_client.ib.pnlEvent += on_pnl_update
        self.ib_client.ib.accountSummaryEvent += on_account_summary_update
        
        logger.info("✅ Event handlers connected")
    
    async def start_streaming(self):
        """Start all streaming subscriptions"""
        logger.info(f"📡 Starting streaming subscriptions for account: {self.account}")
        
        try:
            # Start account value updates
            logger.info("   📊 Starting account value updates...")
            await self.ib_client.ib.reqAccountUpdatesAsync(self.account)
            logger.info("   ✅ Account value updates started")
            
            # Start position updates
            logger.info("   📍 Starting position updates...")
            await self.ib_client.ib.reqPositionsAsync()
            logger.info("   ✅ Position updates started")
            
            # Start P&L updates
            logger.info("   💰 Starting P&L updates...")
            await self.ib_client.ib.reqPnLAsync(self.account, '')
            logger.info("   ✅ P&L updates started")
            
            # Start account summary updates
            logger.info("   📈 Starting account summary updates...")
            await self.ib_client.ib.reqAccountSummaryAsync()
            logger.info("   ✅ Account summary updates started")
            
            logger.info("🎯 All streaming subscriptions active")
            
        except Exception as e:
            logger.error(f"❌ Error starting subscriptions: {e}")
            raise
    
    async def handle_account_value_update(self, account_value):
        """Handle account value updates (balance, margin, etc.)"""
        try:
            self.stats['account_updates'] += 1
            self.stats['last_update'] = datetime.now(timezone.utc)
            
            # Store in database
            await self.store_account_value(account_value)
            
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
            if account_value.tag in ['NetLiquidation', 'TotalCashValue', 'BuyingPower']:
                logger.info(f"💰 {account_value.tag}: {account_value.value} {account_value.currency}")
                
        except Exception as e:
            logger.error(f"Error handling account value update: {e}")
    
    async def handle_position_update(self, position):
        """Handle position updates"""
        try:
            self.stats['position_updates'] += 1
            self.stats['last_update'] = datetime.now(timezone.utc)
            
            # Store in database
            await self.store_position(position)
            
            # Broadcast to WebSocket clients
            await self.websocket_manager.broadcast({
                'type': 'position',
                'data': {
                    'account': position.account,
                    'symbol': position.contract.symbol,
                    'position': position.position,
                    'avg_cost': position.avgCost,
                    'market_price': position.marketPrice,
                    'market_value': position.marketValue,
                    'unrealized_pnl': position.unrealizedPNL,
                    'timestamp': self.stats['last_update'].isoformat()
                }
            })
            
            logger.info(f"📍 Position: {position.contract.symbol} = {position.position} @ {position.avgCost}")
            
        except Exception as e:
            logger.error(f"Error handling position update: {e}")
    
    async def handle_pnl_update(self, pnl):
        """Handle P&L updates"""
        try:
            self.stats['pnl_updates'] += 1
            self.stats['last_update'] = datetime.now(timezone.utc)
            
            # Store in database
            await self.store_pnl(pnl)
            
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
            
            # Store in database
            await self.store_account_summary(account_summary)
            
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
    
    async def store_account_value(self, account_value):
        """Store account value in database"""
        query = """
            INSERT INTO account_values (account, tag, value, currency, timestamp)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (account, tag, currency) 
            DO UPDATE SET 
                value = EXCLUDED.value,
                timestamp = EXCLUDED.timestamp
        """
        await self.db_connection.execute(
            query,
            account_value.account,
            account_value.tag,
            account_value.value,
            account_value.currency,
            self.stats['last_update']
        )
    
    async def store_position(self, position):
        """Store position in database"""
        query = """
            INSERT INTO positions (
                account, symbol, position, avg_cost, market_price, 
                market_value, unrealized_pnl, timestamp
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (account, symbol) 
            DO UPDATE SET 
                position = EXCLUDED.position,
                avg_cost = EXCLUDED.avg_cost,
                market_price = EXCLUDED.market_price,
                market_value = EXCLUDED.market_value,
                unrealized_pnl = EXCLUDED.unrealized_pnl,
                timestamp = EXCLUDED.timestamp
        """
        await self.db_connection.execute(
            query,
            position.account,
            position.contract.symbol,
            position.position,
            position.avgCost,
            position.marketPrice,
            position.marketValue,
            position.unrealizedPNL,
            self.stats['last_update']
        )
    
    async def store_pnl(self, pnl):
        """Store P&L in database"""
        query = """
            INSERT INTO pnl_data (
                account, daily_pnl, unrealized_pnl, realized_pnl, timestamp
            )
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (account) 
            DO UPDATE SET 
                daily_pnl = EXCLUDED.daily_pnl,
                unrealized_pnl = EXCLUDED.unrealized_pnl,
                realized_pnl = EXCLUDED.realized_pnl,
                timestamp = EXCLUDED.timestamp
        """
        await self.db_connection.execute(
            query,
            getattr(pnl, 'account', self.account),
            pnl.dailyPnL,
            pnl.unrealizedPnL,
            pnl.realizedPnL,
            self.stats['last_update']
        )
    
    async def store_account_summary(self, account_summary):
        """Store account summary in database"""
        query = """
            INSERT INTO account_summary (account, tag, value, currency, timestamp)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (account, tag) 
            DO UPDATE SET 
                value = EXCLUDED.value,
                currency = EXCLUDED.currency,
                timestamp = EXCLUDED.timestamp
        """
        await self.db_connection.execute(
            query,
            account_summary.account,
            account_summary.tag,
            account_summary.value,
            account_summary.currency,
            self.stats['last_update']
        )
    
    async def cleanup_service(self):
        """Cleanup service resources"""
        logger.info("🧹 Cleaning up Account Service...")
        
        if self.db_connection:
            await self.db_connection.close()
            logger.info("✅ Database connection closed")
    
    def create_app(self) -> FastAPI:
        """Create FastAPI application with account-specific endpoints"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self.start()
            yield
            # Shutdown
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
            if not self.ib_client or not self.ib_client.ib.isConnected():
                raise HTTPException(status_code=503, detail="TWS not connected")
            
            return JSONResponse({
                "status": "healthy",
                "service": "account",
                "tws_connected": self.ib_client.ib.isConnected(),
                "account": self.account,
                "stats": self.stats,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        @app.get("/account/values")
        async def get_account_values():
            """Get current account values"""
            if not self.db_connection:
                raise HTTPException(status_code=503, detail="Database not connected")
            
            query = "SELECT * FROM account_values WHERE account = $1 ORDER BY timestamp DESC"
            rows = await self.db_connection.fetch(query, self.account)
            
            return [dict(row) for row in rows]
        
        @app.get("/account/positions")
        async def get_positions():
            """Get current positions"""
            if not self.db_connection:
                raise HTTPException(status_code=503, detail="Database not connected")
            
            query = "SELECT * FROM positions WHERE account = $1 ORDER BY timestamp DESC"
            rows = await self.db_connection.fetch(query, self.account)
            
            return [dict(row) for row in rows]
        
        @app.get("/account/pnl")
        async def get_pnl():
            """Get current P&L data"""
            if not self.db_connection:
                raise HTTPException(status_code=503, detail="Database not connected")
            
            query = "SELECT * FROM pnl_data WHERE account = $1 ORDER BY timestamp DESC LIMIT 1"
            row = await self.db_connection.fetchrow(query, self.account)
            
            return dict(row) if row else None
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time data"""
            await self.websocket_manager.connect(websocket)
            try:
                while True:
                    # Keep connection alive
                    await asyncio.sleep(1)
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)
        
        return app

# Service instance
service = AccountService()

async def main():
    """Main entry point"""
    setup_logging("account")
    
    try:
        logger.info("🚀 Starting Account Service...")
        app = service.create_app()
        
        # Run with uvicorn
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8001,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    except KeyboardInterrupt:
        logger.info("👋 Account Service stopped by user")
    except Exception as e:
        logger.error(f"💥 Account Service error: {e}")
        raise
    finally:
        await service.stop()

if __name__ == "__main__":
    asyncio.run(main())