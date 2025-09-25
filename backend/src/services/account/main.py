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
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from ib_insync import IB

# Add the backend src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.config import get_settings
from common.logging import setup_logging

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
            await self.ib.reqPositionsAsync()
            logger.info("   ✅ Position updates started")
            
            # Start P&L updates
            logger.info("   💰 Starting P&L updates...")
            await self.ib.reqPnLAsync(self.account, '')
            logger.info("   ✅ P&L updates started")
            
            # Start account summary updates
            logger.info("   📈 Starting account summary updates...")
            await self.ib.reqAccountSummaryAsync()
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
    
    async def stop(self):
        """Stop the account service"""
        logger.info("🛑 Stopping Account Service...")
        
        self.running = False
        
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("✅ Disconnected from TWS")
        
        logger.info("👋 Account Service stopped")
    
    def create_app(self) -> FastAPI:
        """Create FastAPI application"""
        app = FastAPI(
            title="Account Service",
            description="Real-time TWS account data streaming service",
            version="1.0.0"
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
        
        @app.get("/account/stats")
        async def get_account_stats():
            """Get current account stats"""
            return {
                "account": self.account, 
                "stats": self.stats,
                "tws_connected": self.ib.isConnected()
            }
        
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

async def shutdown():
    """Graceful shutdown"""
    logger.info("📶 Shutting down Account Service...")
    await service.stop()

async def main():
    """Main entry point"""
    setup_logging("account")
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"📶 Received signal {sig}")
        asyncio.create_task(shutdown())
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Create FastAPI server first
        app = service.create_app()
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8001,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        logger.info("🌐 Starting FastAPI server on port 8001...")
        
        # Start TWS connection in background
        tws_task = asyncio.create_task(service.start_tws_connection())
        
        # Run server (this will block)
        await server.serve()
        
    except KeyboardInterrupt:
        logger.info("👋 Account Service stopped by user")
    except Exception as e:
        logger.error(f"💥 Account Service error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        await service.stop()

if __name__ == "__main__":
    asyncio.run(main())