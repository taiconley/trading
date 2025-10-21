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
            await self.initialize()
            
            # Start main execution loop
            self.running = True
            
            # Start background tasks
            asyncio.create_task(self._strategy_execution_loop())
            asyncio.create_task(self._strategy_reload_loop())
            asyncio.create_task(self._heartbeat_loop())
            
            logger.info("Strategy service started successfully")
            
        except Exception as e:
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
        logger.info("Starting strategy execution loop")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Process each active strategy
                for strategy_id, runner in list(self.strategy_runners.items()):
                    if not runner.is_running or not runner.strategy.config.enabled:
                        continue
                    
                    # Get latest bars for each symbol
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
            # Get latest bars from database
            timeframe = runner.strategy.config.bar_timeframe
            lookback = runner.strategy.config.lookback_periods
            
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
    
    async def _get_latest_bars(self, symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
        """Get latest bars from database."""
        try:
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
    
    async def _handle_strategy_signals(self, runner: StrategyRunner, signals: List[Any]):
        """Handle signals generated by a strategy."""
        for signal in signals:
            try:
                # Store signal in database
                await self._store_signal(signal, runner.strategy.config.strategy_id)
                
                # Check if strategy is enabled and risk limits allow trading
                if runner.strategy.config.enabled and await self._should_execute_signal(signal):
                    # Place order through Trader service
                    await self._place_order_for_signal(signal, runner.strategy.config.strategy_id)
                
                # Broadcast signal via WebSocket
                await self._broadcast_signal(signal, runner.strategy.config.strategy_id)
                
            except Exception as e:
                logger.error(f"Error handling signal: {e}")
    
    async def _store_signal(self, signal: Any, strategy_id: str):
        """Store signal in database."""
        try:
            with self.db_session_factory() as db:
                db_signal = Signal(
                    strategy_id=strategy_id,
                    symbol=signal.symbol,
                    signal_type=signal.signal_type.value,
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
        """Place order through Trader service REST API."""
        try:
            # Determine order side
            order_side = "BUY" if signal.signal_type == SignalType.BUY else "SELL"
            
            # Build order request
            order_request = {
                "symbol": signal.symbol,
                "side": order_side,
                "quantity": signal.quantity or 100,
                "order_type": "LMT" if signal.price else "MKT",
                "limit_price": float(signal.price) if signal.price else None,
                "time_in_force": "DAY",
                "strategy_id": strategy_id
            }
            
            # Send to Trader service
            trader_url = f"http://backend-trader:8004/orders"  # Internal Docker network
            
            async with aiohttp.ClientSession() as session:
                async with session.post(trader_url, json=order_request) as response:
                    if response.status == 200:
                        order_data = await response.json()
                        logger.info(f"Order placed for signal: {order_data.get('order_id')}")
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to place order: {response.status} - {error_text}")
                        
        except Exception as e:
            logger.error(f"Error placing order for signal: {e}")
    
    async def _broadcast_signal(self, signal: Any, strategy_id: str):
        """Broadcast signal via WebSocket."""
        try:
            signal_data = {
                "type": "signal_generated",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "strategy_id": strategy_id,
                    "symbol": signal.symbol,
                    "signal_type": signal.signal_type.value,
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
                # Get latest strategy update time
                latest_update = db.query(func.max(Strategy.created_at)).scalar()
                
                if latest_update and latest_update > self.last_strategy_reload:
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


# Global service instance
strategy_service = StrategyService()


async def main():
    """Main entry point."""
    # Setup logging
    setup_logging("strategy")
    logger.info("Starting Strategy Service...")
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(strategy_service.stop())
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start service
        await strategy_service.start()
        
        # Start FastAPI server
        config = uvicorn.Config(
            app=strategy_service.app,
            host="0.0.0.0",
            port=8005,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        # Run server
        await server.serve()
        
    except Exception as e:
        logger.error(f"Strategy service failed: {e}")
        raise
    finally:
        await strategy_service.stop()


if __name__ == "__main__":
    asyncio.run(main())