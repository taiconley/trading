"""
Historical Data Service - Batched historical data pulls with pacing and queueing.

This service handles historical data requests from TWS/IB Gateway with proper
pacing controls, request queueing, and idempotent storage to the candles table.
Follows TWS best practices for historical data requests.
"""

import asyncio
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from ib_insync import IB, Stock, BarData

# Import our common modules
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from common.config import get_settings
from common.db import get_db_session, execute_with_retry, initialize_database
from common.models import Candle, Symbol, HealthStatus, WatchlistEntry
from common.logging import configure_service_logging, log_system_event
from tws_bridge.ib_client import create_ib_client
from tws_bridge.client_ids import allocate_service_client_id, heartbeat_service_client_id, release_service_client_id


class RequestStatus(Enum):
    """Status of historical data requests."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"


@dataclass
class HistoricalRequest:
    """Historical data request."""
    id: str
    symbol: str
    bar_size: str
    what_to_show: str
    duration: str
    end_datetime: str
    use_rth: bool
    status: RequestStatus = RequestStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    bars_count: int = 0


class HistoricalDataService:
    """Historical data service with request queueing and pacing controls."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = configure_service_logging("historical")
        self.app = FastAPI(title="Historical Data Service")
        
        # TWS connection
        self.client_id = None
        self.ib_client = None
        self.connected = False
        
        # Request management
        self.request_queue: asyncio.Queue[HistoricalRequest] = asyncio.Queue()
        self.active_requests: Dict[str, HistoricalRequest] = {}
        self.completed_requests: Dict[str, HistoricalRequest] = {}
        self.max_requests_per_minute = self.settings.historical.max_requests_per_min
        self.request_timestamps: List[float] = []
        
        # Service state
        self.running = False
        self.processing_requests = False
        self.last_heartbeat = time.time()
        
        # Setup FastAPI routes
        self._setup_routes()
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/healthz")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy" if self.connected else "unhealthy",
                "service": "historical",
                "client_id": self.client_id,
                "connected": self.connected,
                "queue_size": self.request_queue.qsize(),
                "active_requests": len(self.active_requests),
                "completed_requests": len(self.completed_requests),
                "processing": self.processing_requests,
                "max_requests_per_minute": self.max_requests_per_minute,
                "last_heartbeat": self.last_heartbeat
            }
        
        @self.app.get("/queue/status")
        async def queue_status():
            """Get request queue status."""
            return {
                "queue_size": self.request_queue.qsize(),
                "active_requests": [
                    {
                        "id": req.id,
                        "symbol": req.symbol,
                        "bar_size": req.bar_size,
                        "status": req.status.value,
                        "started_at": req.started_at.isoformat() if req.started_at else None
                    }
                    for req in self.active_requests.values()
                ],
                "recent_completions": [
                    {
                        "id": req.id,
                        "symbol": req.symbol,
                        "bar_size": req.bar_size,
                        "status": req.status.value,
                        "bars_count": req.bars_count,
                        "completed_at": req.completed_at.isoformat() if req.completed_at else None,
                        "error": req.error_message
                    }
                    for req in list(self.completed_requests.values())[-10:]  # Last 10
                ]
            }
        
        @self.app.post("/historical/request")
        async def request_historical_data(
            symbol: str,
            bar_size: str = "1 min",
            what_to_show: str = "TRADES",
            duration: str = "1 D",
            end_datetime: str = "",
            use_rth: bool = True
        ):
            """Request historical data for a symbol."""
            try:
                request_id = f"{symbol}_{bar_size}_{duration}_{int(time.time())}"
                
                # Validate bar size
                valid_bar_sizes = self.settings.historical.bar_sizes_list
                if bar_size not in valid_bar_sizes:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid bar_size. Must be one of: {valid_bar_sizes}"
                    )
                
                # Create request
                request = HistoricalRequest(
                    id=request_id,
                    symbol=symbol,
                    bar_size=bar_size,
                    what_to_show=what_to_show,
                    duration=duration,
                    end_datetime=end_datetime,
                    use_rth=use_rth
                )
                
                # Add to queue
                await self.request_queue.put(request)
                
                self.logger.info(f"Queued historical data request: {request_id}")
                
                return {
                    "request_id": request_id,
                    "status": "queued",
                    "queue_position": self.request_queue.qsize()
                }
                
            except Exception as e:
                self.logger.error(f"Failed to queue historical data request: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/historical/bulk")
        async def bulk_historical_request():
            """Request historical data for all watchlist symbols."""
            try:
                # Get watchlist symbols
                def _get_watchlist(session):
                    entries = session.query(WatchlistEntry).all()
                    return [entry.symbol for entry in entries]
                
                symbols = execute_with_retry(_get_watchlist)
                
                if not symbols:
                    return {"message": "No symbols in watchlist", "requests": []}
                
                # Create requests for each symbol and bar size
                requests = []
                bar_sizes = self.settings.historical.bar_sizes_list
                
                for symbol in symbols:
                    for bar_size in bar_sizes:
                        request_id = f"bulk_{symbol}_{bar_size}_{int(time.time())}"
                        
                        request = HistoricalRequest(
                            id=request_id,
                            symbol=symbol,
                            bar_size=bar_size,
                            what_to_show=self.settings.market_data.what_to_show,
                            duration=self.settings.market_data.lookback,
                            end_datetime="",
                            use_rth=self.settings.market_data.rth
                        )
                        
                        await self.request_queue.put(request)
                        requests.append(request_id)
                
                self.logger.info(f"Queued {len(requests)} bulk historical requests")
                
                return {
                    "message": f"Queued {len(requests)} historical data requests",
                    "symbols": symbols,
                    "bar_sizes": bar_sizes,
                    "requests": requests
                }
                
            except Exception as e:
                self.logger.error(f"Failed to queue bulk historical requests: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/historical/request/{request_id}")
        async def get_request_status(request_id: str):
            """Get status of a specific historical data request."""
            # Check active requests
            if request_id in self.active_requests:
                req = self.active_requests[request_id]
                return {
                    "request_id": request_id,
                    "status": req.status.value,
                    "symbol": req.symbol,
                    "bar_size": req.bar_size,
                    "started_at": req.started_at.isoformat() if req.started_at else None,
                    "bars_count": req.bars_count
                }
            
            # Check completed requests
            if request_id in self.completed_requests:
                req = self.completed_requests[request_id]
                return {
                    "request_id": request_id,
                    "status": req.status.value,
                    "symbol": req.symbol,
                    "bar_size": req.bar_size,
                    "started_at": req.started_at.isoformat() if req.started_at else None,
                    "completed_at": req.completed_at.isoformat() if req.completed_at else None,
                    "bars_count": req.bars_count,
                    "error": req.error_message
                }
            
            raise HTTPException(status_code=404, detail="Request not found")
    
    async def start(self):
        """Start the historical data service."""
        try:
            log_system_event(self.logger, "service_start", service="historical")
            
            # Initialize database
            initialize_database()
            
            # Allocate client ID
            self.client_id = allocate_service_client_id("historical")
            self.logger.info(f"Allocated TWS client ID: {self.client_id}")
            
            # Create IB client
            self.ib_client = create_ib_client(self.client_id)
            
            # Setup event handlers
            self._setup_event_handlers()
            
            # Connect to TWS
            await self._connect_to_tws()
            
            # Update health status
            await self._update_health_status("healthy")
            
            # Start request processing task
            asyncio.create_task(self._process_requests())
            
            # Start heartbeat task
            asyncio.create_task(self._heartbeat_loop())
            
            self.running = True
            self.logger.info("Historical data service started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start historical data service: {e}")
            await self._update_health_status("unhealthy")
            raise
    
    async def stop(self):
        """Stop the historical data service."""
        try:
            log_system_event(self.logger, "service_stop", service="historical")
            
            self.running = False
            self.processing_requests = False
            
            # Disconnect from TWS
            if self.ib_client:
                await self.ib_client.disconnect()
                self.connected = False
            
            # Release client ID
            if self.client_id:
                release_service_client_id(self.client_id, "historical")
            
            # Update health status
            await self._update_health_status("stopping")
            
            self.logger.info("Historical data service stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping historical data service: {e}")
    
    def _setup_event_handlers(self):
        """Setup TWS event handlers."""
        
        def on_error(req_id, error_code, error_string, contract):
            """Handle TWS errors."""
            if error_code == 200:  # No security definition found
                symbol = contract.symbol if contract else "unknown"
                self.logger.warning(f"No security definition for {symbol}")
            elif error_code == 162:  # Historical data request pacing violation
                self.logger.warning(f"Historical data pacing violation: {error_string}")
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
    
    async def _process_requests(self):
        """Process historical data requests from the queue."""
        self.processing_requests = True
        self.logger.info("Started historical data request processing")
        
        while self.running:
            try:
                # Wait for a request (with timeout to allow checking running status)
                try:
                    request = await asyncio.wait_for(self.request_queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                
                # Check if we need to wait for pacing
                await self._wait_for_pacing()
                
                # Process the request
                await self._process_single_request(request)
                
                # Mark task as done
                self.request_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in request processing loop: {e}")
                await asyncio.sleep(1)
        
        self.processing_requests = False
        self.logger.info("Stopped historical data request processing")
    
    async def _wait_for_pacing(self):
        """Wait if needed to respect pacing limits."""
        current_time = time.time()
        
        # Remove old timestamps (older than 1 minute)
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if current_time - ts < 60.0
        ]
        
        # Check if we need to wait
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            wait_time = 60.0 - (current_time - self.request_timestamps[0])
            if wait_time > 0:
                self.logger.info(f"Pacing: waiting {wait_time:.1f}s before next request")
                await asyncio.sleep(wait_time)
    
    async def _process_single_request(self, request: HistoricalRequest):
        """Process a single historical data request."""
        try:
            self.logger.info(f"Processing historical request: {request.id}")
            
            # Move to active requests
            request.status = RequestStatus.IN_PROGRESS
            request.started_at = datetime.now(timezone.utc)
            self.active_requests[request.id] = request
            
            # Record request timestamp for pacing
            self.request_timestamps.append(time.time())
            
            if not self.connected:
                raise Exception("Not connected to TWS")
            
            # Create contract
            contract = Stock(request.symbol, 'SMART', 'USD')
            
            # Request historical data (following notes.md pattern)
            bars = await self.ib_client.req_historical_data_with_retry(
                contract=contract,
                end_datetime=request.end_datetime,
                duration_str=request.duration,
                bar_size_setting=request.bar_size,
                what_to_show=request.what_to_show,
                use_rth=request.use_rth
            )
            
            # Store bars in database
            bars_stored = await self._store_bars(request.symbol, request.bar_size, bars)
            
            # Mark as completed
            request.status = RequestStatus.COMPLETED
            request.completed_at = datetime.now(timezone.utc)
            request.bars_count = bars_stored
            
            # Move to completed requests
            self.completed_requests[request.id] = request
            if request.id in self.active_requests:
                del self.active_requests[request.id]
            
            self.logger.info(f"Completed historical request {request.id}: {bars_stored} bars stored")
            
        except Exception as e:
            self.logger.error(f"Failed to process historical request {request.id}: {e}")
            
            # Mark as failed
            request.status = RequestStatus.FAILED
            request.completed_at = datetime.now(timezone.utc)
            request.error_message = str(e)
            
            # Move to completed requests
            self.completed_requests[request.id] = request
            if request.id in self.active_requests:
                del self.active_requests[request.id]
    
    async def _store_bars(self, symbol: str, timeframe: str, bars: List[BarData]) -> int:
        """Store historical bars in database with idempotent upserts."""
        if not bars:
            return 0
        
        def _upsert_bars(session):
            bars_stored = 0
            
            for bar in bars:
                # Convert bar data
                bar_data = {
                    'symbol': symbol,
                    'tf': timeframe,
                    'ts': bar.date,
                    'open': Decimal(str(bar.open)),
                    'high': Decimal(str(bar.high)),
                    'low': Decimal(str(bar.low)),
                    'close': Decimal(str(bar.close)),
                    'volume': int(bar.volume) if bar.volume else 0
                }
                
                # Check if bar already exists (idempotent)
                existing = session.query(Candle).filter(
                    Candle.symbol == symbol,
                    Candle.tf == timeframe,
                    Candle.ts == bar.date
                ).first()
                
                if existing:
                    # Update existing bar
                    for key, value in bar_data.items():
                        if key not in ['symbol', 'tf', 'ts']:  # Don't update key fields
                            setattr(existing, key, value)
                else:
                    # Insert new bar
                    candle = Candle(**bar_data)
                    session.add(candle)
                    bars_stored += 1
            
            session.commit()
            return bars_stored
        
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: execute_with_retry(_upsert_bars)
        )
    
    async def _heartbeat_loop(self):
        """Heartbeat loop for health monitoring."""
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
                    except Exception as e:
                        self.logger.error(f"Reconnection failed: {e}")
                
                # Update health status
                status = "healthy" if self.connected else "unhealthy"
                await self._update_health_status(status)
                
                # Clean up old completed requests (keep last 100)
                if len(self.completed_requests) > 100:
                    # Keep only the most recent 50
                    sorted_requests = sorted(
                        self.completed_requests.items(),
                        key=lambda x: x[1].completed_at or x[1].created_at,
                        reverse=True
                    )
                    self.completed_requests = dict(sorted_requests[:50])
                
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(30)
    
    async def _update_health_status(self, status: str):
        """Update service health status in database."""
        try:
            def _update_health(session):
                health_record = session.query(HealthStatus).filter(
                    HealthStatus.service == "historical"
                ).first()
                
                if health_record:
                    health_record.status = status
                    health_record.updated_at = datetime.now(timezone.utc)
                else:
                    health_record = HealthStatus(
                        service="historical",
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
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(self.stop())


async def main():
    """Main entry point for the historical data service."""
    service = HistoricalDataService()
    
    try:
        # Start the service
        await service.start()
        
        # Start FastAPI server
        config = uvicorn.Config(
            service.app,
            host="0.0.0.0",
            port=8003,
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
    asyncio.run(main())
