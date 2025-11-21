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
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from ib_insync import IB, Stock, BarData

# Import our common modules
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from common.config import get_settings
from common.db import get_db_session, execute_with_retry, initialize_database
from common.models import (
    Candle,
    Symbol,
    HealthStatus,
    WatchlistEntry,
    HistoricalJob,
    HistoricalJobChunk,
    HistoricalCoverage
)
from common.logging import configure_service_logging, log_system_event
from tws_bridge.ib_client import create_ib_client
from tws_bridge.client_ids import allocate_service_client_id, heartbeat_service_client_id, release_service_client_id
from sqlalchemy import func


class RequestStatus(Enum):
    """Status of historical data requests."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    SKIPPED = "skipped"


class BulkHistoricalRequestBody(BaseModel):
    """Request body for bulk historical data requests."""
    bar_size: Optional[str] = None
    duration: Optional[str] = None


class CustomBulkHistoricalRequestBody(BaseModel):
    """Request body for bulk historical data requests with explicit symbols."""
    symbols: List[str]
    bar_size: Optional[str] = None
    duration: Optional[str] = None
    end_datetime: Optional[str] = None
    use_rth: Optional[bool] = None
    what_to_show: Optional[str] = None


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
    job_id: Optional[int] = None
    chunk_id: Optional[int] = None
    chunk_index: Optional[int] = None
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None
    duration_seconds: int = 0
    status: RequestStatus = RequestStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    bars_count: int = 0


class HistoricalDataService:
    """Historical data service with request queueing and pacing controls."""
    
    # TWS Historical Data Limits (max duration per bar size)
    BAR_SIZE_LIMITS = {
        '5 secs': '7200 S',   # 2 hours
        '10 secs': '14400 S',  # 4 hours
        '15 secs': '14400 S',  # 4 hours
        '30 secs': '28800 S',  # 8 hours
        '1 min': '1 W',        # 1 week
        '2 mins': '2 W',       # 2 weeks
        '5 mins': '1 M',       # 1 month
        '15 mins': '1 M',      # 1 month
        '30 mins': '1 M',      # 1 month
        '1 hour': '1 Y',       # 1 year
        '1 day': '1 Y',        # 1 year (though more is available)
    }
    
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
    
    def _duration_to_seconds(self, duration_str: str) -> int:
        """Convert TWS duration string to seconds."""
        duration_str = duration_str.strip()
        if duration_str.endswith('S'):
            return int(duration_str[:-2])
        elif duration_str.endswith('D'):
            return int(duration_str[:-2]) * 86400
        elif duration_str.endswith('W'):
            return int(duration_str[:-2]) * 604800
        elif duration_str.endswith('M'):
            return int(duration_str[:-2]) * 2592000  # 30 days
        elif duration_str.endswith('Y'):
            return int(duration_str[:-2]) * 31536000  # 365 days
        return 0
    
    def _seconds_to_duration_str(self, seconds: int) -> str:
        """Convert seconds to a TWS duration string (defaults to seconds granularity)."""
        if seconds <= 0:
            return "1 S"
        # Prefer larger units when the division is clean, otherwise fall back to seconds
        if seconds % 31536000 == 0:
            years = seconds // 31536000
            return f"{years} Y"
        if seconds % 2592000 == 0:
            months = seconds // 2592000
            return f"{months} M"
        if seconds % 604800 == 0:
            weeks = seconds // 604800
            return f"{weeks} W"
        if seconds % 86400 == 0:
            days = seconds // 86400
            return f"{days} D"
        return f"{seconds} S"
    
    def _parse_end_datetime(self, end_datetime: Optional[str]) -> datetime:
        """Parse API end datetime formats into a timezone-aware datetime (UTC)."""
        if not end_datetime:
            return datetime.now(timezone.utc).replace(microsecond=0)
        
        patterns = [
            "%Y%m%d %H:%M:%S",
            "%Y-%m-%d-%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
        ]
        
        for pattern in patterns:
            try:
                parsed = datetime.strptime(end_datetime, pattern)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                else:
                    parsed = parsed.astimezone(timezone.utc)
                return parsed.replace(microsecond=0)
            except ValueError:
                continue
        
        # Fallback: attempt ISO parsing
        try:
            parsed_iso = datetime.fromisoformat(end_datetime)
            if parsed_iso.tzinfo is None:
                parsed_iso = parsed_iso.replace(tzinfo=timezone.utc)
            else:
                parsed_iso = parsed_iso.astimezone(timezone.utc)
            return parsed_iso.replace(microsecond=0)
        except ValueError:
            self.logger.warning(f"Unrecognized end_datetime format '{end_datetime}', defaulting to now()")
            return datetime.now(timezone.utc).replace(microsecond=0)
    
    def _format_end_datetime(self, dt: Optional[datetime]) -> str:
        """Format datetime for TWS historical data request."""
        if not dt:
            return ""
        return dt.strftime("%Y%m%d %H:%M:%S")
    
    def _plan_chunk_windows(
        self,
        symbol: str,
        bar_size: str,
        duration: str,
        end_datetime: Optional[str],
        base_id: str
    ) -> Tuple[List[dict], datetime]:
        """Plan chunk windows for a request and return chunk metadata."""
        max_duration_str = self.BAR_SIZE_LIMITS.get(bar_size, duration)
        max_duration_seconds = self._duration_to_seconds(max_duration_str)
        requested_duration_seconds = self._duration_to_seconds(duration)
        end_time = self._parse_end_datetime(end_datetime)
        
        if requested_duration_seconds == 0:
            requested_duration_seconds = max_duration_seconds
        
        if requested_duration_seconds <= 0:
            requested_duration_seconds = max_duration_seconds
        
        num_chunks = max(
            1,
            (requested_duration_seconds + max_duration_seconds - 1) // max_duration_seconds
        )
        
        remaining_seconds = requested_duration_seconds
        current_end = end_time
        chunk_plans: List[dict] = []
        
        for index in range(1, num_chunks + 1):
            chunk_seconds = min(max_duration_seconds, remaining_seconds)
            chunk_start = current_end - timedelta(seconds=chunk_seconds)
            chunk_duration_str = self._seconds_to_duration_str(chunk_seconds)
            
            chunk_id = f"{base_id}_chunk{index}of{num_chunks}" if num_chunks > 1 else base_id
            
            chunk_plans.append({
                "chunk_index": index,
                "chunk_total": num_chunks,
                "request_id": chunk_id,
                "duration_str": chunk_duration_str,
                "start": chunk_start,
                "end": current_end,
                "end_str": self._format_end_datetime(current_end),
            })
            
            current_end = chunk_start
            remaining_seconds -= chunk_seconds
        
        return chunk_plans, end_time

    def _generate_request_id(self, prefix: str, symbol: str, bar_size: str) -> str:
        """Generate a unique base identifier for historical jobs and chunks."""
        timestamp_ms = int(time.time() * 1000)
        unique_suffix = uuid.uuid4().hex[:8]
        sanitized_bar_size = bar_size.replace(" ", "")
        return f"{prefix}_{symbol}_{sanitized_bar_size}_{timestamp_ms}_{unique_suffix}"
    
    def _ensure_symbol_record(self, symbol: str):
        """Ensure a symbol exists in the database (create placeholder if missing)."""
        def _ensure(session):
            existing = session.query(Symbol).filter(Symbol.symbol == symbol).first()
            if existing:
                return existing
            
            new_symbol = Symbol(
                symbol=symbol,
                currency='USD',
                active=True,
                updated_at=datetime.now(timezone.utc)
            )
            session.add(new_symbol)
            session.flush()
            return new_symbol
        
        return execute_with_retry(_ensure)
    
    async def _create_job_with_chunks(
        self,
        symbol: str,
        bar_size: str,
        what_to_show: str,
        duration: str,
        end_datetime: Optional[str],
        use_rth: bool,
        base_id: str
    ) -> Tuple[HistoricalJob, List[HistoricalRequest]]:
        """Create a persisted job and associated chunk records, returning queue-ready requests."""
        chunk_plans, planned_end = self._plan_chunk_windows(
            symbol=symbol,
            bar_size=bar_size,
            duration=duration,
            end_datetime=end_datetime,
            base_id=base_id
        )
        
        now = datetime.now(timezone.utc)
        loop = asyncio.get_event_loop()
        
        # Ensure symbol exists (placeholder if necessary) before creating job records
        await loop.run_in_executor(None, lambda: self._ensure_symbol_record(symbol))
        
        def _create(session):
            job = HistoricalJob(
                job_key=base_id,
                symbol=symbol,
                bar_size=bar_size,
                what_to_show=what_to_show,
                use_rth=use_rth,
                duration=duration,
                end_datetime=planned_end,
                status='pending',
                total_chunks=len(chunk_plans),
                created_at=now,
                updated_at=now,
            )
            session.add(job)
            session.flush()
            
            requests: List[HistoricalRequest] = []
            
            for plan in chunk_plans:
                chunk = HistoricalJobChunk(
                    job_id=job.id,
                    chunk_index=plan["chunk_index"],
                    request_id=plan["request_id"],
                    status='pending',
                    duration=plan["duration_str"],
                    start_datetime=plan["start"],
                    end_datetime=plan["end"],
                    scheduled_for=now,
                    priority=0,
                    attempts=0,
                    max_attempts=5,
                )
                session.add(chunk)
                session.flush()
                
                requests.append(
                    HistoricalRequest(
                        id=chunk.request_id,
                        symbol=symbol,
                        bar_size=bar_size,
                        what_to_show=what_to_show,
                        duration=chunk.duration,
                        end_datetime=plan["end_str"],
                        use_rth=use_rth,
                        job_id=job.id,
                        chunk_id=chunk.id,
                        chunk_index=chunk.chunk_index,
                        start_dt=chunk.start_datetime,
                        end_dt=chunk.end_datetime,
                        duration_seconds=self._duration_to_seconds(chunk.duration)
                    )
                )
            
            return job, requests
        
        job, requests = await loop.run_in_executor(
            None,
            lambda: execute_with_retry(_create)
        )
        
        for request in requests:
            await self.request_queue.put(request)
        
        return job, requests
    
    async def _load_pending_chunks_from_db(self) -> int:
        """Load pending or retryable chunks from the database into the in-memory queue."""
        loop = asyncio.get_event_loop()
        now = datetime.now(timezone.utc)
        
        def _load(session):
            pending_requests: List[HistoricalRequest] = []
            
            query = (
                session.query(HistoricalJobChunk, HistoricalJob)
                .join(HistoricalJob, HistoricalJobChunk.job_id == HistoricalJob.id)
                .filter(
                    HistoricalJobChunk.status.in_(["pending", "in_progress"]),
                    HistoricalJobChunk.scheduled_for <= now
                )
                .order_by(
                    HistoricalJobChunk.priority.desc(),
                    HistoricalJobChunk.scheduled_for.asc(),
                    HistoricalJobChunk.id.asc()
                )
            )
            
            for chunk, job in query.all():
                if chunk.status == "in_progress":
                    # Reset any dangling in-progress work so we can resume cleanly
                    chunk.status = "pending"
                    chunk.started_at = None
                    chunk.updated_at = now
                
                pending_requests.append(
                    HistoricalRequest(
                        id=chunk.request_id,
                        symbol=job.symbol,
                        bar_size=job.bar_size,
                        what_to_show=job.what_to_show,
                        duration=chunk.duration,
                        end_datetime=self._format_end_datetime(chunk.end_datetime),
                        use_rth=job.use_rth,
                        job_id=job.id,
                        chunk_id=chunk.id,
                        chunk_index=chunk.chunk_index,
                        start_dt=chunk.start_datetime,
                        end_dt=chunk.end_datetime,
                        duration_seconds=self._duration_to_seconds(chunk.duration)
                    )
                )
            
            return pending_requests
        
        pending_requests = await loop.run_in_executor(
            None,
            lambda: execute_with_retry(_load)
        )
        
        for request in pending_requests:
            await self.request_queue.put(request)
        
        if pending_requests:
            self.logger.info(f"Recovered {len(pending_requests)} queued historical chunk(s) from database")
        
        return len(pending_requests)
    
    def _should_skip_chunk(self, session, request: HistoricalRequest) -> bool:
        """Determine whether a chunk should be skipped based on existing coverage."""
        if not request.start_dt or not request.end_dt:
            return False
        
        coverage = session.query(HistoricalCoverage).filter(
            HistoricalCoverage.symbol == request.symbol,
            HistoricalCoverage.timeframe == request.bar_size
        ).first()
        
        if not coverage:
            return False
        
        if coverage.min_ts and coverage.max_ts:
            if coverage.min_ts <= request.start_dt and coverage.max_ts >= request.end_dt:
                # Double-check that data actually exists near the edges of the window
                start_bar = session.query(Candle.ts).filter(
                    Candle.symbol == request.symbol,
                    Candle.tf == request.bar_size,
                    Candle.ts >= request.start_dt
                ).order_by(Candle.ts.asc()).first()
                
                end_bar = session.query(Candle.ts).filter(
                    Candle.symbol == request.symbol,
                    Candle.tf == request.bar_size,
                    Candle.ts <= request.end_dt
                ).order_by(Candle.ts.desc()).first()
                
                if start_bar and end_bar:
                    return True
        
        return False
    
    async def _mark_chunk_started(self, request: HistoricalRequest) -> Tuple[bool, Optional[str]]:
        """Mark a chunk as in-progress; return (skipped, reason)."""
        loop = asyncio.get_event_loop()
        now = datetime.now(timezone.utc)
        
        def _mark(session):
            chunk = session.query(HistoricalJobChunk).filter(
                HistoricalJobChunk.id == request.chunk_id
            ).first()
            if not chunk:
                return True, "chunk_not_found"
            
            job = session.query(HistoricalJob).filter(
                HistoricalJob.id == chunk.job_id
            ).first()
            if not job:
                return True, "job_not_found"
            
            # Refresh request metadata with the latest persisted values
            request.duration = chunk.duration
            request.start_dt = chunk.start_datetime
            request.end_dt = chunk.end_datetime
            request.end_datetime = self._format_end_datetime(chunk.end_datetime)
            request.duration_seconds = self._duration_to_seconds(chunk.duration)
            
            if chunk.status == "completed":
                return True, "already_completed"
            
            if chunk.status == "skipped":
                return True, "already_skipped"
            
            if self._should_skip_chunk(session, request):
                chunk.status = "skipped"
                chunk.completed_at = now
                chunk.updated_at = now
                chunk.error_message = None
                job.completed_chunks = (job.completed_chunks or 0) + 1
                if job.status in ("pending", "running"):
                    if job.completed_chunks >= job.total_chunks:
                        job.status = "completed"
                        job.completed_at = now
                    else:
                        job.status = "running"
                session.flush()
                return True, "coverage_satisfied"
            
            chunk.status = "in_progress"
            chunk.started_at = now
            chunk.updated_at = now
            chunk.attempts = (chunk.attempts or 0) + 1
            chunk.error_message = None
            
            if job.status == "pending":
                job.status = "running"
                job.started_at = job.started_at or now
            job.updated_at = now
            
            session.flush()
            return False, None
        
        return await loop.run_in_executor(
            None,
            lambda: execute_with_retry(_mark)
        )
    
    async def _mark_chunk_completed(
        self,
        request: HistoricalRequest,
        bars: List[BarData],
        bars_stored: int
    ) -> None:
        """Mark chunk completion, update job counters and coverage metadata."""
        loop = asyncio.get_event_loop()
        now = datetime.now(timezone.utc)
        
        # Derive time bounds from returned bars (fallback to planned window if empty)
        bar_timestamps = [
            bar.date if isinstance(bar.date, datetime) else None
            for bar in bars
            if hasattr(bar, 'date')
        ]
        bar_timestamps = [ts.replace(tzinfo=timezone.utc) if ts and ts.tzinfo is None else ts for ts in bar_timestamps]
        bar_timestamps = [ts for ts in bar_timestamps if ts]
        
        chunk_min = min(bar_timestamps) if bar_timestamps else request.start_dt
        chunk_max = max(bar_timestamps) if bar_timestamps else request.end_dt
        
        def _complete(session):
            chunk = session.query(HistoricalJobChunk).filter(
                HistoricalJobChunk.id == request.chunk_id
            ).first()
            if not chunk:
                return
            
            job = session.query(HistoricalJob).filter(
                HistoricalJob.id == chunk.job_id
            ).first()
            if not job:
                return
            
            chunk.status = "completed"
            chunk.completed_at = now
            chunk.updated_at = now
            chunk.bars_received = (chunk.bars_received or 0) + max(bars_stored, 0)
            chunk.error_message = None
            
            job.completed_chunks = (job.completed_chunks or 0) + 1
            job.failed_chunks = job.failed_chunks or 0
            job.updated_at = now
            if job.status in ("pending", "running"):
                if job.completed_chunks >= job.total_chunks:
                    job.status = "completed"
                    job.completed_at = now
                else:
                    job.status = "running"
            
            coverage = session.query(HistoricalCoverage).filter(
                HistoricalCoverage.symbol == request.symbol,
                HistoricalCoverage.timeframe == request.bar_size
            ).first()
            
            if not coverage:
                coverage = HistoricalCoverage(
                    symbol=request.symbol,
                    timeframe=request.bar_size,
                    min_ts=chunk_min,
                    max_ts=chunk_max,
                    total_bars=bars_stored,
                    last_updated_at=now,
                    last_verified_at=now
                )
                session.add(coverage)
            else:
                if chunk_min:
                    coverage.min_ts = (
                        chunk_min if not coverage.min_ts else min(coverage.min_ts, chunk_min)
                    )
                if chunk_max:
                    coverage.max_ts = (
                        chunk_max if not coverage.max_ts else max(coverage.max_ts, chunk_max)
                    )
                coverage.total_bars = (coverage.total_bars or 0) + max(bars_stored, 0)
                coverage.last_updated_at = now
                coverage.last_verified_at = now
            
            session.flush()
        
        await loop.run_in_executor(None, lambda: execute_with_retry(_complete))
    
    async def _handle_chunk_failure(self, request: HistoricalRequest, error_message: str) -> bool:
        """Record failure information and decide whether to retry the chunk."""
        loop = asyncio.get_event_loop()
        now = datetime.now(timezone.utc)
        
        def _fail(session):
            chunk = session.query(HistoricalJobChunk).filter(
                HistoricalJobChunk.id == request.chunk_id
            ).first()
            if not chunk:
                return False
            
            job = session.query(HistoricalJob).filter(
                HistoricalJob.id == chunk.job_id
            ).first()
            if not job:
                return False
            
            max_attempts = chunk.max_attempts or 5
            attempts = chunk.attempts or 0
            chunk.error_message = error_message[:1024]
            chunk.updated_at = now
            
            if attempts >= max_attempts:
                chunk.status = "failed"
                chunk.completed_at = now
                job.failed_chunks = (job.failed_chunks or 0) + 1
                job.status = "failed"
                job.completed_at = job.completed_at or now
                job.updated_at = now
                session.flush()
                return False
            
            # Schedule retry with basic backoff (attempt count already incremented in _mark_chunk_started)
            backoff_seconds = min(60 * attempts, 300)
            chunk.status = "pending"
            chunk.started_at = None
            chunk.scheduled_for = now + timedelta(seconds=backoff_seconds)
            job.status = "running"
            job.updated_at = now
            
            session.flush()
            return True
        
        return await loop.run_in_executor(None, lambda: execute_with_retry(_fail))
    
    def _split_request_if_needed(self, symbol: str, bar_size: str, duration: str, 
                                   what_to_show: str, use_rth: bool, base_id: str,
                                   end_datetime: Optional[str] = None) -> List[dict]:
        """Split a request into multiple chunks if it exceeds TWS limits for the bar size."""
        chunk_plans, _ = self._plan_chunk_windows(
            symbol=symbol,
            bar_size=bar_size,
            duration=duration,
            end_datetime=end_datetime,
            base_id=base_id
        )
        
        if len(chunk_plans) > 1:
            self.logger.info(
                f"Split request for {symbol} ({bar_size}, {duration}) into {len(chunk_plans)} chunks"
            )
        return chunk_plans
    
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
            def _job_summary(session):
                return {
                    "pending_chunks": session.query(func.count(HistoricalJobChunk.id)).filter(HistoricalJobChunk.status == "pending").scalar(),
                    "in_progress_chunks": session.query(func.count(HistoricalJobChunk.id)).filter(HistoricalJobChunk.status == "in_progress").scalar(),
                    "failed_chunks": session.query(func.count(HistoricalJobChunk.id)).filter(HistoricalJobChunk.status == "failed").scalar(),
                    "skipped_chunks": session.query(func.count(HistoricalJobChunk.id)).filter(HistoricalJobChunk.status == "skipped").scalar(),
                    "total_jobs": session.query(func.count(HistoricalJob.id)).scalar(),
                    "completed_jobs": session.query(func.count(HistoricalJob.id)).filter(HistoricalJob.status == "completed").scalar(),
                    "failed_jobs": session.query(func.count(HistoricalJob.id)).filter(HistoricalJob.status == "failed").scalar(),
                }
            
            summary = execute_with_retry(_job_summary)
            
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
                ],
                "db_summary": summary
            }
        
        @self.app.post("/historical/request")
        async def request_historical_data(request: dict):
            """Request historical data for a symbol."""
            try:
                # Extract parameters from request dict
                symbol = request.get("symbol")
                bar_size = request.get("bar_size", "1 min")
                what_to_show = request.get("what_to_show", "TRADES")
                duration = request.get("duration", "1 D")
                end_datetime = request.get("end_datetime", "")
                use_rth = request.get("use_rth", True)
                
                if not symbol:
                    raise HTTPException(status_code=400, detail="symbol is required")
                
                request_id = self._generate_request_id("req", symbol, bar_size)
                
                # Validate bar size
                valid_bar_sizes = self.settings.historical.bar_sizes_list
                if bar_size not in valid_bar_sizes:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid bar_size. Must be one of: {valid_bar_sizes}"
                    )

                # Persist job and enqueue all chunks
                job, queued_requests = await self._create_job_with_chunks(
                    symbol=symbol,
                    bar_size=bar_size,
                    what_to_show=what_to_show,
                    duration=duration,
                    end_datetime=end_datetime,
                    use_rth=use_rth,
                    base_id=request_id
                )
                chunk_count = len(queued_requests)
                
                self.logger.info(
                    f"Queued historical job {job.job_key} with {chunk_count} chunk(s) for {symbol}"
                )
                
                return {
                    "request_id": request_id,
                    "status": "queued",
                    "chunks": chunk_count,
                    "job_id": job.id,
                    "message": f"Queued {chunk_count} chunk(s) for {symbol}",
                    "queue_position": self.request_queue.qsize()
                }
                
            except Exception as e:
                self.logger.error(f"Failed to queue historical data request: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/historical/bulk")
        async def bulk_historical_request(body: Optional[BulkHistoricalRequestBody] = None):
            """Request historical data for all watchlist symbols."""
            try:
                # Get watchlist symbols
                def _get_watchlist(session):
                    entries = session.query(WatchlistEntry).all()
                    return [entry.symbol for entry in entries]
                
                symbols = execute_with_retry(_get_watchlist)
                
                if not symbols:
                    return {"message": "No symbols in watchlist", "requests": []}
                
                # Determine bar sizes and duration to use
                if body and body.bar_size:
                    bar_sizes = [body.bar_size]
                else:
                    bar_sizes = self.settings.historical.bar_sizes_list
                
                duration = body.duration if body and body.duration else self.settings.market_data.lookback
                
                # Create requests for each symbol and bar size (with splitting if needed)
                request_ids = []
                job_ids = []
                total_chunks = 0
                
                for symbol in symbols:
                    for bar_size in bar_sizes:
                        request_id = self._generate_request_id("bulk", symbol, bar_size)
                        
                        job, queued_requests = await self._create_job_with_chunks(
                            symbol=symbol,
                            bar_size=bar_size,
                            duration=duration,
                            what_to_show=self.settings.market_data.what_to_show,
                            use_rth=self.settings.market_data.rth,
                            base_id=request_id,
                            end_datetime=None
                        )
                        
                        job_ids.append(job.id)
                        request_ids.append(request_id)
                        total_chunks += len(queued_requests)
                
                self.logger.info(f"Queued {total_chunks} bulk historical chunks ({len(request_ids)} base requests)")
                
                return {
                    "message": f"Queued {total_chunks} historical data chunks ({len(request_ids)} symbols × bar sizes)",
                    "symbols": symbols,
                    "bar_sizes": bar_sizes,
                    "duration": duration,
                    "total_chunks": total_chunks,
                    "requests": request_ids,
                    "jobs": job_ids
                }
                
            except Exception as e:
                self.logger.error(f"Failed to queue bulk historical requests: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/historical/bulk/upload")
        async def bulk_historical_upload(body: CustomBulkHistoricalRequestBody):
            """Request historical data for a provided list of symbols."""
            try:
                symbols = [s.strip().upper() for s in body.symbols if s and s.strip()]
                if not symbols:
                    raise HTTPException(status_code=400, detail="symbols list cannot be empty")

                # Deduplicate while preserving order
                seen = set()
                unique_symbols: List[str] = []
                for symbol in symbols:
                    if symbol not in seen:
                        seen.add(symbol)
                        unique_symbols.append(symbol)

                bar_size = body.bar_size if body and body.bar_size else self.settings.historical.bar_sizes_list[0]
                duration = body.duration if body and body.duration else self.settings.market_data.lookback
                end_datetime = body.end_datetime if body else None
                use_rth = body.use_rth if body and body.use_rth is not None else self.settings.market_data.rth
                what_to_show = body.what_to_show if body and body.what_to_show else self.settings.market_data.what_to_show

                # Validate bar size
                valid_bar_sizes = self.settings.historical.bar_sizes_list
                if bar_size not in valid_bar_sizes:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid bar_size. Must be one of: {valid_bar_sizes}"
                    )

                request_ids = []
                job_ids = []
                total_chunks = 0

                for symbol in unique_symbols:
                    request_id = self._generate_request_id("upload", symbol, bar_size)
                    job, queued_requests = await self._create_job_with_chunks(
                        symbol=symbol,
                        bar_size=bar_size,
                        what_to_show=what_to_show,
                        duration=duration,
                        end_datetime=end_datetime,
                        use_rth=use_rth,
                        base_id=request_id
                    )
                    request_ids.append(request_id)
                    job_ids.append(job.id)
                    total_chunks += len(queued_requests)

                self.logger.info(
                    f"Queued {total_chunks} uploaded-symbol historical chunks ({len(request_ids)} symbols)"
                )

                return {
                    "message": f"Queued {total_chunks} historical data chunks for {len(request_ids)} uploaded symbols",
                    "symbols": unique_symbols,
                    "bar_size": bar_size,
                    "duration": duration,
                    "total_chunks": total_chunks,
                    "requests": request_ids,
                    "jobs": job_ids
                }

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to queue uploaded historical requests: {e}")
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
            
            # Fallback to persisted chunk info
            def _lookup(session):
                result = (
                    session.query(HistoricalJobChunk, HistoricalJob)
                    .join(HistoricalJob, HistoricalJobChunk.job_id == HistoricalJob.id)
                    .filter(HistoricalJobChunk.request_id == request_id)
                    .first()
                )
                if not result:
                    return None
                chunk, job = result
                return {
                    "request_id": request_id,
                    "status": chunk.status,
                    "symbol": job.symbol,
                    "bar_size": job.bar_size,
                    "started_at": chunk.started_at.isoformat() if chunk.started_at else None,
                    "completed_at": chunk.completed_at.isoformat() if chunk.completed_at else None,
                    "bars_count": chunk.bars_received or 0,
                    "error": chunk.error_message,
                    "job_id": job.id,
                }
            
            persisted = execute_with_retry(_lookup)
            if persisted:
                return persisted
            
            raise HTTPException(status_code=404, detail="Request not found")
        
        @self.app.get("/historical/jobs")
        async def list_historical_jobs(limit: int = 50):
            """List recent historical jobs with summary information."""
            def _list(session):
                jobs = (
                    session.query(HistoricalJob)
                    .order_by(HistoricalJob.created_at.desc())
                    .limit(limit)
                    .all()
                )
                return [
                    {
                        "id": job.id,
                        "job_key": job.job_key,
                        "symbol": job.symbol,
                        "bar_size": job.bar_size,
                        "status": job.status,
                        "total_chunks": job.total_chunks,
                        "completed_chunks": job.completed_chunks,
                        "failed_chunks": job.failed_chunks,
                        "created_at": job.created_at.isoformat() if job.created_at else None,
                        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
                        "started_at": job.started_at.isoformat() if job.started_at else None,
                        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    }
                    for job in jobs
                ]
            
            jobs = execute_with_retry(_list)
            return {"jobs": jobs, "count": len(jobs)}
        
        @self.app.get("/historical/jobs/{job_id}")
        async def get_historical_job(job_id: int):
            """Retrieve job details including chunk breakdown."""
            def _get(session):
                job = session.query(HistoricalJob).filter(HistoricalJob.id == job_id).first()
                if not job:
                    return None
                
                chunks = (
                    session.query(HistoricalJobChunk)
                    .filter(HistoricalJobChunk.job_id == job.id)
                    .order_by(HistoricalJobChunk.chunk_index.asc())
                    .all()
                )
                
                return {
                    "id": job.id,
                    "job_key": job.job_key,
                    "symbol": job.symbol,
                    "bar_size": job.bar_size,
                    "status": job.status,
                    "duration": job.duration,
                    "end_datetime": job.end_datetime.isoformat() if job.end_datetime else None,
                    "total_chunks": job.total_chunks,
                    "completed_chunks": job.completed_chunks,
                    "failed_chunks": job.failed_chunks,
                    "created_at": job.created_at.isoformat() if job.created_at else None,
                    "updated_at": job.updated_at.isoformat() if job.updated_at else None,
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "chunks": [
                        {
                            "id": chunk.id,
                            "request_id": chunk.request_id,
                            "chunk_index": chunk.chunk_index,
                            "status": chunk.status,
                            "duration": chunk.duration,
                            "start_datetime": chunk.start_datetime.isoformat() if chunk.start_datetime else None,
                            "end_datetime": chunk.end_datetime.isoformat() if chunk.end_datetime else None,
                            "scheduled_for": chunk.scheduled_for.isoformat() if chunk.scheduled_for else None,
                            "attempts": chunk.attempts,
                            "max_attempts": chunk.max_attempts,
                            "bars_received": chunk.bars_received,
                            "error_message": chunk.error_message,
                            "completed_at": chunk.completed_at.isoformat() if chunk.completed_at else None,
                        }
                        for chunk in chunks
                    ]
                }
            
            job = execute_with_retry(_get)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            return job
    
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
            
            # Recover any pending work from previous runs
            recovered = await self._load_pending_chunks_from_db()
            if recovered:
                self.logger.info(f"Recovered {recovered} pending chunk(s) before starting processing loop")
            
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
            
            skipped, reason = await self._mark_chunk_started(request)
            if skipped:
                request.status = RequestStatus.COMPLETED if reason in ("already_completed", "already_skipped") else RequestStatus.SKIPPED
                request.completed_at = datetime.now(timezone.utc)
                request.bars_count = 0
                if reason == "coverage_satisfied":
                    request.error_message = None
                else:
                    request.error_message = reason
                self.completed_requests[request.id] = request
                if request.id in self.active_requests:
                    del self.active_requests[request.id]
                self.logger.info(f"Skipped historical request {request.id} ({reason})")
                return
            
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
            await self._mark_chunk_completed(request, bars, bars_stored)
            
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
            
            should_retry = await self._handle_chunk_failure(request, str(e))
            request.error_message = str(e)
            request.completed_at = datetime.now(timezone.utc)
            
            if request.id in self.active_requests:
                del self.active_requests[request.id]
            
            if should_retry:
                request.status = RequestStatus.PENDING
                request.started_at = None
                await self.request_queue.put(request)
                self.logger.info(f"Re-queued historical request {request.id} for retry")
            else:
                request.status = RequestStatus.FAILED
                self.completed_requests[request.id] = request
    
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
    from src.common.debug import enable_remote_debugging
    enable_remote_debugging(5681)
    asyncio.run(main())
