"""
Correlation ID Management for Request Tracing

Provides context management for tracking requests across multiple services
using unique correlation IDs. This enables distributed tracing and debugging
of multi-service operations.

Usage:
    # In FastAPI middleware or request handler
    with correlation_context(correlation_id="abc-123"):
        # All logs and operations will include this correlation ID
        logger.info("Processing order")  # Will include correlation_id in log
        
    # Or use decorator
    @with_correlation
    async def process_order(order_id: int):
        # Correlation ID automatically managed
        pass
"""

import uuid
import logging
from contextvars import ContextVar
from typing import Optional, Dict, Any
from functools import wraps
from datetime import datetime, timezone

# Context variable to store correlation ID
_correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
_request_start_time: ContextVar[Optional[float]] = ContextVar('request_start_time', default=None)


def generate_correlation_id() -> str:
    """Generate a new unique correlation ID"""
    return str(uuid.uuid4())


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID from context"""
    return _correlation_id.get()


def set_correlation_id(correlation_id: str):
    """Set the correlation ID in context"""
    _correlation_id.set(correlation_id)


def get_request_start_time() -> Optional[float]:
    """Get the request start time from context"""
    return _request_start_time.get()


def set_request_start_time(timestamp: float):
    """Set the request start time in context"""
    _request_start_time.set(timestamp)


def get_request_duration_ms() -> Optional[float]:
    """Get the duration of the current request in milliseconds"""
    start_time = get_request_start_time()
    if start_time is None:
        return None
    
    import time
    duration = (time.time() - start_time) * 1000
    return round(duration, 2)


class correlation_context:
    """
    Context manager for correlation ID.
    
    Usage:
        with correlation_context(correlation_id="abc-123"):
            # All operations will have this correlation ID
            logger.info("Processing request")
    """
    
    def __init__(self, correlation_id: Optional[str] = None, generate_if_missing: bool = True):
        """
        Initialize correlation context.
        
        Args:
            correlation_id: Correlation ID to use (or None to generate)
            generate_if_missing: If True, generate ID if not provided
        """
        if correlation_id is None and generate_if_missing:
            correlation_id = generate_correlation_id()
        
        self.correlation_id = correlation_id
        self.previous_id = None
        self.previous_start_time = None
    
    def __enter__(self):
        """Enter context and set correlation ID"""
        self.previous_id = get_correlation_id()
        self.previous_start_time = get_request_start_time()
        
        if self.correlation_id:
            set_correlation_id(self.correlation_id)
        
        import time
        set_request_start_time(time.time())
        
        return self.correlation_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore previous correlation ID"""
        # Restore previous values
        if self.previous_id:
            set_correlation_id(self.previous_id)
        else:
            _correlation_id.set(None)
        
        if self.previous_start_time:
            set_request_start_time(self.previous_start_time)
        else:
            _request_start_time.set(None)
        
        return False


def with_correlation(func):
    """
    Decorator to automatically manage correlation IDs for functions.
    
    Usage:
        @with_correlation
        async def my_function(arg1, arg2):
            # Correlation ID is automatically set
            logger.info("Processing...")
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Check if correlation_id is in kwargs
        correlation_id = kwargs.pop('correlation_id', None)
        
        with correlation_context(correlation_id):
            return await func(*args, **kwargs)
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        # Check if correlation_id is in kwargs
        correlation_id = kwargs.pop('correlation_id', None)
        
        with correlation_context(correlation_id):
            return func(*args, **kwargs)
    
    # Return appropriate wrapper based on function type
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


class CorrelationLogFilter(logging.Filter):
    """
    Log filter that adds correlation ID to log records.
    
    Usage:
        import logging
        logger = logging.getLogger(__name__)
        logger.addFilter(CorrelationLogFilter())
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to log record"""
        correlation_id = get_correlation_id()
        record.correlation_id = correlation_id or "none"
        
        # Add request duration if available
        duration = get_request_duration_ms()
        record.request_duration_ms = duration if duration is not None else 0.0
        
        return True


def get_correlation_context() -> Dict[str, Any]:
    """
    Get all correlation context information.
    
    Returns:
        Dictionary with correlation_id, request_start_time, and duration
    """
    correlation_id = get_correlation_id()
    start_time = get_request_start_time()
    duration_ms = get_request_duration_ms()
    
    context = {
        "correlation_id": correlation_id,
    }
    
    if start_time is not None:
        context["request_start_time"] = datetime.fromtimestamp(
            start_time, tz=timezone.utc
        ).isoformat()
        context["request_duration_ms"] = duration_ms
    
    return context


# FastAPI middleware helper
class CorrelationMiddleware:
    """
    FastAPI middleware to automatically add correlation IDs to requests.
    
    Usage in FastAPI app:
        from fastapi import FastAPI
        from common.correlation import CorrelationMiddleware
        
        app = FastAPI()
        app.add_middleware(CorrelationMiddleware)
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Extract correlation ID from headers or generate new one
        headers = dict(scope.get("headers", []))
        correlation_id = headers.get(b"x-correlation-id")
        
        if correlation_id:
            correlation_id = correlation_id.decode()
        else:
            correlation_id = generate_correlation_id()
        
        # Set correlation context
        with correlation_context(correlation_id):
            # Add correlation ID to response headers
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = message.get("headers", [])
                    headers.append((b"x-correlation-id", correlation_id.encode()))
                    message["headers"] = headers
                await send(message)
            
            await self.app(scope, receive, send_wrapper)


# HTTP client helper
def get_correlation_headers() -> Dict[str, str]:
    """
    Get HTTP headers with correlation ID for outgoing requests.
    
    Usage:
        import httpx
        headers = get_correlation_headers()
        response = httpx.get("http://api/endpoint", headers=headers)
    """
    correlation_id = get_correlation_id()
    
    if correlation_id:
        return {"X-Correlation-ID": correlation_id}
    
    return {}


def propagate_correlation_id(target_kwargs: Dict[str, Any]):
    """
    Add correlation ID to kwargs for function calls.
    
    Usage:
        kwargs = {"order_id": 123}
        propagate_correlation_id(kwargs)
        # Now kwargs = {"order_id": 123, "correlation_id": "abc-123"}
    """
    correlation_id = get_correlation_id()
    
    if correlation_id and 'correlation_id' not in target_kwargs:
        target_kwargs['correlation_id'] = correlation_id

