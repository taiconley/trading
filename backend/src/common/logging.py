"""
Structured logging configuration for the trading bot.

This module provides JSON-structured logging with Docker-friendly output,
optional database log sampling, and service-specific logger creation.
"""

import json
import logging
import logging.handlers
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager

from .config import get_settings
from .db import execute_with_retry


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def __init__(self, service_name: str = "unknown"):
        super().__init__()
        self.service_name = service_name
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "service": self.service_name,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields if present
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                'thread', 'threadName', 'processName', 'process', 'message', 'exc_info',
                'exc_text', 'stack_info', 'getMessage'
            }:
                extra_fields[key] = value
        
        if extra_fields:
            log_entry["extra"] = extra_fields
        
        return json.dumps(log_entry, default=str, separators=(',', ':'))


class DatabaseLogHandler(logging.Handler):
    """Log handler that optionally samples logs to database."""
    
    def __init__(self, service_name: str, sample_rate: float = 0.1):
        super().__init__()
        self.service_name = service_name
        self.sample_rate = sample_rate  # Fraction of logs to store in DB
        self._last_sample_time = 0
    
    def emit(self, record: logging.LogRecord):
        """Emit log record to database with sampling."""
        try:
            # Sample logs based on rate and time
            current_time = time.time()
            should_sample = (
                record.levelno >= logging.WARNING or  # Always store warnings and errors
                (current_time - self._last_sample_time) > (1.0 / self.sample_rate) or
                hash(record.getMessage()) % 100 < (self.sample_rate * 100)
            )
            
            if not should_sample:
                return
            
            self._last_sample_time = current_time
            
            # Prepare log entry for database
            meta_json = {}
            if hasattr(record, 'extra'):
                meta_json.update(record.extra)
            
            if record.exc_info:
                meta_json['exception'] = {
                    "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                    "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                    "traceback": self.format(record)
                }
            
            # Store in database
            def _store_log(session):
                from .models import LogEntry
                log_entry = LogEntry(
                    service=self.service_name,
                    level=record.levelname,
                    msg=record.getMessage(),
                    meta_json=meta_json if meta_json else None
                )
                session.add(log_entry)
                session.commit()
            
            execute_with_retry(_store_log, max_retries=1)
            
        except Exception:
            # Don't let logging errors break the application
            self.handleError(record)


def setup_logging(
    service_name: str,
    log_level: Optional[str] = None,
    enable_db_logging: bool = False,
    db_sample_rate: float = 0.1
) -> logging.Logger:
    """
    Set up structured logging for a service.
    
    Args:
        service_name: Name of the service
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_db_logging: Whether to enable database log sampling
        db_sample_rate: Fraction of logs to sample to database
        
    Returns:
        Configured logger instance
    """
    settings = get_settings()
    
    # Determine log level
    if log_level is None:
        log_level = settings.logging.level
    
    # Get root logger for this service
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler (JSON format for Docker)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(JSONFormatter(service_name))
    logger.addHandler(console_handler)
    
    # Database handler (optional)
    if enable_db_logging:
        try:
            db_handler = DatabaseLogHandler(service_name, db_sample_rate)
            db_handler.setLevel(logging.INFO)  # Only INFO and above to database
            logger.addHandler(db_handler)
        except Exception as e:
            # Log the error but don't fail startup
            logger.warning(f"Failed to setup database logging: {e}")
    
    # Set up module-level loggers
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.WARNING)
    
    logger.info(f"Logging initialized for service: {service_name}")
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the current service configuration.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


@contextmanager
def log_execution_time(logger: logging.Logger, operation: str, level: int = logging.INFO):
    """
    Context manager to log execution time of operations.
    
    Args:
        logger: Logger instance
        operation: Description of the operation
        level: Log level
    """
    start_time = time.time()
    try:
        yield
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        logger.log(level, f"{operation} completed", extra={
            "operation": operation,
            "execution_time_ms": round(execution_time, 2)
        })
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"{operation} failed", extra={
            "operation": operation,
            "execution_time_ms": round(execution_time, 2),
            "error": str(e)
        })
        raise


def log_trade_event(
    logger: logging.Logger,
    event_type: str,
    symbol: str,
    **kwargs
):
    """
    Log trading-specific events with structured data.
    
    Args:
        logger: Logger instance
        event_type: Type of trading event (order_placed, execution, signal, etc.)
        symbol: Trading symbol
        **kwargs: Additional event data
    """
    logger.info(f"Trading event: {event_type}", extra={
        "event_type": event_type,
        "symbol": symbol,
        "trading_data": kwargs
    })


def log_market_data_event(
    logger: logging.Logger,
    event_type: str,
    symbol: str,
    **kwargs
):
    """
    Log market data events with structured data.
    
    Args:
        logger: Logger instance
        event_type: Type of market data event (tick, bar, subscription, etc.)
        symbol: Trading symbol
        **kwargs: Additional event data
    """
    logger.debug(f"Market data event: {event_type}", extra={
        "event_type": event_type,
        "symbol": symbol,
        "market_data": kwargs
    })


def log_system_event(
    logger: logging.Logger,
    event_type: str,
    **kwargs
):
    """
    Log system events with structured data.
    
    Args:
        logger: Logger instance
        event_type: Type of system event (startup, shutdown, health_check, etc.)
        **kwargs: Additional event data
    """
    logger.info(f"System event: {event_type}", extra={
        "event_type": event_type,
        "system_data": kwargs
    })


def configure_service_logging(service_name: str) -> logging.Logger:
    """
    Configure logging for a service with default settings.
    
    Args:
        service_name: Name of the service
        
    Returns:
        Configured logger
    """
    settings = get_settings()
    
    # Enable database logging for production services
    enable_db = not settings.development.debug
    
    return setup_logging(
        service_name=service_name,
        enable_db_logging=enable_db,
        db_sample_rate=0.05 if enable_db else 0.0  # 5% sampling in production
    )


# Utility functions for common logging patterns

def log_function_entry(logger: logging.Logger, func_name: str, **kwargs):
    """Log function entry with parameters."""
    logger.debug(f"Entering {func_name}", extra={"function": func_name, "params": kwargs})


def log_function_exit(logger: logging.Logger, func_name: str, result: Any = None):
    """Log function exit with optional result."""
    extra = {"function": func_name}
    if result is not None:
        extra["result"] = str(result)
    logger.debug(f"Exiting {func_name}", extra=extra)


def log_api_request(logger: logging.Logger, method: str, path: str, **kwargs):
    """Log API request details."""
    logger.info(f"API request: {method} {path}", extra={
        "api_method": method,
        "api_path": path,
        "request_data": kwargs
    })


def log_api_response(logger: logging.Logger, method: str, path: str, status_code: int, **kwargs):
    """Log API response details."""
    logger.info(f"API response: {method} {path} -> {status_code}", extra={
        "api_method": method,
        "api_path": path,
        "status_code": status_code,
        "response_data": kwargs
    })
