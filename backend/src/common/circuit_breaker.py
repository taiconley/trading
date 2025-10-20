"""
Circuit Breaker Pattern Implementation

Prevents cascading failures by temporarily blocking operations to failing services.
When a service is consistently failing, the circuit breaker "opens" and rejects
requests immediately rather than waiting for timeouts.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Failure threshold exceeded, requests are rejected immediately
- HALF_OPEN: Testing if service has recovered
"""

import time
import asyncio
from enum import Enum
from typing import Callable, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes in half-open before closing
    timeout: float = 60.0  # Seconds to wait before attempting recovery
    expected_exception: type = Exception  # Exception type to catch


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker"""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    last_state_change: Optional[float] = None
    total_failures: int = 0
    total_successes: int = 0
    total_rejected: int = 0


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation.
    
    Usage:
        cb = CircuitBreaker(
            failure_threshold=5,
            timeout=60,
            expected_exception=DatabaseError
        )
        
        async def risky_operation():
            return await cb.call(my_database_operation)
    """
    
    def __init__(
        self,
        name: str = "circuit_breaker",
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name for logging
            failure_threshold: Number of failures before opening circuit
            success_threshold: Number of successes in half-open before closing
            timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type that triggers circuit breaker
        """
        self.name = name
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
            expected_exception=expected_exception
        )
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
        
        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"failure_threshold={failure_threshold}, "
            f"timeout={timeout}s"
        )
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute (can be sync or async)
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Result of function call
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Original exception from function
        """
        async with self._lock:
            # Check if we should attempt recovery
            if self.stats.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
                else:
                    self.stats.total_rejected += 1
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Last failure: {self._format_time(self.stats.last_failure_time)}"
                    )
        
        # Execute the function
        try:
            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            await self._on_success()
            return result
            
        except self.config.expected_exception as e:
            await self._on_failure()
            raise
    
    async def _on_success(self):
        """Handle successful operation"""
        async with self._lock:
            self.stats.success_count += 1
            self.stats.total_successes += 1
            self.stats.last_success_time = time.time()
            self.stats.failure_count = 0  # Reset failure count on success
            
            if self.stats.state == CircuitState.HALF_OPEN:
                if self.stats.success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
    
    async def _on_failure(self):
        """Handle failed operation"""
        async with self._lock:
            self.stats.failure_count += 1
            self.stats.total_failures += 1
            self.stats.last_failure_time = time.time()
            self.stats.success_count = 0  # Reset success count on failure
            
            if self.stats.state == CircuitState.HALF_OPEN:
                # Immediate transition to open on any failure in half-open
                self._transition_to(CircuitState.OPEN)
            elif self.stats.failure_count >= self.config.failure_threshold:
                # Open circuit after threshold failures
                self._transition_to(CircuitState.OPEN)
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        if self.stats.last_failure_time is None:
            return True
        
        time_since_failure = time.time() - self.stats.last_failure_time
        return time_since_failure >= self.config.timeout
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state"""
        old_state = self.stats.state
        self.stats.state = new_state
        self.stats.last_state_change = time.time()
        
        if new_state == CircuitState.CLOSED:
            self.stats.failure_count = 0
            self.stats.success_count = 0
        
        logger.warning(
            f"Circuit breaker '{self.name}' state change: "
            f"{old_state.value} -> {new_state.value}"
        )
    
    def _format_time(self, timestamp: Optional[float]) -> str:
        """Format timestamp for display"""
        if timestamp is None:
            return "never"
        
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    async def reset(self):
        """Manually reset circuit breaker to closed state"""
        async with self._lock:
            logger.info(f"Manually resetting circuit breaker '{self.name}'")
            self._transition_to(CircuitState.CLOSED)
            self.stats.failure_count = 0
            self.stats.success_count = 0
    
    def get_state(self) -> dict:
        """Get current circuit breaker state and statistics"""
        return {
            "name": self.name,
            "state": self.stats.state.value,
            "failure_count": self.stats.failure_count,
            "success_count": self.stats.success_count,
            "total_failures": self.stats.total_failures,
            "total_successes": self.stats.total_successes,
            "total_rejected": self.stats.total_rejected,
            "last_failure": self._format_time(self.stats.last_failure_time),
            "last_success": self._format_time(self.stats.last_success_time),
            "last_state_change": self._format_time(self.stats.last_state_change),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
            }
        }


# Global circuit breakers for common services
_database_circuit_breaker: Optional[CircuitBreaker] = None
_tws_circuit_breaker: Optional[CircuitBreaker] = None


def get_database_circuit_breaker() -> CircuitBreaker:
    """Get or create global database circuit breaker"""
    global _database_circuit_breaker
    
    if _database_circuit_breaker is None:
        from sqlalchemy.exc import OperationalError, DisconnectionError
        
        _database_circuit_breaker = CircuitBreaker(
            name="database",
            failure_threshold=5,
            success_threshold=2,
            timeout=30.0,  # 30 seconds
            expected_exception=(OperationalError, DisconnectionError)
        )
    
    return _database_circuit_breaker


def get_tws_circuit_breaker() -> CircuitBreaker:
    """Get or create global TWS circuit breaker"""
    global _tws_circuit_breaker
    
    if _tws_circuit_breaker is None:
        _tws_circuit_breaker = CircuitBreaker(
            name="tws_connection",
            failure_threshold=3,
            success_threshold=2,
            timeout=60.0,  # 60 seconds
            expected_exception=Exception  # Generic for TWS errors
        )
    
    return _tws_circuit_breaker

