"""
Database utilities and connection management.

This module provides SQLAlchemy engine creation, session management,
connection pooling, retry logic, and health checks for the trading bot database.
"""

import time
import logging
from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError, OperationalError
from .config import get_settings

logger = logging.getLogger(__name__)

# Global engine and session factory
_engine: Optional[Engine] = None
_SessionFactory: Optional[sessionmaker] = None


def create_database_engine(echo: bool = False) -> Engine:
    """
    Create SQLAlchemy engine with connection pooling and retry logic.
    
    Args:
        echo: Whether to echo SQL statements (for debugging)
        
    Returns:
        Configured SQLAlchemy engine
    """
    settings = get_settings()
    
    engine = create_engine(
        settings.database.url,
        echo=echo,
        poolclass=QueuePool,
        pool_size=10,  # Number of connections to maintain in the pool
        max_overflow=20,  # Additional connections that can be created on demand
        pool_pre_ping=True,  # Validate connections before use
        pool_recycle=3600,  # Recycle connections after 1 hour
        connect_args={
            "connect_timeout": 10,
            "application_name": f"trading_bot_{settings.service_name or 'unknown'}",
        }
    )
    
    # Add connection event listeners for logging and monitoring
    @event.listens_for(engine, "connect")
    def receive_connect(dbapi_connection, connection_record):
        logger.debug("Database connection established")
    
    @event.listens_for(engine, "checkout")
    def receive_checkout(dbapi_connection, connection_record, connection_proxy):
        logger.debug("Connection checked out from pool")
    
    @event.listens_for(engine, "checkin")
    def receive_checkin(dbapi_connection, connection_record):
        logger.debug("Connection checked back into pool")
    
    return engine


def get_engine() -> Engine:
    """Get the global database engine, creating it if necessary."""
    global _engine
    if _engine is None:
        _engine = create_database_engine()
    return _engine


def get_session_factory() -> sessionmaker:
    """Get the global session factory, creating it if necessary."""
    global _SessionFactory
    if _SessionFactory is None:
        engine = get_engine()
        _SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)
    return _SessionFactory


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Get a database session with automatic cleanup.
    
    Usage:
        with get_db_session() as session:
            # Use session here
            session.query(...)
            session.commit()
    
    Yields:
        SQLAlchemy session
    """
    SessionFactory = get_session_factory()
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def execute_with_retry(
    operation_func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential_base: float = 2.0
):
    """
    Execute a database operation with exponential backoff retry logic.
    
    Args:
        operation_func: Function to execute (should take a session parameter)
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff calculation
        
    Returns:
        Result of operation_func
        
    Raises:
        SQLAlchemyError: If all retries are exhausted
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            with get_db_session() as session:
                return operation_func(session)
        except (OperationalError, DisconnectionError) as e:
            last_exception = e
            
            if attempt == max_retries:
                logger.error(f"Database operation failed after {max_retries} retries: {e}")
                raise
            
            # Calculate delay with exponential backoff and jitter
            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            jitter = delay * 0.1  # 10% jitter
            actual_delay = delay + (jitter * (0.5 - time.time() % 1))
            
            logger.warning(f"Database operation failed (attempt {attempt + 1}), retrying in {actual_delay:.2f}s: {e}")
            time.sleep(actual_delay)
        except SQLAlchemyError:
            # Don't retry for other SQL errors (syntax errors, constraint violations, etc.)
            raise
    
    # This should never be reached, but just in case
    raise last_exception


def check_database_health() -> dict:
    """
    Check database connectivity and return health status.
    
    Returns:
        Dictionary with health information
    """
    try:
        def health_check(session: Session):
            # Simple query to test connectivity
            result = session.execute(text("SELECT 1 as health_check"))
            return result.fetchone()[0] == 1
        
        start_time = time.time()
        is_healthy = execute_with_retry(health_check, max_retries=1)
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "response_time_ms": round(response_time, 2),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }


def get_database_info() -> dict:
    """
    Get database connection information.
    
    Returns:
        Dictionary with database information
    """
    try:
        engine = get_engine()
        
        def get_info(session: Session):
            # Get database version and connection info
            version_result = session.execute(text("SELECT version()"))
            version = version_result.fetchone()[0]
            
            # Get current database name
            db_result = session.execute(text("SELECT current_database()"))
            database = db_result.fetchone()[0]
            
            return {
                "database": database,
                "version": version,
                "pool_size": engine.pool.size(),
                "checked_out_connections": engine.pool.checkedout(),
                "overflow_connections": engine.pool.overflow(),
                "invalid_connections": engine.pool.invalidated(),
            }
        
        return execute_with_retry(get_info, max_retries=1)
    except Exception as e:
        logger.error(f"Failed to get database info: {e}")
        return {"error": str(e)}


def initialize_database():
    """
    Initialize database connection and verify connectivity.
    
    This function should be called during application startup.
    """
    try:
        logger.info("Initializing database connection...")
        
        # Create engine and test connectivity
        engine = get_engine()
        health = check_database_health()
        
        if health["status"] == "healthy":
            logger.info(f"Database connection established successfully (response time: {health['response_time_ms']}ms)")
            
            # Log database information
            info = get_database_info()
            if "error" not in info:
                logger.info(f"Connected to database: {info['database']}")
                logger.info(f"Pool status - Size: {info['pool_size']}, Checked out: {info['checked_out_connections']}")
        else:
            logger.error(f"Database health check failed: {health.get('error', 'Unknown error')}")
            raise ConnectionError("Failed to establish database connection")
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def close_database_connections():
    """
    Close all database connections.
    
    This function should be called during application shutdown.
    """
    global _engine, _SessionFactory
    
    try:
        if _engine is not None:
            logger.info("Closing database connections...")
            _engine.dispose()
            _engine = None
            _SessionFactory = None
            logger.info("Database connections closed successfully")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")


# Utility functions for common database operations

def upsert_health_status(service_name: str, status: str):
    """
    Upsert service health status.
    
    Args:
        service_name: Name of the service
        status: Health status (healthy, unhealthy, starting, stopping)
    """
    from .models import HealthStatus
    
    def _upsert(session: Session):
        health_record = session.query(HealthStatus).filter(
            HealthStatus.service == service_name
        ).first()
        
        if health_record:
            health_record.status = status
            health_record.updated_at = time.time()
        else:
            health_record = HealthStatus(
                service=service_name,
                status=status
            )
            session.add(health_record)
        
        session.commit()
        return health_record
    
    return execute_with_retry(_upsert)


def get_service_health_statuses() -> list:
    """
    Get all service health statuses.
    
    Returns:
        List of health status records
    """
    from .models import HealthStatus
    
    def _get_statuses(session: Session):
        return session.query(HealthStatus).all()
    
    return execute_with_retry(_get_statuses)
