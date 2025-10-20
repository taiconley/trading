"""
Graceful Shutdown Handler

Provides standardized signal handling for graceful shutdown across all services.
Handles SIGTERM and SIGINT signals, ensuring proper cleanup of resources,
connections, and in-flight operations.

Usage:
    from common.shutdown import GracefulShutdownHandler
    
    shutdown_handler = GracefulShutdownHandler("my_service")
    
    # Register cleanup tasks
    shutdown_handler.add_cleanup_task(close_database_connections)
    shutdown_handler.add_cleanup_task(disconnect_from_tws)
    
    # Install signal handlers
    shutdown_handler.install()
    
    # In your main loop
    while not shutdown_handler.should_shutdown():
        # Do work...
        await asyncio.sleep(1)
"""

import signal
import asyncio
import logging
from typing import Callable, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class GracefulShutdownHandler:
    """
    Handler for graceful shutdown of services.
    
    Manages signal handling, cleanup tasks, and shutdown state.
    """
    
    def __init__(
        self,
        service_name: str,
        timeout: float = 30.0,
        handle_sigterm: bool = True,
        handle_sigint: bool = True
    ):
        """
        Initialize graceful shutdown handler.
        
        Args:
            service_name: Name of the service for logging
            timeout: Maximum time to wait for cleanup (seconds)
            handle_sigterm: Whether to handle SIGTERM signal
            handle_sigint: Whether to handle SIGINT signal
        """
        self.service_name = service_name
        self.timeout = timeout
        self.handle_sigterm = handle_sigterm
        self.handle_sigint = handle_sigint
        
        self._shutdown_requested = False
        self._shutdown_complete = False
        self._cleanup_tasks: List[Callable] = []
        self._shutdown_start_time: Optional[datetime] = None
        
        logger.info(f"Graceful shutdown handler initialized for {service_name}")
    
    def add_cleanup_task(self, task: Callable):
        """
        Add a cleanup task to be executed during shutdown.
        
        Tasks are executed in the order they are added.
        
        Args:
            task: Callable (sync or async) to execute during cleanup
        """
        self._cleanup_tasks.append(task)
        logger.debug(f"Registered cleanup task: {task.__name__}")
    
    def should_shutdown(self) -> bool:
        """Check if shutdown has been requested"""
        return self._shutdown_requested
    
    def is_shutdown_complete(self) -> bool:
        """Check if shutdown is complete"""
        return self._shutdown_complete
    
    def install(self):
        """Install signal handlers"""
        if self.handle_sigterm:
            signal.signal(signal.SIGTERM, self._signal_handler)
            logger.info("Installed SIGTERM handler")
        
        if self.handle_sigint:
            signal.signal(signal.SIGINT, self._signal_handler)
            logger.info("Installed SIGINT handler")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        signal_name = signal.Signals(signum).name
        logger.warning(f"Received {signal_name} signal - initiating graceful shutdown")
        
        self._shutdown_requested = True
        self._shutdown_start_time = datetime.now(timezone.utc)
        
        # Create async task to run cleanup
        try:
            # Try to get or create event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Schedule cleanup
            if not loop.is_closed():
                asyncio.ensure_future(self._execute_cleanup())
        except Exception as e:
            logger.error(f"Error scheduling cleanup: {e}")
            # Fallback to synchronous cleanup
            self._execute_cleanup_sync()
    
    async def _execute_cleanup(self):
        """Execute all cleanup tasks asynchronously"""
        logger.info(f"Starting graceful shutdown for {self.service_name}")
        
        try:
            # Execute all cleanup tasks
            for i, task in enumerate(self._cleanup_tasks, 1):
                try:
                    task_name = task.__name__
                    logger.info(f"Executing cleanup task {i}/{len(self._cleanup_tasks)}: {task_name}")
                    
                    # Handle both sync and async tasks
                    if asyncio.iscoroutinefunction(task):
                        await asyncio.wait_for(task(), timeout=self.timeout / len(self._cleanup_tasks))
                    else:
                        await asyncio.get_event_loop().run_in_executor(None, task)
                    
                    logger.info(f"Cleanup task completed: {task_name}")
                    
                except asyncio.TimeoutError:
                    logger.error(f"Cleanup task timed out: {task_name}")
                except Exception as e:
                    logger.error(f"Error in cleanup task {task_name}: {e}")
                    # Continue with other cleanup tasks
            
            self._shutdown_complete = True
            
            if self._shutdown_start_time:
                duration = (datetime.now(timezone.utc) - self._shutdown_start_time).total_seconds()
                logger.info(f"Graceful shutdown completed in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            logger.info(f"Service {self.service_name} shutdown")
    
    def _execute_cleanup_sync(self):
        """Execute all cleanup tasks synchronously (fallback)"""
        logger.info(f"Starting synchronous cleanup for {self.service_name}")
        
        for task in self._cleanup_tasks:
            try:
                task_name = task.__name__
                logger.info(f"Executing cleanup task: {task_name}")
                
                # Only call if it's synchronous
                if not asyncio.iscoroutinefunction(task):
                    task()
                else:
                    logger.warning(f"Skipping async task in sync cleanup: {task_name}")
                
            except Exception as e:
                logger.error(f"Error in cleanup task {task_name}: {e}")
        
        self._shutdown_complete = True
        logger.info(f"Synchronous cleanup completed for {self.service_name}")
    
    async def shutdown(self):
        """Manually trigger shutdown"""
        if not self._shutdown_requested:
            logger.info(f"Manual shutdown requested for {self.service_name}")
            self._shutdown_requested = True
            self._shutdown_start_time = datetime.now(timezone.utc)
            await self._execute_cleanup()
    
    def get_status(self) -> dict:
        """Get current shutdown status"""
        status = {
            "service": self.service_name,
            "shutdown_requested": self._shutdown_requested,
            "shutdown_complete": self._shutdown_complete,
            "cleanup_tasks": len(self._cleanup_tasks),
        }
        
        if self._shutdown_start_time:
            status["shutdown_start_time"] = self._shutdown_start_time.isoformat()
            
            if self._shutdown_complete:
                duration = (datetime.now(timezone.utc) - self._shutdown_start_time).total_seconds()
                status["shutdown_duration_seconds"] = round(duration, 2)
        
        return status


# Global shutdown handler for convenience
_global_shutdown_handler: Optional[GracefulShutdownHandler] = None


def get_shutdown_handler(service_name: Optional[str] = None) -> GracefulShutdownHandler:
    """
    Get or create the global shutdown handler.
    
    Args:
        service_name: Name of the service (required on first call)
        
    Returns:
        Global shutdown handler instance
    """
    global _global_shutdown_handler
    
    if _global_shutdown_handler is None:
        if service_name is None:
            raise ValueError("service_name required when creating shutdown handler")
        
        _global_shutdown_handler = GracefulShutdownHandler(service_name)
    
    return _global_shutdown_handler


def register_cleanup(task: Callable):
    """
    Register a cleanup task with the global shutdown handler.
    
    Usage:
        @register_cleanup
        async def cleanup_database():
            await db.close()
    """
    handler = get_shutdown_handler()
    handler.add_cleanup_task(task)
    return task

