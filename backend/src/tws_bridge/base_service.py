"""
Base TWS service class with common functionality.

This module provides a base class that TWS-connected services can inherit from,
including connection management, health checks, graceful shutdown, and logging.
"""

import asyncio
import signal
import sys
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .ib_client import EnhancedIBClient, create_ib_client
from .client_ids import allocate_service_client_id, release_service_client_id, heartbeat_service_client_id
from ..common.config import get_settings
from ..common.logging import configure_service_logging, get_logger
from ..common.db import initialize_database, close_database_connections, upsert_health_status


class BaseTWSService(ABC):
    """Base class for TWS-connected services."""
    
    def __init__(self, service_name: str, instance_id: Optional[str] = None):
        """
        Initialize base TWS service.
        
        Args:
            service_name: Name of the service
            instance_id: Instance identifier for multi-instance services
        """
        self.service_name = service_name
        self.instance_id = instance_id
        self.settings = get_settings()
        
        # Setup logging
        self.logger = configure_service_logging(service_name)
        
        # Initialize components
        self.client_id: Optional[int] = None
        self.ib_client: Optional[EnhancedIBClient] = None
        self.app: Optional[FastAPI] = None
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Health status tracking
        self.last_health_update = None
        self.health_status = "starting"
        
        self.logger.info(f"Initializing {service_name} service")
    
    async def initialize(self):
        """Initialize the service."""
        try:
            self.logger.info("Initializing service components...")
            
            # Initialize database
            initialize_database()
            
            # Allocate client ID
            self.client_id = allocate_service_client_id(self.service_name, self.instance_id)
            self.logger.info(f"Allocated client ID: {self.client_id}")
            
            # Create IB client
            self.ib_client = create_ib_client(self.client_id)
            
            # Setup IB client event handlers
            self.ib_client.add_connection_handler(self._on_tws_connected)
            self.ib_client.add_disconnection_handler(self._on_tws_disconnected)
            self.ib_client.add_error_handler(self._on_tws_error)
            
            # Connect to TWS
            connected = await self.ib_client.connect()
            if not connected:
                raise ConnectionError("Failed to connect to TWS")
            
            # Service-specific initialization
            await self.on_initialize()
            
            # Create FastAPI app if needed
            if self.should_create_api():
                self.app = self.create_fastapi_app()
            
            self.health_status = "healthy"
            await self.update_health_status()
            
            self.logger.info("Service initialization completed")
            
        except Exception as e:
            self.health_status = "unhealthy"
            await self.update_health_status()
            self.logger.error(f"Service initialization failed: {e}")
            raise
    
    async def run(self):
        """Run the service."""
        try:
            self.running = True
            self.logger.info("Starting service...")
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # Start health monitoring
            health_task = asyncio.create_task(self._health_monitor_loop())
            
            # Start service-specific tasks
            service_tasks = await self.start_service_tasks()
            
            # Start API server if we have one
            api_task = None
            if self.app:
                api_task = asyncio.create_task(self._run_api_server())
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            self.logger.info("Shutdown signal received, stopping service...")
            
            # Cancel tasks
            health_task.cancel()
            if api_task:
                api_task.cancel()
            
            for task in service_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(health_task, *service_tasks, api_task, return_exceptions=True)
            
            self.logger.info("Service stopped")
            
        except Exception as e:
            self.logger.error(f"Service error: {e}")
            raise
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up service resources."""
        try:
            self.logger.info("Cleaning up service resources...")
            
            self.health_status = "stopping"
            await self.update_health_status()
            
            # Service-specific cleanup
            await self.on_cleanup()
            
            # Disconnect from TWS
            if self.ib_client:
                await self.ib_client.disconnect()
            
            # Release client ID
            if self.client_id:
                release_service_client_id(self.client_id, self.service_name)
            
            # Close database connections
            close_database_connections()
            
            self.running = False
            self.logger.info("Service cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def create_fastapi_app(self) -> FastAPI:
        """Create FastAPI application with common endpoints."""
        app = FastAPI(
            title=f"{self.service_name.title()} Service",
            description=f"Trading bot {self.service_name} service",
            version="1.0.0"
        )
        
        # Health check endpoint
        @app.get("/healthz")
        async def health_check():
            """Health check endpoint."""
            health_data = await self.get_health_status()
            status_code = 200 if health_data["status"] == "healthy" else 503
            return JSONResponse(content=health_data, status_code=status_code)
        
        # Status endpoint
        @app.get("/status")
        async def get_status():
            """Get service status."""
            return {
                "service": self.service_name,
                "client_id": self.client_id,
                "instance_id": self.instance_id,
                "running": self.running,
                "health": self.health_status,
                "tws_connected": self.ib_client.state.connected if self.ib_client else False
            }
        
        # Add service-specific endpoints
        self.add_api_endpoints(app)
        
        return app
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        health_data = {
            "status": self.health_status,
            "service": self.service_name,
            "timestamp": self.last_health_update.isoformat() if self.last_health_update else None,
            "details": {
                "client_id": self.client_id,
                "tws_connected": self.ib_client.state.connected if self.ib_client else False,
                "running": self.running
            }
        }
        
        # Add service-specific health details
        service_health = await self.get_service_health_details()
        if service_health:
            health_data["details"].update(service_health)
        
        return health_data
    
    async def update_health_status(self):
        """Update health status in database."""
        try:
            upsert_health_status(self.service_name, self.health_status)
            
            # Send heartbeat for client ID
            if self.client_id:
                heartbeat_service_client_id(self.client_id)
            
            from datetime import datetime, timezone
            self.last_health_update = datetime.now(timezone.utc)
            
        except Exception as e:
            self.logger.error(f"Failed to update health status: {e}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self._shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def _shutdown(self):
        """Initiate shutdown."""
        self.shutdown_event.set()
    
    async def _health_monitor_loop(self):
        """Health monitoring loop."""
        while self.running:
            try:
                await self.update_health_status()
                await asyncio.sleep(30)  # Update every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(5)
    
    async def _run_api_server(self):
        """Run the FastAPI server."""
        import uvicorn
        
        config = uvicorn.Config(
            app=self.app,
            host="0.0.0.0",
            port=8000,
            log_config=None  # Use our custom logging
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    def _on_tws_connected(self):
        """Handle TWS connection event."""
        self.logger.info("TWS connection established")
        asyncio.create_task(self.on_tws_connected())
    
    def _on_tws_disconnected(self):
        """Handle TWS disconnection event."""
        self.logger.warning("TWS connection lost")
        asyncio.create_task(self.on_tws_disconnected())
    
    def _on_tws_error(self, req_id, error_code, error_string, contract):
        """Handle TWS error event."""
        self.logger.error(f"TWS error: {error_code} - {error_string}")
        asyncio.create_task(self.on_tws_error(req_id, error_code, error_string, contract))
    
    # Abstract methods that subclasses must implement
    
    @abstractmethod
    async def on_initialize(self):
        """Service-specific initialization logic."""
        pass
    
    @abstractmethod
    async def start_service_tasks(self) -> list:
        """Start service-specific background tasks."""
        pass
    
    @abstractmethod
    async def on_cleanup(self):
        """Service-specific cleanup logic."""
        pass
    
    # Optional methods that subclasses can override
    
    def should_create_api(self) -> bool:
        """Whether this service should create a FastAPI app."""
        return True
    
    def add_api_endpoints(self, app: FastAPI):
        """Add service-specific API endpoints."""
        pass
    
    async def get_service_health_details(self) -> Optional[Dict[str, Any]]:
        """Get service-specific health details."""
        return None
    
    async def on_tws_connected(self):
        """Handle TWS connection event (override in subclass)."""
        pass
    
    async def on_tws_disconnected(self):
        """Handle TWS disconnection event (override in subclass)."""
        pass
    
    async def on_tws_error(self, req_id, error_code, error_string, contract):
        """Handle TWS error event (override in subclass)."""
        pass


@asynccontextmanager
async def tws_service_context(service_class, service_name: str, instance_id: Optional[str] = None):
    """Context manager for TWS service lifecycle."""
    service = service_class(service_name, instance_id)
    
    try:
        await service.initialize()
        yield service
    finally:
        await service.cleanup()


def run_tws_service(service_class, service_name: str, instance_id: Optional[str] = None):
    """
    Run a TWS service with proper lifecycle management.
    
    Args:
        service_class: Service class to instantiate
        service_name: Name of the service
        instance_id: Instance identifier for multi-instance services
    """
    async def _run_service():
        async with tws_service_context(service_class, service_name, instance_id) as service:
            await service.run()
    
    try:
        asyncio.run(_run_service())
    except KeyboardInterrupt:
        print("Service interrupted by user")
    except Exception as e:
        print(f"Service failed: {e}")
        sys.exit(1)
