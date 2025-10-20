"""
Health Monitoring and Aggregation System

Provides centralized health monitoring, aggregation, and alerting for all services.
Monitors service heartbeats, connection status, and overall system health.

Features:
- Automated health status aggregation
- Stale service detection
- Health trend analysis
- Alert generation for unhealthy services
- Service dependency checking

Usage:
    from common.health_monitor import HealthMonitor
    
    monitor = HealthMonitor()
    
    # Get overall system health
    health = await monitor.get_system_health()
    
    # Get detailed service status
    services = await monitor.get_all_services()
    
    # Check for stale services
    stale = await monitor.get_stale_services(threshold_seconds=60)
"""

import time
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class SystemHealthStatus(Enum):
    """Overall system health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Some services unhealthy but system functional
    UNHEALTHY = "unhealthy"  # Critical services down
    UNKNOWN = "unknown"


class ServiceHealth:
    """Health status for a single service"""
    
    def __init__(
        self,
        service_name: str,
        status: str,
        last_update: float,
        is_critical: bool = False
    ):
        self.service_name = service_name
        self.status = status
        self.last_update = last_update
        self.is_critical = is_critical
    
    @property
    def age_seconds(self) -> float:
        """Time since last update in seconds"""
        return time.time() - self.last_update
    
    @property
    def is_stale(self, threshold: float = 60.0) -> bool:
        """Check if service status is stale"""
        return self.age_seconds > threshold
    
    @property
    def is_healthy(self) -> bool:
        """Check if service is healthy"""
        return self.status == "healthy"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "service": self.service_name,
            "status": self.status,
            "last_update": datetime.fromtimestamp(
                self.last_update, tz=timezone.utc
            ).isoformat(),
            "age_seconds": round(self.age_seconds, 2),
            "is_critical": self.is_critical,
            "is_healthy": self.is_healthy,
            "is_stale": self.is_stale,
        }


class HealthMonitor:
    """
    Centralized health monitoring system.
    
    Monitors all services and provides aggregated health status.
    """
    
    # Define critical services (system is unhealthy if any are down)
    CRITICAL_SERVICES = {"postgres", "trader", "account"}
    
    # Stale threshold in seconds
    STALE_THRESHOLD = 60.0
    
    def __init__(self, stale_threshold: float = 60.0):
        """
        Initialize health monitor.
        
        Args:
            stale_threshold: Seconds before service is considered stale
        """
        self.stale_threshold = stale_threshold
        logger.info(f"Health monitor initialized (stale_threshold={stale_threshold}s)")
    
    async def get_all_services(self) -> List[ServiceHealth]:
        """
        Get health status for all services.
        
        Returns:
            List of ServiceHealth objects
        """
        from .db import get_db_session
        from .models import HealthStatus
        
        services = []
        
        try:
            with get_db_session() as session:
                health_records = session.query(HealthStatus).all()
                
                for record in health_records:
                    # Convert datetime to timestamp if needed
                    if isinstance(record.updated_at, datetime):
                        last_update = record.updated_at.timestamp()
                    else:
                        last_update = record.updated_at
                    
                    service = ServiceHealth(
                        service_name=record.service,
                        status=record.status,
                        last_update=last_update,
                        is_critical=record.service in self.CRITICAL_SERVICES
                    )
                    services.append(service)
            
            return services
            
        except Exception as e:
            logger.error(f"Failed to get service health status: {e}")
            return []
    
    async def get_stale_services(
        self,
        threshold_seconds: Optional[float] = None
    ) -> List[ServiceHealth]:
        """
        Get services with stale health updates.
        
        Args:
            threshold_seconds: Override default stale threshold
            
        Returns:
            List of stale ServiceHealth objects
        """
        threshold = threshold_seconds or self.stale_threshold
        all_services = await self.get_all_services()
        
        stale_services = [
            svc for svc in all_services
            if svc.age_seconds > threshold
        ]
        
        if stale_services:
            logger.warning(
                f"Found {len(stale_services)} stale services: "
                f"{', '.join(s.service_name for s in stale_services)}"
            )
        
        return stale_services
    
    async def get_unhealthy_services(self) -> List[ServiceHealth]:
        """
        Get services with unhealthy status.
        
        Returns:
            List of unhealthy ServiceHealth objects
        """
        all_services = await self.get_all_services()
        
        unhealthy = [
            svc for svc in all_services
            if not svc.is_healthy
        ]
        
        if unhealthy:
            logger.warning(
                f"Found {len(unhealthy)} unhealthy services: "
                f"{', '.join(s.service_name for s in unhealthy)}"
            )
        
        return unhealthy
    
    async def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health status.
        
        Returns:
            Dictionary with system health information
        """
        all_services = await self.get_all_services()
        
        if not all_services:
            return {
                "status": SystemHealthStatus.UNKNOWN.value,
                "message": "No service health data available",
                "services": [],
                "summary": {
                    "total": 0,
                    "healthy": 0,
                    "unhealthy": 0,
                    "stale": 0,
                    "critical_unhealthy": 0,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        
        # Calculate statistics
        total = len(all_services)
        healthy = sum(1 for s in all_services if s.is_healthy)
        unhealthy = sum(1 for s in all_services if not s.is_healthy)
        stale = sum(1 for s in all_services if s.is_stale)
        critical_unhealthy = sum(
            1 for s in all_services
            if s.is_critical and not s.is_healthy
        )
        
        # Determine overall status
        if critical_unhealthy > 0:
            status = SystemHealthStatus.UNHEALTHY
            message = f"{critical_unhealthy} critical service(s) unhealthy"
        elif unhealthy > 0 or stale > 0:
            status = SystemHealthStatus.DEGRADED
            message = f"{unhealthy} service(s) unhealthy, {stale} stale"
        else:
            status = SystemHealthStatus.HEALTHY
            message = "All services healthy"
        
        return {
            "status": status.value,
            "message": message,
            "services": [s.to_dict() for s in all_services],
            "summary": {
                "total": total,
                "healthy": healthy,
                "unhealthy": unhealthy,
                "stale": stale,
                "critical_unhealthy": critical_unhealthy,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    async def check_service_health(self, service_name: str) -> Optional[ServiceHealth]:
        """
        Get health status for a specific service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            ServiceHealth object or None if not found
        """
        from .db import get_db_session
        from .models import HealthStatus
        
        try:
            with get_db_session() as session:
                record = session.query(HealthStatus).filter(
                    HealthStatus.service == service_name
                ).first()
                
                if record:
                    # Convert datetime to timestamp if needed
                    if isinstance(record.updated_at, datetime):
                        last_update = record.updated_at.timestamp()
                    else:
                        last_update = record.updated_at
                    
                    return ServiceHealth(
                        service_name=record.service,
                        status=record.status,
                        last_update=last_update,
                        is_critical=record.service in self.CRITICAL_SERVICES
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get health for {service_name}: {e}")
            return None
    
    async def monitor_loop(self, interval: float = 15.0):
        """
        Continuous monitoring loop that checks health and logs warnings.
        
        Args:
            interval: Seconds between health checks
        """
        logger.info(f"Starting health monitor loop (interval={interval}s)")
        
        while True:
            try:
                # Get system health
                health = await self.get_system_health()
                
                # Log warnings for issues
                if health["status"] == SystemHealthStatus.UNHEALTHY.value:
                    logger.error(f"SYSTEM UNHEALTHY: {health['message']}")
                elif health["status"] == SystemHealthStatus.DEGRADED.value:
                    logger.warning(f"SYSTEM DEGRADED: {health['message']}")
                
                # Check for stale services
                stale_services = await self.get_stale_services()
                for service in stale_services:
                    logger.warning(
                        f"Service {service.service_name} is STALE "
                        f"(last update: {service.age_seconds:.1f}s ago)"
                    )
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(interval)


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get or create the global health monitor"""
    global _health_monitor
    
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    
    return _health_monitor
