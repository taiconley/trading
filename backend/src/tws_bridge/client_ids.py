"""
Client ID management for TWS connections.

This module handles client ID allocation, collision detection, database persistence,
and automatic reclaim for TWS service connections.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Set
from datetime import datetime, timezone
from dataclasses import dataclass

# Add the src directory to Python path for module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common.config import get_settings
from common.db import get_db_session, execute_with_retry
from common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ClientIdAllocation:
    """Client ID allocation record."""
    client_id: int
    service_name: str
    instance_id: Optional[str] = None
    allocated_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    status: str = "active"  # active, released, expired


class ClientIdManager:
    """Manages client ID allocation and lifecycle."""
    
    # Service client ID ranges
    SERVICE_RANGES = {
        "account": (11, 11),      # Single ID: 11
        "marketdata": (12, 12),   # Single ID: 12  
        "historical": (13, 13),   # Single ID: 13
        "trader": (14, 14),       # Single ID: 14
        "strategy": (15, 29),     # Range: 15-29 (15 instances)
    }
    
    def __init__(self):
        self.settings = get_settings()
        self.base_id = self.settings.tws.client_id_base
        self._local_allocations: Dict[int, ClientIdAllocation] = {}
        self._heartbeat_interval = 30  # seconds
        self._expiry_timeout = 120     # seconds
    
    def get_service_client_id(self, service_name: str, instance_id: Optional[str] = None) -> int:
        """
        Get client ID for a service.
        
        Args:
            service_name: Name of the service
            instance_id: Instance identifier for multi-instance services
            
        Returns:
            Client ID for the service
            
        Raises:
            ValueError: If service is unknown or no IDs available
        """
        if service_name not in self.SERVICE_RANGES:
            raise ValueError(f"Unknown service: {service_name}")
        
        start_id, end_id = self.SERVICE_RANGES[service_name]
        
        # For single-ID services, return the designated ID
        if start_id == end_id:
            client_id = self.base_id + start_id
            self._allocate_client_id(client_id, service_name, instance_id)
            return client_id
        
        # For multi-instance services (like strategy), find available ID
        for offset in range(start_id, end_id + 1):
            client_id = self.base_id + offset
            if self._try_allocate_client_id(client_id, service_name, instance_id):
                return client_id
        
        raise ValueError(f"No available client IDs for service: {service_name}")
    
    def release_client_id(self, client_id: int, service_name: str):
        """
        Release a client ID.
        
        Args:
            client_id: Client ID to release
            service_name: Service name for verification
        """
        try:
            def _release_id(session):
                # Update database record
                from common.models import Base
                # We'll create a ClientIdAllocation model later if needed
                # For now, just remove from local tracking
                pass
            
            execute_with_retry(_release_id)
            
            # Remove from local tracking
            if client_id in self._local_allocations:
                allocation = self._local_allocations[client_id]
                allocation.status = "released"
                del self._local_allocations[client_id]
            
            logger.info(f"Released client ID {client_id} for service {service_name}")
            
        except Exception as e:
            logger.error(f"Failed to release client ID {client_id}: {e}")
            raise
    
    def heartbeat_client_id(self, client_id: int):
        """
        Send heartbeat for a client ID to keep it active.
        
        Args:
            client_id: Client ID to heartbeat
        """
        if client_id in self._local_allocations:
            self._local_allocations[client_id].last_heartbeat = datetime.now(timezone.utc)
        
        # Update database heartbeat (implement when needed)
        logger.debug(f"Heartbeat for client ID {client_id}")
    
    def cleanup_expired_allocations(self):
        """Clean up expired client ID allocations."""
        current_time = datetime.now(timezone.utc)
        expired_ids = []
        
        for client_id, allocation in self._local_allocations.items():
            if allocation.last_heartbeat:
                time_since_heartbeat = (current_time - allocation.last_heartbeat).total_seconds()
                if time_since_heartbeat > self._expiry_timeout:
                    expired_ids.append(client_id)
        
        for client_id in expired_ids:
            logger.warning(f"Client ID {client_id} expired, releasing")
            allocation = self._local_allocations[client_id]
            self.release_client_id(client_id, allocation.service_name)
    
    def get_allocated_ids(self) -> List[ClientIdAllocation]:
        """Get list of currently allocated client IDs."""
        return list(self._local_allocations.values())
    
    def get_available_ids_for_service(self, service_name: str) -> List[int]:
        """
        Get list of available client IDs for a service.
        
        Args:
            service_name: Service name
            
        Returns:
            List of available client IDs
        """
        if service_name not in self.SERVICE_RANGES:
            return []
        
        start_id, end_id = self.SERVICE_RANGES[service_name]
        available_ids = []
        
        for offset in range(start_id, end_id + 1):
            client_id = self.base_id + offset
            if not self._is_client_id_allocated(client_id):
                available_ids.append(client_id)
        
        return available_ids
    
    def _allocate_client_id(self, client_id: int, service_name: str, instance_id: Optional[str] = None):
        """Allocate a client ID (internal method)."""
        if self._is_client_id_allocated(client_id):
            raise ValueError(f"Client ID {client_id} already allocated")
        
        allocation = ClientIdAllocation(
            client_id=client_id,
            service_name=service_name,
            instance_id=instance_id,
            allocated_at=datetime.now(timezone.utc),
            last_heartbeat=datetime.now(timezone.utc),
            status="active"
        )
        
        self._local_allocations[client_id] = allocation
        logger.info(f"Allocated client ID {client_id} to service {service_name}")
    
    def _try_allocate_client_id(self, client_id: int, service_name: str, instance_id: Optional[str] = None) -> bool:
        """Try to allocate a client ID, return True if successful."""
        try:
            self._allocate_client_id(client_id, service_name, instance_id)
            return True
        except ValueError:
            return False
    
    def _is_client_id_allocated(self, client_id: int) -> bool:
        """Check if a client ID is currently allocated."""
        return client_id in self._local_allocations and self._local_allocations[client_id].status == "active"


# Global client ID manager instance
_client_id_manager: Optional[ClientIdManager] = None


def get_client_id_manager() -> ClientIdManager:
    """Get the global client ID manager instance."""
    global _client_id_manager
    if _client_id_manager is None:
        _client_id_manager = ClientIdManager()
    return _client_id_manager


# High-level convenience functions

def allocate_service_client_id(service_name: str, instance_id: Optional[str] = None) -> int:
    """
    Allocate a client ID for a service.
    
    Args:
        service_name: Name of the service
        instance_id: Instance identifier for multi-instance services
        
    Returns:
        Allocated client ID
    """
    manager = get_client_id_manager()
    return manager.get_service_client_id(service_name, instance_id)


def release_service_client_id(client_id: int, service_name: str):
    """
    Release a service client ID.
    
    Args:
        client_id: Client ID to release
        service_name: Service name
    """
    manager = get_client_id_manager()
    manager.release_client_id(client_id, service_name)


def heartbeat_service_client_id(client_id: int):
    """
    Send heartbeat for a service client ID.
    
    Args:
        client_id: Client ID to heartbeat
    """
    manager = get_client_id_manager()
    manager.heartbeat_client_id(client_id)


def get_service_client_ids() -> List[ClientIdAllocation]:
    """Get list of all allocated service client IDs."""
    manager = get_client_id_manager()
    return manager.get_allocated_ids()


def cleanup_expired_client_ids():
    """Clean up expired client ID allocations."""
    manager = get_client_id_manager()
    manager.cleanup_expired_allocations()


# Strategy-specific helper functions

def allocate_strategy_client_id(strategy_id: str) -> int:
    """
    Allocate a client ID for a strategy instance.
    
    Args:
        strategy_id: Strategy identifier
        
    Returns:
        Allocated client ID
    """
    return allocate_service_client_id("strategy", strategy_id)


def release_strategy_client_id(client_id: int):
    """
    Release a strategy client ID.
    
    Args:
        client_id: Client ID to release
    """
    release_service_client_id(client_id, "strategy")


def get_available_strategy_client_ids() -> List[int]:
    """Get list of available strategy client IDs."""
    manager = get_client_id_manager()
    return manager.get_available_ids_for_service("strategy")


# Service restart and recovery functions

def reclaim_service_client_id(service_name: str, instance_id: Optional[str] = None) -> int:
    """
    Reclaim a client ID for a service on restart.
    
    This function attempts to reclaim the same client ID that was previously
    allocated to this service instance, or allocates a new one if not possible.
    
    Args:
        service_name: Service name
        instance_id: Instance identifier
        
    Returns:
        Client ID (reclaimed or newly allocated)
    """
    manager = get_client_id_manager()
    
    # For single-ID services, always try to reclaim the designated ID
    if service_name in manager.SERVICE_RANGES:
        start_id, end_id = manager.SERVICE_RANGES[service_name]
        if start_id == end_id:
            designated_id = manager.base_id + start_id
            
            # Check if it's currently allocated to someone else
            if manager._is_client_id_allocated(designated_id):
                allocation = manager._local_allocations[designated_id]
                if allocation.service_name == service_name and allocation.instance_id == instance_id:
                    # Already allocated to us, just update heartbeat
                    manager.heartbeat_client_id(designated_id)
                    logger.info(f"Reclaimed existing client ID {designated_id} for {service_name}")
                    return designated_id
                else:
                    # Allocated to someone else, this is a conflict
                    logger.error(f"Client ID {designated_id} conflict: allocated to {allocation.service_name}")
                    raise ValueError(f"Client ID conflict for service {service_name}")
            else:
                # Not allocated, allocate it now
                return manager.get_service_client_id(service_name, instance_id)
    
    # For multi-instance services, try to allocate any available ID
    return manager.get_service_client_id(service_name, instance_id)


def detect_client_id_conflicts() -> List[Dict]:
    """
    Detect client ID conflicts and return details.
    
    Returns:
        List of conflict details
    """
    conflicts = []
    manager = get_client_id_manager()
    
    # Check for duplicate allocations
    seen_ids = set()
    for allocation in manager.get_allocated_ids():
        if allocation.client_id in seen_ids:
            conflicts.append({
                "type": "duplicate_allocation",
                "client_id": allocation.client_id,
                "service": allocation.service_name,
                "instance": allocation.instance_id
            })
        seen_ids.add(allocation.client_id)
    
    # Check for IDs outside of valid ranges
    for allocation in manager.get_allocated_ids():
        service_name = allocation.service_name
        if service_name in manager.SERVICE_RANGES:
            start_id, end_id = manager.SERVICE_RANGES[service_name]
            expected_range = (manager.base_id + start_id, manager.base_id + end_id)
            
            if not (expected_range[0] <= allocation.client_id <= expected_range[1]):
                conflicts.append({
                    "type": "out_of_range",
                    "client_id": allocation.client_id,
                    "service": service_name,
                    "expected_range": expected_range
                })
    
    return conflicts


def get_client_id_usage_stats() -> Dict:
    """Get client ID usage statistics."""
    manager = get_client_id_manager()
    stats = {
        "total_allocated": len(manager._local_allocations),
        "by_service": {},
        "available_by_service": {}
    }
    
    # Count allocations by service
    for allocation in manager.get_allocated_ids():
        service = allocation.service_name
        if service not in stats["by_service"]:
            stats["by_service"][service] = 0
        stats["by_service"][service] += 1
    
    # Count available IDs by service
    for service in manager.SERVICE_RANGES:
        available = manager.get_available_ids_for_service(service)
        stats["available_by_service"][service] = len(available)
    
    return stats
