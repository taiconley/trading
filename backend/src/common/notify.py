"""
Postgres LISTEN/NOTIFY wrapper for inter-service communication.

This module provides event-driven communication between services using
PostgreSQL's LISTEN/NOTIFY functionality for real-time updates.
"""

import asyncio
import json
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union
from contextlib import contextmanager
import psycopg2
import psycopg2.extensions
from psycopg2.extras import RealDictCursor

from .config import get_settings
from .logging import get_logger

logger = get_logger(__name__)


class NotificationManager:
    """Manages PostgreSQL LISTEN/NOTIFY connections and handlers."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize notification manager.
        
        Args:
            connection_string: Database connection string (uses config if None)
        """
        if connection_string is None:
            settings = get_settings()
            connection_string = settings.database.url
        
        self.connection_string = connection_string
        self.connection: Optional[psycopg2.extensions.connection] = None
        self.handlers: Dict[str, List[Callable]] = {}
        self.running = False
        self.listen_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def connect(self):
        """Establish database connection for notifications."""
        try:
            self.connection = psycopg2.connect(
                self.connection_string,
                cursor_factory=RealDictCursor
            )
            self.connection.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            logger.info("Notification manager connected to database")
        except Exception as e:
            logger.error(f"Failed to connect notification manager: {e}")
            raise
    
    def disconnect(self):
        """Close database connection."""
        if self.connection:
            try:
                self.connection.close()
                self.connection = None
                logger.info("Notification manager disconnected from database")
            except Exception as e:
                logger.error(f"Error disconnecting notification manager: {e}")
    
    def add_handler(self, channel: str, handler: Callable[[str, Dict], None]):
        """
        Add a handler for a notification channel.
        
        Args:
            channel: Channel name to listen on
            handler: Function to call when notification received
        """
        with self._lock:
            if channel not in self.handlers:
                self.handlers[channel] = []
            self.handlers[channel].append(handler)
            logger.debug(f"Added handler for channel: {channel}")
    
    def remove_handler(self, channel: str, handler: Callable[[str, Dict], None]):
        """
        Remove a handler for a notification channel.
        
        Args:
            channel: Channel name
            handler: Handler function to remove
        """
        with self._lock:
            if channel in self.handlers:
                try:
                    self.handlers[channel].remove(handler)
                    if not self.handlers[channel]:
                        del self.handlers[channel]
                    logger.debug(f"Removed handler for channel: {channel}")
                except ValueError:
                    logger.warning(f"Handler not found for channel: {channel}")
    
    def start_listening(self):
        """Start listening for notifications in a background thread."""
        if self.running:
            logger.warning("Notification manager already running")
            return
        
        if not self.connection:
            self.connect()
        
        # Subscribe to all channels with handlers
        cursor = self.connection.cursor()
        for channel in self.handlers.keys():
            cursor.execute(f"LISTEN {channel}")
            logger.info(f"Listening on channel: {channel}")
        
        self.running = True
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()
        logger.info("Started notification listener thread")
    
    def stop_listening(self):
        """Stop listening for notifications."""
        self.running = False
        if self.listen_thread:
            self.listen_thread.join(timeout=5.0)
            self.listen_thread = None
        
        if self.connection:
            cursor = self.connection.cursor()
            cursor.execute("UNLISTEN *")
        
        logger.info("Stopped notification listener")
    
    def _listen_loop(self):
        """Main listening loop (runs in background thread)."""
        logger.info("Notification listener thread started")
        
        while self.running:
            try:
                if not self.connection:
                    logger.error("Database connection lost, attempting to reconnect")
                    self.connect()
                    continue
                
                # Check for notifications
                self.connection.poll()
                
                while self.connection.notifies:
                    notify = self.connection.notifies.pop(0)
                    self._handle_notification(notify.channel, notify.payload)
                
                # Small sleep to prevent busy waiting
                time.sleep(0.1)
                
            except psycopg2.OperationalError as e:
                logger.error(f"Database connection error in listener: {e}")
                self.connection = None
                time.sleep(1.0)  # Wait before retry
            except Exception as e:
                logger.error(f"Unexpected error in notification listener: {e}")
                time.sleep(1.0)
        
        logger.info("Notification listener thread stopped")
    
    def _handle_notification(self, channel: str, payload: str):
        """
        Handle incoming notification.
        
        Args:
            channel: Channel name
            payload: Notification payload (JSON string)
        """
        try:
            # Parse JSON payload
            data = json.loads(payload) if payload else {}
            
            logger.debug(f"Received notification on {channel}: {data}")
            
            # Call all handlers for this channel
            with self._lock:
                handlers = self.handlers.get(channel, [])
            
            for handler in handlers:
                try:
                    handler(channel, data)
                except Exception as e:
                    logger.error(f"Error in notification handler for {channel}: {e}")
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON payload on channel {channel}: {payload}")
        except Exception as e:
            logger.error(f"Error handling notification on {channel}: {e}")
    
    def notify(self, channel: str, data: Dict[str, Any]):
        """
        Send a notification to a channel.
        
        Args:
            channel: Channel name
            data: Data to send (will be JSON encoded)
        """
        try:
            if not self.connection:
                self.connect()
            
            payload = json.dumps(data, default=str)
            cursor = self.connection.cursor()
            cursor.execute("SELECT pg_notify(%s, %s)", (channel, payload))
            
            logger.debug(f"Sent notification to {channel}: {data}")
            
        except Exception as e:
            logger.error(f"Failed to send notification to {channel}: {e}")
            raise


# Global notification manager instance
_notification_manager: Optional[NotificationManager] = None


def get_notification_manager() -> NotificationManager:
    """Get the global notification manager instance."""
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager()
    return _notification_manager


def initialize_notifications():
    """Initialize the notification system."""
    manager = get_notification_manager()
    manager.connect()
    logger.info("Notification system initialized")


def shutdown_notifications():
    """Shutdown the notification system."""
    global _notification_manager
    if _notification_manager:
        _notification_manager.stop_listening()
        _notification_manager.disconnect()
        _notification_manager = None
    logger.info("Notification system shutdown")


# High-level convenience functions

def listen_for_signals(handler: Callable[[str, Dict], None]):
    """
    Listen for new trading signals.
    
    Args:
        handler: Function to call when signal notification received
    """
    manager = get_notification_manager()
    manager.add_handler("signals.new", handler)
    if not manager.running:
        manager.start_listening()


def listen_for_orders(handler: Callable[[str, Dict], None]):
    """
    Listen for new orders.
    
    Args:
        handler: Function to call when order notification received
    """
    manager = get_notification_manager()
    manager.add_handler("orders.new", handler)
    if not manager.running:
        manager.start_listening()


def listen_for_watchlist_updates(handler: Callable[[str, Dict], None]):
    """
    Listen for watchlist updates.
    
    Args:
        handler: Function to call when watchlist notification received
    """
    manager = get_notification_manager()
    manager.add_handler("watchlist.update", handler)
    if not manager.running:
        manager.start_listening()


def notify_new_signal(strategy_id: str, symbol: str, signal_type: str, **kwargs):
    """
    Notify about a new trading signal.
    
    Args:
        strategy_id: Strategy that generated the signal
        symbol: Trading symbol
        signal_type: Type of signal
        **kwargs: Additional signal data
    """
    data = {
        "strategy_id": strategy_id,
        "symbol": symbol,
        "signal_type": signal_type,
        "timestamp": time.time(),
        **kwargs
    }
    
    manager = get_notification_manager()
    manager.notify("signals.new", data)


def notify_new_order(order_id: int, symbol: str, side: str, qty: float, **kwargs):
    """
    Notify about a new order.
    
    Args:
        order_id: Order ID
        symbol: Trading symbol
        side: Order side (BUY/SELL)
        qty: Order quantity
        **kwargs: Additional order data
    """
    data = {
        "order_id": order_id,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "timestamp": time.time(),
        **kwargs
    }
    
    manager = get_notification_manager()
    manager.notify("orders.new", data)


def notify_watchlist_update(action: str, symbol: str, **kwargs):
    """
    Notify about watchlist changes.
    
    Args:
        action: Action taken (add/remove)
        symbol: Trading symbol
        **kwargs: Additional data
    """
    data = {
        "action": action,
        "symbol": symbol,
        "timestamp": time.time(),
        **kwargs
    }
    
    manager = get_notification_manager()
    manager.notify("watchlist.update", data)


# Context manager for temporary notifications

@contextmanager
def temporary_listener(channel: str, handler: Callable[[str, Dict], None]):
    """
    Context manager for temporary notification listening.
    
    Args:
        channel: Channel to listen on
        handler: Handler function
    """
    manager = get_notification_manager()
    manager.add_handler(channel, handler)
    
    if not manager.running:
        manager.start_listening()
    
    try:
        yield
    finally:
        manager.remove_handler(channel, handler)


# Async support for modern applications

class AsyncNotificationManager:
    """Async version of notification manager using asyncio."""
    
    def __init__(self, connection_string: Optional[str] = None):
        if connection_string is None:
            settings = get_settings()
            connection_string = settings.database.url
        
        self.connection_string = connection_string
        self.handlers: Dict[str, List[Callable]] = {}
        self.running = False
        self._listen_task: Optional[asyncio.Task] = None
    
    async def start_listening(self):
        """Start async listening loop."""
        if self.running:
            return
        
        self.running = True
        self._listen_task = asyncio.create_task(self._async_listen_loop())
        logger.info("Started async notification listener")
    
    async def stop_listening(self):
        """Stop async listening."""
        self.running = False
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None
        logger.info("Stopped async notification listener")
    
    async def _async_listen_loop(self):
        """Async listening loop."""
        # This would require asyncpg or similar async PostgreSQL driver
        # For now, this is a placeholder for future async implementation
        logger.warning("Async notification manager not fully implemented")
        
        while self.running:
            await asyncio.sleep(1.0)


# Factory function for creating notification managers

def create_notification_manager(async_mode: bool = False) -> Union[NotificationManager, AsyncNotificationManager]:
    """
    Create a notification manager.
    
    Args:
        async_mode: Whether to create async version
        
    Returns:
        Notification manager instance
    """
    if async_mode:
        return AsyncNotificationManager()
    else:
        return NotificationManager()
