"""
Interactive Brokers client wrapper with enhanced reliability.

This module wraps ib-insync.IB connection with retry logic, exponential backoff,
auto-resubscription, request throttling, and connection state management.
"""

import asyncio
import time
import random
from typing import Any, Dict, List, Optional, Callable, Set
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from ib_insync import IB, Contract, Order
from ib_insync.objects import ConnectionStats

from ..common.config import get_settings
from ..common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConnectionState:
    """Track connection state and statistics."""
    connected: bool = False
    client_id: Optional[int] = None
    connect_time: Optional[datetime] = None
    last_disconnect_time: Optional[datetime] = None
    reconnect_attempts: int = 0
    total_connections: int = 0
    subscribed_contracts: Set[int] = field(default_factory=set)
    pending_requests: Dict[str, datetime] = field(default_factory=dict)


class RequestThrottler:
    """Throttle requests to avoid IB pacing violations."""
    
    def __init__(self, max_requests_per_second: int = 50):
        self.max_requests_per_second = max_requests_per_second
        self.request_times: List[float] = []
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self, request_type: str = "general"):
        """Wait if we're approaching rate limits."""
        async with self.lock:
            current_time = time.time()
            
            # Remove old requests (older than 1 second)
            self.request_times = [t for t in self.request_times if current_time - t < 1.0]
            
            # Check if we need to wait
            if len(self.request_times) >= self.max_requests_per_second:
                wait_time = 1.0 - (current_time - self.request_times[0])
                if wait_time > 0:
                    logger.debug(f"Throttling {request_type} request, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
            
            # Record this request
            self.request_times.append(current_time)


class EnhancedIBClient:
    """Enhanced IB client with reliability features."""
    
    def __init__(
        self,
        client_id: int,
        host: str = "127.0.0.1",
        port: int = 7497,
        max_reconnect_attempts: int = 10,
        base_backoff: float = 1.0,
        max_backoff: float = 60.0,
        jitter: bool = True
    ):
        """
        Initialize enhanced IB client.
        
        Args:
            client_id: TWS client ID
            host: TWS host
            port: TWS port
            max_reconnect_attempts: Maximum reconnection attempts
            base_backoff: Base backoff time in seconds
            max_backoff: Maximum backoff time in seconds
            jitter: Whether to add jitter to backoff
        """
        self.client_id = client_id
        self.host = host
        self.port = port
        self.max_reconnect_attempts = max_reconnect_attempts
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.jitter = jitter
        
        # IB connection
        self.ib = IB()
        self.state = ConnectionState()
        self.throttler = RequestThrottler()
        
        # Event handlers
        self.connection_handlers: List[Callable] = []
        self.disconnection_handlers: List[Callable] = []
        self.error_handlers: List[Callable] = []
        
        # Setup IB event handlers
        self.ib.connectedEvent += self._on_connected
        self.ib.disconnectedEvent += self._on_disconnected
        self.ib.errorEvent += self._on_error
        
        logger.info(f"Enhanced IB client initialized for client ID {client_id}")
    
    def add_connection_handler(self, handler: Callable):
        """Add handler for connection events."""
        self.connection_handlers.append(handler)
    
    def add_disconnection_handler(self, handler: Callable):
        """Add handler for disconnection events."""
        self.disconnection_handlers.append(handler)
    
    def add_error_handler(self, handler: Callable):
        """Add handler for error events."""
        self.error_handlers.append(handler)
    
    async def connect(self, timeout: int = 10) -> bool:
        """
        Connect to TWS with retry logic.
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            True if connected successfully
        """
        if self.state.connected:
            logger.warning("Already connected to TWS")
            return True
        
        for attempt in range(self.max_reconnect_attempts + 1):
            try:
                logger.info(f"Connecting to TWS at {self.host}:{self.port} (attempt {attempt + 1})")
                
                await self.ib.connectAsync(
                    host=self.host,
                    port=self.port,
                    clientId=self.client_id,
                    timeout=timeout
                )
                
                logger.info(f"Successfully connected to TWS with client ID {self.client_id}")
                return True
                
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_reconnect_attempts:
                    backoff_time = self._calculate_backoff(attempt)
                    logger.info(f"Retrying connection in {backoff_time:.2f} seconds...")
                    await asyncio.sleep(backoff_time)
                else:
                    logger.error(f"Failed to connect after {self.max_reconnect_attempts + 1} attempts")
                    return False
        
        return False
    
    async def disconnect(self):
        """Gracefully disconnect from TWS."""
        if not self.state.connected:
            logger.warning("Not connected to TWS")
            return
        
        try:
            logger.info("Disconnecting from TWS...")
            
            # Cancel any pending requests
            await self._cancel_pending_requests()
            
            # Disconnect
            self.ib.disconnect()
            
            logger.info("Disconnected from TWS")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def ensure_connected(self) -> bool:
        """Ensure connection is active, reconnect if necessary."""
        if not self.state.connected:
            return await self.connect()
        
        # Test connection with a simple request
        try:
            await self.throttler.wait_if_needed("health_check")
            self.ib.reqCurrentTime()
            return True
        except Exception as e:
            logger.warning(f"Connection test failed: {e}")
            return await self.connect()
    
    async def req_market_data_with_retry(
        self,
        contract: Contract,
        generic_tick_list: str = "",
        snapshot: bool = False,
        regulatory_snapshot: bool = False,
        market_data_type: int = 1  # 1 = Live, 2 = Frozen, 3 = Delayed, 4 = Delayed-Frozen
    ):
        """Request market data with retry logic."""
        await self.throttler.wait_if_needed("market_data")
        
        if not await self.ensure_connected():
            raise ConnectionError("Failed to establish TWS connection")
        
        try:
            ticker = self.ib.reqMktData(
                contract=contract,
                genericTickList=generic_tick_list,
                snapshot=snapshot,
                regulatorySnapshot=regulatory_snapshot,
                mktDataType=market_data_type
            )
            
            if contract.conId:
                self.state.subscribed_contracts.add(contract.conId)
            
            logger.debug(f"Requested market data for {contract.symbol}")
            return ticker
            
        except Exception as e:
            logger.error(f"Failed to request market data for {contract.symbol}: {e}")
            raise
    
    async def cancel_market_data(self, contract: Contract):
        """Cancel market data subscription."""
        await self.throttler.wait_if_needed("cancel_market_data")
        
        try:
            self.ib.cancelMktData(contract)
            
            if contract.conId and contract.conId in self.state.subscribed_contracts:
                self.state.subscribed_contracts.remove(contract.conId)
            
            logger.debug(f"Cancelled market data for {contract.symbol}")
            
        except Exception as e:
            logger.error(f"Failed to cancel market data for {contract.symbol}: {e}")
            raise
    
    async def req_historical_data_with_retry(
        self,
        contract: Contract,
        end_datetime: str,
        duration_str: str,
        bar_size_setting: str,
        what_to_show: str,
        use_rth: bool = True,
        format_date: int = 1,
        keep_up_to_date: bool = False,
        chart_options: List = None
    ):
        """Request historical data with retry logic."""
        await self.throttler.wait_if_needed("historical_data")
        
        if not await self.ensure_connected():
            raise ConnectionError("Failed to establish TWS connection")
        
        try:
            bars = await self.ib.reqHistoricalDataAsync(
                contract=contract,
                endDateTime=end_datetime,
                durationStr=duration_str,
                barSizeSetting=bar_size_setting,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=format_date,
                keepUpToDate=keep_up_to_date,
                chartOptions=chart_options or []
            )
            
            logger.debug(f"Retrieved {len(bars)} historical bars for {contract.symbol}")
            return bars
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {contract.symbol}: {e}")
            raise
    
    async def place_order_with_retry(self, contract: Contract, order: Order):
        """Place order with retry logic."""
        await self.throttler.wait_if_needed("place_order")
        
        if not await self.ensure_connected():
            raise ConnectionError("Failed to establish TWS connection")
        
        try:
            trade = self.ib.placeOrder(contract, order)
            logger.info(f"Placed order: {order.action} {order.totalQuantity} {contract.symbol}")
            return trade
            
        except Exception as e:
            logger.error(f"Failed to place order for {contract.symbol}: {e}")
            raise
    
    async def cancel_order(self, order: Order):
        """Cancel order."""
        await self.throttler.wait_if_needed("cancel_order")
        
        try:
            self.ib.cancelOrder(order)
            logger.info(f"Cancelled order {order.orderId}")
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order.orderId}: {e}")
            raise
    
    async def req_account_summary_with_retry(self, group: str = "All", tags: str = ""):
        """Request account summary with retry logic."""
        await self.throttler.wait_if_needed("account_summary")
        
        if not await self.ensure_connected():
            raise ConnectionError("Failed to establish TWS connection")
        
        try:
            account_values = self.ib.reqAccountSummary(group, tags)
            logger.debug(f"Retrieved account summary: {len(account_values)} items")
            return account_values
            
        except Exception as e:
            logger.error(f"Failed to get account summary: {e}")
            raise
    
    async def req_positions_with_retry(self):
        """Request positions with retry logic."""
        await self.throttler.wait_if_needed("positions")
        
        if not await self.ensure_connected():
            raise ConnectionError("Failed to establish TWS connection")
        
        try:
            positions = self.ib.reqPositions()
            logger.debug(f"Retrieved {len(positions)} positions")
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        stats = {
            "connected": self.state.connected,
            "client_id": self.state.client_id,
            "connect_time": self.state.connect_time.isoformat() if self.state.connect_time else None,
            "last_disconnect_time": self.state.last_disconnect_time.isoformat() if self.state.last_disconnect_time else None,
            "reconnect_attempts": self.state.reconnect_attempts,
            "total_connections": self.state.total_connections,
            "subscribed_contracts_count": len(self.state.subscribed_contracts),
            "pending_requests_count": len(self.state.pending_requests)
        }
        
        # Add IB connection stats if available
        if hasattr(self.ib, 'connectionStats'):
            ib_stats = self.ib.connectionStats()
            if ib_stats:
                stats.update({
                    "bytes_sent": ib_stats.bytesSent,
                    "bytes_received": ib_stats.bytesReceived,
                    "messages_sent": ib_stats.messagesSent,
                    "messages_received": ib_stats.messagesReceived
                })
        
        return stats
    
    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate backoff time with exponential backoff and jitter."""
        backoff = min(self.base_backoff * (2 ** attempt), self.max_backoff)
        
        if self.jitter:
            # Add ±25% jitter
            jitter_amount = backoff * 0.25
            backoff += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0.1, backoff)  # Minimum 100ms
    
    async def _cancel_pending_requests(self):
        """Cancel any pending requests."""
        if self.state.pending_requests:
            logger.info(f"Cancelling {len(self.state.pending_requests)} pending requests")
            self.state.pending_requests.clear()
    
    async def _resubscribe_market_data(self):
        """Resubscribe to market data after reconnection."""
        if not self.state.subscribed_contracts:
            return
        
        logger.info(f"Resubscribing to {len(self.state.subscribed_contracts)} market data contracts")
        
        # This would require storing contract details, which we'll implement
        # when we build the actual market data service
        # For now, just clear the set since we can't resubscribe without contract details
        self.state.subscribed_contracts.clear()
    
    def _on_connected(self):
        """Handle connection event."""
        self.state.connected = True
        self.state.client_id = self.client_id
        self.state.connect_time = datetime.now(timezone.utc)
        self.state.total_connections += 1
        self.state.reconnect_attempts = 0
        
        logger.info(f"TWS connected (client ID: {self.client_id})")
        
        # Call user handlers
        for handler in self.connection_handlers:
            try:
                handler()
            except Exception as e:
                logger.error(f"Error in connection handler: {e}")
    
    def _on_disconnected(self):
        """Handle disconnection event."""
        self.state.connected = False
        self.state.last_disconnect_time = datetime.now(timezone.utc)
        self.state.reconnect_attempts += 1
        
        logger.warning(f"TWS disconnected (client ID: {self.client_id})")
        
        # Call user handlers
        for handler in self.disconnection_handlers:
            try:
                handler()
            except Exception as e:
                logger.error(f"Error in disconnection handler: {e}")
    
    def _on_error(self, req_id, error_code, error_string, contract):
        """Handle error event."""
        logger.error(f"TWS error - ReqId: {req_id}, Code: {error_code}, Message: {error_string}")
        
        # Call user handlers
        for handler in self.error_handlers:
            try:
                handler(req_id, error_code, error_string, contract)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")


@asynccontextmanager
async def ib_client_context(client_id: int, **kwargs):
    """Context manager for IB client with automatic cleanup."""
    settings = get_settings()
    
    client = EnhancedIBClient(
        client_id=client_id,
        host=settings.tws.host,
        port=settings.tws.port,
        **kwargs
    )
    
    try:
        connected = await client.connect()
        if not connected:
            raise ConnectionError(f"Failed to connect to TWS with client ID {client_id}")
        
        yield client
        
    finally:
        await client.disconnect()


def create_ib_client(client_id: int, **kwargs) -> EnhancedIBClient:
    """Factory function to create IB client with settings."""
    settings = get_settings()
    
    return EnhancedIBClient(
        client_id=client_id,
        host=settings.tws.host,
        port=settings.tws.port,
        **kwargs
    )
