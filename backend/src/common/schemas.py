"""
Pydantic schemas for API requests/responses and data validation.

This module defines all the data structures used for API communication,
validation, and type safety across the trading bot services.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, validator
from pydantic.types import conint, confloat, constr


# =============================================================================
# Enums and Constants
# =============================================================================

class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP-LMT"


class TimeInForce(str, Enum):
    """Time in force enumeration."""
    DAY = "DAY"
    GTC = "GTC"  # Good Till Canceled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING_SUBMIT = "PendingSubmit"
    PENDING_CANCEL = "PendingCancel"
    PRE_SUBMITTED = "PreSubmitted"
    SUBMITTED = "Submitted"
    CANCELLED = "Cancelled"
    FILLED = "Filled"
    INACTIVE = "Inactive"


class SignalType(str, Enum):
    """Signal type enumeration."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT = "EXIT"


class HealthStatus(str, Enum):
    """Service health status enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"


# =============================================================================
# Base Schemas
# =============================================================================

class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    
    class Config:
        # Allow population by field name or alias
        populate_by_name = True
        # Use enum values rather than names
        use_enum_values = True
        # Validate assignment
        validate_assignment = True


class TimestampedSchema(BaseSchema):
    """Base schema with timestamp fields."""
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


# =============================================================================
# Symbol and Market Data Schemas
# =============================================================================

class SymbolSchema(BaseSchema):
    """Symbol information schema."""
    symbol: constr(min_length=1, max_length=20) = Field(..., description="Trading symbol")
    conid: Optional[int] = Field(None, description="IB contract ID")
    primary_exchange: Optional[str] = Field(None, description="Primary exchange")
    currency: str = Field(default="USD", description="Currency")
    active: bool = Field(default=True, description="Whether symbol is active")


class TickSchema(BaseSchema):
    """Market tick data schema."""
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Tick timestamp")
    bid: Optional[Decimal] = Field(None, description="Bid price")
    ask: Optional[Decimal] = Field(None, description="Ask price")
    last: Optional[Decimal] = Field(None, description="Last trade price")
    bid_size: Optional[int] = Field(None, description="Bid size")
    ask_size: Optional[int] = Field(None, description="Ask size")
    last_size: Optional[int] = Field(None, description="Last trade size")


class CandleSchema(BaseSchema):
    """Candle/bar data schema."""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., alias="tf", description="Timeframe (e.g., '1 min', '1 day')")
    timestamp: datetime = Field(..., alias="ts", description="Bar timestamp")
    open: Decimal = Field(..., description="Open price")
    high: Decimal = Field(..., description="High price")
    low: Decimal = Field(..., description="Low price")
    close: Decimal = Field(..., description="Close price")
    volume: int = Field(default=0, description="Volume")
    
    @validator('high')
    def high_must_be_highest(cls, v, values):
        """Validate that high is the highest price."""
        if 'low' in values and v < values['low']:
            raise ValueError('High must be >= low')
        if 'open' in values and v < values['open']:
            raise ValueError('High must be >= open')
        if 'close' in values and v < values['close']:
            raise ValueError('High must be >= close')
        return v
    
    @validator('low')
    def low_must_be_lowest(cls, v, values):
        """Validate that low is the lowest price."""
        if 'high' in values and v > values['high']:
            raise ValueError('Low must be <= high')
        if 'open' in values and v > values['open']:
            raise ValueError('Low must be <= open')
        if 'close' in values and v > values['close']:
            raise ValueError('Low must be <= close')
        return v


# =============================================================================
# Trading Schemas
# =============================================================================

class OrderRequest(BaseSchema):
    """Order placement request schema."""
    symbol: constr(min_length=1, max_length=20) = Field(..., description="Trading symbol")
    side: OrderSide = Field(..., description="Order side")
    quantity: confloat(gt=0) = Field(..., alias="qty", description="Order quantity")
    order_type: OrderType = Field(..., description="Order type")
    limit_price: Optional[confloat(gt=0)] = Field(None, description="Limit price")
    stop_price: Optional[confloat(gt=0)] = Field(None, description="Stop price")
    time_in_force: TimeInForce = Field(default=TimeInForce.DAY, alias="tif", description="Time in force")
    strategy_id: Optional[str] = Field(None, description="Strategy ID")
    
    @validator('limit_price')
    def limit_price_required_for_limit_orders(cls, v, values):
        """Validate limit price for limit orders."""
        if values.get('order_type') in [OrderType.LIMIT, OrderType.STOP_LIMIT] and v is None:
            raise ValueError('Limit price required for limit orders')
        return v
    
    @validator('stop_price')
    def stop_price_required_for_stop_orders(cls, v, values):
        """Validate stop price for stop orders."""
        if values.get('order_type') in [OrderType.STOP, OrderType.STOP_LIMIT] and v is None:
            raise ValueError('Stop price required for stop orders')
        return v


class OrderResponse(TimestampedSchema):
    """Order response schema."""
    id: int = Field(..., description="Order ID")
    symbol: str = Field(..., description="Trading symbol")
    side: OrderSide = Field(..., description="Order side")
    quantity: Decimal = Field(..., alias="qty", description="Order quantity")
    order_type: OrderType = Field(..., description="Order type")
    limit_price: Optional[Decimal] = Field(None, description="Limit price")
    stop_price: Optional[Decimal] = Field(None, description="Stop price")
    time_in_force: TimeInForce = Field(..., alias="tif", description="Time in force")
    status: OrderStatus = Field(..., description="Order status")
    external_order_id: Optional[str] = Field(None, description="External order ID")
    strategy_id: Optional[str] = Field(None, description="Strategy ID")
    account_id: str = Field(..., description="Account ID")
    placed_at: datetime = Field(..., description="Order placement timestamp")


class ExecutionSchema(BaseSchema):
    """Execution schema."""
    id: int = Field(..., description="Execution ID")
    order_id: int = Field(..., description="Order ID")
    trade_id: Optional[str] = Field(None, description="Trade ID")
    symbol: str = Field(..., description="Trading symbol")
    quantity: Decimal = Field(..., alias="qty", description="Executed quantity")
    price: Decimal = Field(..., description="Execution price")
    timestamp: datetime = Field(..., alias="ts", description="Execution timestamp")


# =============================================================================
# Account and Portfolio Schemas
# =============================================================================

class AccountSummarySchema(BaseSchema):
    """Account summary schema."""
    account_id: str = Field(..., description="Account ID")
    tag: str = Field(..., description="Summary tag")
    value: str = Field(..., description="Summary value")
    currency: str = Field(..., description="Currency")
    timestamp: datetime = Field(..., alias="ts", description="Update timestamp")


class PositionSchema(BaseSchema):
    """Position schema."""
    account_id: str = Field(..., description="Account ID")
    symbol: str = Field(..., description="Trading symbol")
    quantity: Decimal = Field(..., alias="qty", description="Position quantity")
    average_price: Decimal = Field(..., alias="avg_price", description="Average price")
    market_value: Optional[Decimal] = Field(None, description="Current market value")
    unrealized_pnl: Optional[Decimal] = Field(None, description="Unrealized P&L")
    timestamp: datetime = Field(..., alias="ts", description="Update timestamp")


# =============================================================================
# Strategy and Signal Schemas
# =============================================================================

class StrategySchema(TimestampedSchema):
    """Strategy schema."""
    strategy_id: str = Field(..., description="Strategy ID")
    name: str = Field(..., description="Strategy name")
    enabled: bool = Field(..., description="Whether strategy is enabled")
    parameters: Optional[Dict[str, Any]] = Field(None, alias="params_json", description="Strategy parameters")


class StrategyUpdateRequest(BaseSchema):
    """Strategy update request schema."""
    enabled: Optional[bool] = Field(None, description="Enable/disable strategy")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Strategy parameters")


class SignalSchema(BaseSchema):
    """Signal schema."""
    strategy_id: str = Field(..., description="Strategy ID")
    symbol: str = Field(..., description="Trading symbol")
    signal_type: SignalType = Field(..., description="Signal type")
    strength: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Signal strength (0-1)")
    timestamp: datetime = Field(..., alias="ts", description="Signal timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, alias="meta_json", description="Additional metadata")


# =============================================================================
# Backtest Schemas
# =============================================================================

class BacktestRequest(BaseSchema):
    """Backtest request schema."""
    strategy_name: str = Field(..., description="Strategy name")
    symbols: List[str] = Field(..., description="List of symbols to backtest")
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Strategy parameters")
    initial_capital: Optional[Decimal] = Field(Decimal("100000"), description="Initial capital")


class BacktestResult(TimestampedSchema):
    """Backtest result schema."""
    id: int = Field(..., description="Backtest run ID")
    strategy_name: str = Field(..., description="Strategy name")
    start_date: datetime = Field(..., alias="start_ts", description="Backtest start date")
    end_date: datetime = Field(..., alias="end_ts", description="Backtest end date")
    total_pnl: Decimal = Field(..., alias="pnl", description="Total P&L")
    sharpe_ratio: Optional[Decimal] = Field(None, alias="sharpe", description="Sharpe ratio")
    max_drawdown: Optional[Decimal] = Field(None, alias="maxdd", description="Maximum drawdown")
    total_trades: int = Field(..., alias="trades", description="Total number of trades")
    parameters: Optional[Dict[str, Any]] = Field(None, alias="params_json", description="Strategy parameters")


class BacktestTradeSchema(BaseSchema):
    """Backtest trade schema."""
    symbol: str = Field(..., description="Trading symbol")
    side: OrderSide = Field(..., description="Trade side")
    quantity: Decimal = Field(..., alias="qty", description="Trade quantity")
    entry_timestamp: datetime = Field(..., alias="entry_ts", description="Entry timestamp")
    entry_price: Decimal = Field(..., alias="entry_px", description="Entry price")
    exit_timestamp: Optional[datetime] = Field(None, alias="exit_ts", description="Exit timestamp")
    exit_price: Optional[Decimal] = Field(None, alias="exit_px", description="Exit price")
    pnl: Optional[Decimal] = Field(None, description="Trade P&L")


# =============================================================================
# System and Health Schemas
# =============================================================================

class HealthCheckResponse(BaseSchema):
    """Health check response schema."""
    status: HealthStatus = Field(..., description="Service health status")
    service: str = Field(..., description="Service name")
    timestamp: datetime = Field(..., description="Health check timestamp")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")


class ServiceStatusSchema(BaseSchema):
    """Service status schema."""
    service: str = Field(..., description="Service name")
    status: HealthStatus = Field(..., description="Service status")
    last_updated: datetime = Field(..., description="Last update timestamp")


class WatchlistEntry(BaseSchema):
    """Watchlist entry schema."""
    symbol: str = Field(..., description="Trading symbol")
    added_at: datetime = Field(..., description="Date added to watchlist")


class WatchlistUpdateRequest(BaseSchema):
    """Watchlist update request schema."""
    action: str = Field(..., description="Action (add/remove)")
    symbol: str = Field(..., description="Trading symbol")
    
    @validator('action')
    def validate_action(cls, v):
        """Validate action value."""
        if v not in ['add', 'remove']:
            raise ValueError('Action must be "add" or "remove"')
        return v


# =============================================================================
# API Response Wrappers
# =============================================================================

class APIResponse(BaseSchema):
    """Generic API response wrapper."""
    success: bool = Field(..., description="Whether request was successful")
    message: Optional[str] = Field(None, description="Response message")
    data: Optional[Any] = Field(None, description="Response data")
    errors: Optional[List[str]] = Field(None, description="Error messages")


class PaginatedResponse(BaseSchema):
    """Paginated response schema."""
    items: List[Any] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_prev: bool = Field(..., description="Whether there are previous pages")


# =============================================================================
# Utility Functions
# =============================================================================

def create_success_response(data: Any = None, message: str = "Success") -> APIResponse:
    """Create a successful API response."""
    return APIResponse(success=True, message=message, data=data)


def create_error_response(message: str, errors: Optional[List[str]] = None) -> APIResponse:
    """Create an error API response."""
    return APIResponse(success=False, message=message, errors=errors or [])


def validate_symbol_format(symbol: str) -> str:
    """Validate and normalize symbol format."""
    if not symbol or not symbol.strip():
        raise ValueError("Symbol cannot be empty")
    
    normalized = symbol.strip().upper()
    if len(normalized) > 20:
        raise ValueError("Symbol too long (max 20 characters)")
    
    return normalized


def validate_price(price: Union[float, Decimal], allow_zero: bool = False) -> Decimal:
    """Validate price value."""
    if isinstance(price, float):
        price = Decimal(str(price))
    
    if not allow_zero and price <= 0:
        raise ValueError("Price must be positive")
    elif allow_zero and price < 0:
        raise ValueError("Price cannot be negative")
    
    return price


def validate_quantity(quantity: Union[float, Decimal]) -> Decimal:
    """Validate quantity value."""
    if isinstance(quantity, float):
        quantity = Decimal(str(quantity))
    
    if quantity <= 0:
        raise ValueError("Quantity must be positive")
    
    return quantity
