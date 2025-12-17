"""
Configuration management using Pydantic settings.

This module provides typed configuration classes that load from environment variables
with validation, defaults, and feature flags for the trading bot system.
"""

import os
from typing import List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database connection configuration."""
    
    user: str = Field(default="bot", env="POSTGRES_USER")
    password: str = Field(default="botpw", env="POSTGRES_PASSWORD") 
    db: str = Field(default="trading", env="POSTGRES_DB")
    host: str = Field(default="postgres", env="POSTGRES_HOST")
    port: int = Field(default=5432, env="POSTGRES_PORT")
    
    @property
    def url(self) -> str:
        """Get SQLAlchemy database URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"
    
    @property
    def async_url(self) -> str:
        """Get async SQLAlchemy database URL."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


class TWSSettings(BaseSettings):
    """Interactive Brokers TWS connection configuration."""
    
    host: str = Field(default="172.17.0.1", env="TWS_HOST")
    port: int = Field(default=7497, env="TWS_PORT")
    use_paper: bool = Field(default=True, env="USE_PAPER")
    enable_live: bool = Field(default=False, env="ENABLE_LIVE")
    dry_run: bool = Field(default=True, env="DRY_RUN")
    client_id_base: int = Field(default=10, env="TWS_CLIENT_ID_BASE")
    reconnect_backoff_min: int = Field(default=1, env="RECONNECT_BACKOFF_MIN")
    reconnect_backoff_max: int = Field(default=30, env="RECONNECT_BACKOFF_MAX")
    
    @validator("port")
    def validate_port(cls, v, values):
        """Validate TWS port matches trading mode."""
        use_paper = values.get("use_paper", True)
        if use_paper and v != 7497:
            raise ValueError("Paper trading requires TWS_PORT=7497")
        elif not use_paper and v != 7496:
            raise ValueError("Live trading requires TWS_PORT=7496")
        return v
    
    @property
    def is_live_mode(self) -> bool:
        """Check if configured for live trading."""
        return (
            self.enable_live and 
            not self.use_paper and 
            not self.dry_run and 
            self.port == 7496
        )


class MarketDataSettings(BaseSettings):
    """Market data configuration."""
    
    default_symbols: str = Field(default="AAPL,MSFT,SPY", env="DEFAULT_SYMBOLS")
    max_subscriptions: int = Field(default=50, env="MAX_SUBSCRIPTIONS")
    bar_size: str = Field(default="1 min", env="BAR_SIZE")
    what_to_show: str = Field(default="TRADES", env="WHAT_TO_SHOW")
    rth: bool = Field(default=True, env="RTH")
    lookback: str = Field(default="30 D", env="LOOKBACK")
    
    @property
    def symbols_list(self) -> List[str]:
        """Get list of default symbols."""
        return [s.strip().upper() for s in self.default_symbols.split(",")]
    
    @validator("max_subscriptions")
    def validate_max_subscriptions(cls, v):
        """Validate subscription limit."""
        if v <= 0 or v > 1000:
            raise ValueError("MAX_SUBSCRIPTIONS must be between 1 and 1000")
        return v


class HistoricalDataSettings(BaseSettings):
    """Historical data configuration."""
    
    model_config = SettingsConfigDict(env_file=None, extra='ignore', populate_by_name=True)
    
    max_requests_per_min: int = Field(default=30, validation_alias="MAX_HIST_REQUESTS_PER_MIN")
    bar_sizes: str = Field(default="1 min,5 mins,1 day", validation_alias="HIST_BAR_SIZES")
    
    @property
    def bar_sizes_list(self) -> List[str]:
        """Get list of supported bar sizes."""
        return [s.strip() for s in self.bar_sizes.split(",")]
    
    @validator("max_requests_per_min")
    def validate_request_rate(cls, v):
        """Validate request rate limit."""
        if v <= 0 or v > 100:
            raise ValueError("MAX_HIST_REQUESTS_PER_MIN must be between 1 and 100")
        return v


class MarketHoursSettings(BaseSettings):
    """Market hours filtering configuration."""
    
    enabled: bool = Field(default=True, env="MARKET_HOURS_FILTER_ENABLED")
    timezone: str = Field(default="America/New_York", env="MARKET_HOURS_TIMEZONE")
    market_open_hour: int = Field(default=9, env="MARKET_OPEN_HOUR")
    market_open_minute: int = Field(default=30, env="MARKET_OPEN_MINUTE")
    market_close_hour: int = Field(default=16, env="MARKET_CLOSE_HOUR")
    market_close_minute: int = Field(default=0, env="MARKET_CLOSE_MINUTE")
    
    @validator("market_open_hour", "market_close_hour")
    def validate_hour(cls, v):
        if not 0 <= v <= 23:
            raise ValueError("Hour must be between 0 and 23")
        return v
    
    @validator("market_open_minute", "market_close_minute")
    def validate_minute(cls, v):
        if not 0 <= v <= 59:
            raise ValueError("Minute must be between 0 and 59")
        return v


class BacktestSettings(BaseSettings):
    """Backtest simulation configuration."""
    
    comm_per_share: float = Field(default=0.005, env="BT_COMM_PER_SHARE")
    min_comm_per_order: float = Field(default=1.0, env="BT_MIN_COMM_PER_ORDER")
    default_slippage_ticks: int = Field(default=1, env="BT_DEFAULT_SLIPPAGE_TICKS")
    tick_size_us_equity: float = Field(default=0.01, env="BT_TICK_SIZE_US_EQUITY")
    
    @validator("comm_per_share", "min_comm_per_order")
    def validate_positive_commission(cls, v):
        """Validate commission values are positive."""
        if v < 0:
            raise ValueError("Commission values must be non-negative")
        return v


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(default="json", env="LOG_FORMAT")
    
    @validator("level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of: {valid_levels}")
        return v.upper()


class DevelopmentSettings(BaseSettings):
    """Development and debugging configuration."""
    
    debug: bool = Field(default=False, env="DEBUG")
    development_mode: bool = Field(default=False, env="DEVELOPMENT_MODE")


class TradingBotSettings(BaseSettings):
    """Main configuration class that combines all settings."""
    
    # Service identification
    service_name: Optional[str] = Field(default=None, env="SERVICE_NAME")
    
    # Sub-configurations
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    tws: TWSSettings = Field(default_factory=TWSSettings)
    market_data: MarketDataSettings = Field(default_factory=MarketDataSettings)
    market_hours: MarketHoursSettings = Field(default_factory=MarketHoursSettings)
    historical: HistoricalDataSettings = Field(default_factory=HistoricalDataSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    development: DevelopmentSettings = Field(default_factory=DevelopmentSettings)
    
    class Config:
        case_sensitive = False
        
    def validate_live_trading_safety(self) -> tuple[bool, List[str]]:
        """
        Validate all safety conditions for live trading.
        
        Returns:
            tuple: (is_safe, list_of_violations)
        """
        violations = []
        
        if not self.tws.enable_live:
            violations.append("ENABLE_LIVE must be 1 for live trading")
            
        if self.tws.use_paper:
            violations.append("USE_PAPER must be 0 for live trading")
            
        if self.tws.port != 7496:
            violations.append("TWS_PORT must be 7496 for live trading")
            
        if self.tws.dry_run:
            violations.append("DRY_RUN must be 0 for live trading")
            
        # TODO: Add check for risk_limits.block_live_trading_until when database is available
        
        return len(violations) == 0, violations
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.development.debug and not self.development.development_mode


# Global settings instance
settings = TradingBotSettings()


def get_settings() -> TradingBotSettings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> TradingBotSettings:
    """Reload settings from environment (useful for testing)."""
    global settings
    settings = TradingBotSettings()
    return settings


# Client ID allocation helper functions
def get_client_id_for_service(service_name: str) -> int:
    """
    Get the designated client ID for a service.
    
    Args:
        service_name: Name of the service
        
    Returns:
        Client ID for the service
        
    Raises:
        ValueError: If service name is not recognized
    """
    base = settings.tws.client_id_base
    
    # NOTE: trader MUST use client ID 0 to receive all order events via reqAutoOpenOrders
    if service_name == "trader":
        return 0
    
    service_offsets = {
        "account": 1,      # Client ID 11
        "marketdata": 2,   # Client ID 12  
        "historical": 3,   # Client ID 13
        "strategy": 5,     # Client IDs 15-29 (base + 5 + instance_id)
    }
    
    if service_name not in service_offsets:
        raise ValueError(f"Unknown service: {service_name}")
    
    return base + service_offsets[service_name]


def get_strategy_client_id(instance_id: int = 0) -> int:
    """
    Get client ID for a strategy instance.
    
    Args:
        instance_id: Strategy instance number (0-14)
        
    Returns:
        Client ID for the strategy instance
        
    Raises:
        ValueError: If instance_id is out of range
    """
    if not 0 <= instance_id <= 14:
        raise ValueError("Strategy instance_id must be between 0 and 14")
    
    return get_client_id_for_service("strategy") + instance_id
