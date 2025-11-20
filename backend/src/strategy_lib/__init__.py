"""
Strategy Library

This package provides the strategy interface and implementations for the trading bot.
It includes the base strategy class, registry for dynamic loading, and strategy implementations.
"""

from .base import (
    BaseStrategy, 
    StrategyConfig, 
    StrategySignal, 
    StrategyState,
    StrategyMetrics,
    SignalStrength,
    validate_bars_dataframe,
    calculate_position_size,
    format_currency,
    format_percentage
)

# Import SignalType from common.schemas
from common.schemas import SignalType

from .registry import (
    StrategyRegistry,
    StrategyInfo,
    get_strategy_registry,
    register_strategy,
    load_strategies_from_directory,
    create_strategy_from_db_config,
    strategy  # decorator
)

# Import strategies to register them (they auto-register via decorators)
from . import sma_cross
from . import mean_revert
from . import pairs_trade
from . import pairs_trade_adaptive
from . import pairs_trade_adaptive_aggregated_stats
from . import pairs_trade_adaptive_kalman

__all__ = [
    # Base classes and types
    'BaseStrategy',
    'StrategyConfig', 
    'StrategySignal',
    'StrategyState',
    'StrategyMetrics',
    'SignalStrength',
    'SignalType',  # Added SignalType
    
    # Registry classes and functions
    'StrategyRegistry',
    'StrategyInfo',
    'get_strategy_registry',
    'register_strategy',
    'load_strategies_from_directory',
    'create_strategy_from_db_config',
    'strategy',
    
    # Strategy implementations (available via registry)
    # 'SMAStrategy', 'MeanReversionStrategy' - use registry instead
    
    # Utility functions
    'validate_bars_dataframe',
    'calculate_position_size',
    'format_currency',
    'format_percentage'
]
