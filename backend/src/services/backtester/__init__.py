"""
Backtester Service

On-demand backtesting service with both CLI and REST API interfaces.
Simulates strategy execution on historical data with realistic fills,
commission, and slippage.
"""

from .engine import (
    BacktestEngine,
    BacktestMetrics,
    BacktestOrder,
    BacktestPosition,
    BacktestTrade,
    OrderStatus
)

from .main import BacktesterService

__all__ = [
    'BacktestEngine',
    'BacktestMetrics',
    'BacktestOrder',
    'BacktestPosition',
    'BacktestTrade',
    'OrderStatus',
    'BacktesterService'
]

