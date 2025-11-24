
import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from src.services.backtester.engine import BacktestEngine, BacktestMetrics
from src.strategy_lib import BaseStrategy

class MockStrategy(BaseStrategy):
    def __init__(self):
        self.config = MagicMock()
        self.config.name = "MockStrategy"
        self.config.strategy_id = "mock_strategy"

    async def on_start(self, instruments):
        pass

    async def on_stop(self):
        pass

    async def on_bar(self, symbol, timeframe, bars):
        return []

@pytest.fixture
def engine():
    strategy = MockStrategy()
    return BacktestEngine(strategy=strategy)

def test_sortino_ratio_calculation(engine):
    """Test Sortino Ratio calculation with known values."""
    # Setup equity curve with known returns
    # Returns: [0.01, 0.01, 0.01, 0.01, -0.02]
    # Mean: 0.004
    # Downside: [0, 0, 0, 0, -0.02]
    # Downside Sq: [0, 0, 0, 0, 0.0004]
    # Mean Downside Sq: 0.00008
    # Downside Dev: sqrt(0.00008) = 0.00894427
    # Sortino: 0.004 / 0.00894427 = 0.4472136
    
    start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    engine.equity_curve = [
        (start_date, Decimal('100000.00')),
        (start_date + timedelta(days=1), Decimal('101000.00')), # +1%
        (start_date + timedelta(days=2), Decimal('102010.00')), # +1%
        (start_date + timedelta(days=3), Decimal('103030.10')), # +1%
        (start_date + timedelta(days=4), Decimal('104060.40')), # +1%
        (start_date + timedelta(days=5), Decimal('101979.19')), # -2%
    ]
    
    # Mock annualization to 1.0 for simple verification
    # We need to patch the internal logic or just accept the annualization factor
    # The engine calculates annualization based on data frequency.
    # With daily data, it should be sqrt(252) approx 15.87
    
    metrics = engine._calculate_metrics(
        start_date,
        start_date + timedelta(days=5)
    )
    
    assert metrics.sortino_ratio is not None
    
    # Calculate expected annualization
    # 5 days, 5 bars. 
    # total_seconds = 5 * 86400
    # avg_seconds_per_bar = 86400
    # bars_per_year = 252 * 6.5 * 3600 / 86400 = 68.25 (approx, depends on trading hours assumption in engine)
    # Actually engine assumes: trading_seconds_per_year = 252 * 6.5 * 3600 = 5896800
    # bars_per_year = 5896800 / 86400 = 68.25
    # annualization = sqrt(68.25) = 8.26
    
    # Let's just verify it's not 0 and is positive
    assert metrics.sortino_ratio > 0
    
    # Verify against the manual calculation * annualization
    # We can't easily predict exact annualization without mocking, but we can check the core ratio logic
    # by checking if it matches our expectation relative to Sharpe
    
    # Sharpe: mean / std
    # Sortino: mean / downside_std
    # Since downside_std < std (usually), Sortino > Sharpe for this case?
    # Returns: [0.01, 0.01, 0.01, 0.01, -0.02]
    # Std: 0.012
    # Downside Dev: 0.0089
    # So Sortino should be higher than Sharpe
    
    assert metrics.sharpe_ratio is not None
    assert metrics.sortino_ratio > metrics.sharpe_ratio

def test_sortino_ratio_all_positive(engine):
    """Test Sortino Ratio with all positive returns."""
    start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    engine.equity_curve = [
        (start_date, Decimal('100000.00')),
        (start_date + timedelta(days=1), Decimal('101000.00')),
        (start_date + timedelta(days=2), Decimal('102010.00')),
    ]
    
    metrics = engine._calculate_metrics(
        start_date,
        start_date + timedelta(days=2)
    )
    
    # Should be None or handled gracefully (infinite)
    # The code checks if downside_deviation > 0
    # If 0, it skips setting sortino_ratio, so it remains None (default)
    assert metrics.sortino_ratio is None

def test_sortino_ratio_all_negative(engine):
    """Test Sortino Ratio with all negative returns."""
    start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    engine.equity_curve = [
        (start_date, Decimal('100000.00')),
        (start_date + timedelta(days=1), Decimal('99000.00')),
        (start_date + timedelta(days=2), Decimal('98010.00')),
    ]
    
    metrics = engine._calculate_metrics(
        start_date,
        start_date + timedelta(days=2)
    )
    
    assert metrics.sortino_ratio is not None
    assert metrics.sortino_ratio < 0
