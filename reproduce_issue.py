
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque

# Add backend/src to path
if os.path.exists('/app/src'):
    sys.path.append('/app/src')
else:
    sys.path.append('/home/taiconley/Desktop/Projects/trading/backend/src')

# Mock StrategyConfig and BaseStrategy
class StrategyConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    def dict(self):
        return self.__dict__

# Import the strategy class (we need to mock the imports inside it or just copy the relevant parts if it's too complex)
# Since we have the file, let's try to import it directly, mocking dependencies if needed.

# Mocking dependencies
import unittest.mock as mock
import sys

# Stub for BaseStrategy
class BaseStrategy:
    def __init__(self, config):
        self.config = config
    
    def log_info(self, msg):
        pass
        # print(f"[INFO] {msg}")
        
    def log_warning(self, msg):
        print(f"[WARN] {msg}")
        
    def set_state(self, state):
        pass
        
    def create_signal(self, **kwargs):
        # Return a simple object
        return MockSignal(**kwargs)

class MockSignal:
    def __init__(self, **kwargs):
        self.signal_type = kwargs.get('signal_type')
        self.symbol = kwargs.get('symbol')
        self.strength = kwargs.get('strength')
        self.metadata = kwargs

# Stub for strategy decorator
def strategy_decorator(name=None, description=None, default_config=None):
    def decorator(cls):
        return cls
    return decorator

# Setup mocks
mock_base = mock.MagicMock()
mock_base.BaseStrategy = BaseStrategy
mock_base.StrategyConfig = StrategyConfig
mock_base.StrategySignal = mock.MagicMock()
mock_base.SignalType = mock.MagicMock()
mock_base.StrategyState = mock.MagicMock()
sys.modules['strategy_lib.base'] = mock_base

mock_registry = mock.MagicMock()
mock_registry.strategy = strategy_decorator
sys.modules['strategy_lib.registry'] = mock_registry

sys.modules['strategy_lib.pairs_trading_kalman_config'] = mock.MagicMock()
sys.modules['strategy_lib.kalman'] = mock.MagicMock()

# Define a simple KalmanFilter mock
class MockKalmanFilter:
    def __init__(self, delta=1e-4, R=1e-3):
        self.state = np.zeros(2)
        self.covariance = np.eye(2)
    
    def update(self, y1, y2):
        # Simple mock update: return spread=0, beta=1, alpha=0
        return 0.0, 1.0, 0.0
    
    def get_state(self):
        return 1.0, 0.0

sys.modules['strategy_lib.kalman'].KalmanFilter = MockKalmanFilter

# Now import the strategy
from strategy_lib.pairs_trade_adaptive_kalman import PairsTradingKalmanStrategy, PairsTradingAggregatedConfig

# Setup the strategy with user parameters
config = PairsTradingAggregatedConfig(
    strategy_id="test_strat",
    pairs=[["A", "B"]],
    bar_timeframe="5 secs",
    lookback_window=30,
    entry_threshold=1.8,
    exit_threshold=1.0,
    stats_aggregation_seconds=1800, # 30 mins
    cooldown_bars=180,
    cooldown_after_all_exits=False,
    stop_loss_zscore=4.0,
    use_kalman=False,
    market_close_hour=23, # Extend for testing
    timezone="UTC"
)

strategy = PairsTradingKalmanStrategy(config)

# Simulate data
# We need to simulate "noise" on 5s bars vs "smooth" aggregated stats
# 30 min aggregation = 360 bars of 5s

print(f"Stats Aggregation Bars: {strategy.stats_aggregation_bars}")
print(f"Config Cooldown After All Exits: {getattr(strategy.config, 'cooldown_after_all_exits', 'NOT FOUND')}")

# Create synthetic data
# Price A: Random walk
# Price B: Price A + noise
np.random.seed(42)
n_bars = 2000
prices_a = 100 + np.cumsum(np.random.randn(n_bars) * 0.1)
prices_b = prices_a + np.random.randn(n_bars) * 0.5 # High noise

dates = [datetime(2023, 1, 1, 9, 30) + timedelta(seconds=5*i) for i in range(n_bars)]

df_a = pd.DataFrame({'timestamp': dates, 'close': prices_a})
df_b = pd.DataFrame({'timestamp': dates, 'close': prices_b})

import asyncio

async def run_simulation():
    print("Running simulation...")
    await strategy.on_start({})
    
    # Pre-populate history to bypass warmup
    pair_state = strategy._pair_states["A/B"]
    # Fill spread history with 0s (mean)
    for _ in range(config.lookback_window + 10):
        pair_state['spread_history'].append(0.0)
    # Fill price history for hedge ratio with random walk to satisfy coint test
    p_a, p_b = 100.0, 100.0
    for _ in range(config.min_hedge_lookback + 10):
        p_a += np.random.randn() * 0.1
        p_b += np.random.randn() * 0.1
        pair_state['price_history_a'].append(p_a)
        pair_state['price_history_b'].append(p_b)
    
    # Set hedge ratio manually
    pair_state['hedge_ratio'] = 1.0
    pair_state['hedge_intercept'] = 0.0
    
    trade_entries = 0
    trade_exits = 0
    positions = []
    
    print("Starting loop...")
    for i in range(n_bars):
        # Manipulate prices to force signals
        # Noise level: 0.001 (0.1%)
        
        base_price = 100.0
        if i < 10:
            p_a = base_price
            p_b = base_price
        elif i < 20:
            # Entry trigger: Spread = log(100.3) - log(100) ~= 0.003
            # Std = 0.001. Z = 3.0. Entry > 1.8.
            p_a = base_price + 0.3
            p_b = base_price
        elif i < 30:
            # Mean Reversion trigger: Z drops to 0.
            # p_a = 100.0. Spread = 0.
            p_a = base_price
            p_b = base_price
        elif i < 80:
            # Cooldown period.
            # Return to entry range: p_a = 100.3 (Z=3.0).
            # Should NOT enter.
            p_a = base_price + 0.3
            p_b = base_price
        else:
            p_a = base_price
            p_b = base_price
            
        # Update history with some noise to ensure std > 0
        if i == 0:
            # Inject some noise into history to get non-zero std
            for k in range(30):
                pair_state['spread_history'].append(np.random.randn() * 0.001)
        
        # Construct bars
        bar_a = pd.DataFrame({'timestamp': [dates[i]], 'close': [p_a]})
        bar_b = pd.DataFrame({'timestamp': [dates[i]], 'close': [p_b]})
        bars_data = {'A': bar_a, 'B': bar_b}
        
        signals = await strategy.on_bar_multi(['A', 'B'], "5 secs", bars_data)
        
        pair_state = strategy._pair_states["A/B"]
        
        # Track position changes
        if i > 0:
            prev_pos = positions[-1]
            curr_pos = pair_state['current_position']
            if prev_pos == "flat" and curr_pos != "flat":
                trade_entries += 1
                print(f"Bar {i}: ENTRY. Z-score: {pair_state.get('last_zscore', 0):.2f}")
            elif prev_pos != "flat" and curr_pos == "flat":
                trade_exits += 1
                exit_reason = signals[0].metadata.get('exit_reason') if signals else 'unknown'
                print(f"Bar {i}: EXIT. Reason: {exit_reason}")
                print(f"  Cooldown remaining: {pair_state['cooldown_remaining']}")
            
            # Check if we entered during cooldown
            if pair_state['cooldown_remaining'] > 0 and curr_pos != "flat" and prev_pos == "flat":
                print(f"!!! ERROR: Entered trade with cooldown remaining: {pair_state['cooldown_remaining']}")
        
        positions.append(pair_state['current_position'])

    print(f"Total Entries: {trade_entries}")
    print(f"Total Exits: {trade_exits}")

asyncio.run(run_simulation())
