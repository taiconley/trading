
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from decimal import Decimal
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategy_lib.pairs_trade_adaptive_kalman import PairsTradingKalmanStrategy, PairsTradingAggregatedConfig
from src.strategy_lib.base import StrategyConfig

async def reproduce_bug():
    print("🧪 Reproducing Cooldown Bug...")

    # Config with cooldown
    config = PairsTradingAggregatedConfig(
        strategy_id="test_bug",
        name="Test Bug",
        symbols=["A", "B"],
        pairs=[["A", "B"]],
        lookback_window=20,
        stop_loss_zscore=2.0,
        cooldown_bars=10,
        entry_threshold=1.5,
        exit_threshold=0.5,
        position_size=10,
        max_pair_loss_pct=1.0,
        volatility_stop_multiplier=100.0,
        max_halflife_bars=10000,
        require_half_life=False,
        use_kalman=False, # Simple mode for easier setup
        stationarity_checks_enabled=False,
        volatility_adaptation_enabled=False
    )

    strategy = PairsTradingKalmanStrategy(config)
    instruments = {"A": {}, "B": {}}
    await strategy.on_start(instruments)

    # Manually set up state to be in a position
    pair_key = "A/B"
    pair_state = strategy._pair_states[pair_key]
    pair_state['current_position'] = "short_a_long_b"
    pair_state['entry_zscore'] = 1.6
    pair_state['entry_quantities'] = {"A": -10, "B": 10}
    pair_state['entry_prices'] = {"A": 100.0, "B": 100.0}
    pair_state['hedge_ratio'] = 1.0
    pair_state['hedge_intercept'] = 0.0
    pair_state['half_life'] = 100.0
    # Pre-fill history to avoid warmup checks
    for _ in range(30):
        pair_state['spread_history'].append(0.0)
        pair_state['price_history_a'].append(100.0)
        pair_state['price_history_b'].append(100.0)

    print(f"Initial Position: {pair_state['current_position']}")
    print(f"Initial Cooldown: {pair_state.get('cooldown_remaining', 0)}")

    # Create bars that trigger stop loss
    # Stop loss is 2.0. We need zscore > 2.0.
    # We want spread to be high. Increase A.
    # Let's set A=105, B=100 => log(1.05) approx 0.048.
    # With mean=0, std=0.01 (artificially set below), zscore ~ 4.8

    pair_state['spread_history'].clear()
    for i in range(20):
        val = 0.01 * (1 if i % 2 == 0 else -1)
        pair_state['spread_history'].append(val)
    
    bars_a = pd.DataFrame([{
        'timestamp': datetime.now(timezone.utc),
        'close': 105.0,
        'open': 105.0, 'high': 105.0, 'low': 105.0, 'volume': 1000
    }])
    bars_b = pd.DataFrame([{
        'timestamp': datetime.now(timezone.utc),
        'close': 100.0,
        'open': 100.0, 'high': 100.0, 'low': 100.0, 'volume': 1000
    }])

    bars_data = {"A": bars_a, "B": bars_b}

    print("Processing bar to trigger Stop Loss...")
    signals = await strategy.on_bar_multi(["A", "B"], "1 min", bars_data)
    
    print(f"Generated {len(signals)} signals.")
    for s in signals:
        print(f"Signal: {s.signal_type} {s.symbol} Reason: {s.metadata.get('exit_reason')}")

    # Check cooldown
    cooldown = pair_state.get('cooldown_remaining', 0)
    print(f"Cooldown after exit: {cooldown}")

    if cooldown == 0:
        print("❌ BUG REPRODUCED: Cooldown is 0 after stop loss exit!")
        return False
    else:
        print(f"✅ Cooldown set correctly: {cooldown}")
        return True

if __name__ == "__main__":
    asyncio.run(reproduce_bug())
