#!/usr/bin/env python3
"""
Test script for strategy interface and examples.

This script tests the strategy interface without requiring a full database
or TWS connection. It's useful for development and validation.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_lib import (
    get_strategy_registry,
    load_strategies_from_directory,
    StrategyConfig,
    SignalType,
    validate_bars_dataframe
)


def create_sample_bars(symbol: str = "AAPL", num_bars: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)  # For reproducible results
    
    # Generate timestamps
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=num_bars)
    timestamps = pd.date_range(start=start_time, end=end_time, periods=num_bars)
    
    # Generate price data (random walk)
    base_price = 150.0
    returns = np.random.normal(0, 0.001, num_bars)  # 0.1% volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
        # Create realistic OHLC from close price
        volatility = abs(np.random.normal(0, 0.005))  # Intraday volatility
        high = close * (1 + volatility)
        low = close * (1 - volatility)
        open_price = prices[i-1] if i > 0 else close
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        volume = np.random.randint(100000, 1000000)
        
        data.append({
            'timestamp': timestamp,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
    
    return pd.DataFrame(data)


async def test_sma_strategy():
    """Test SMA crossover strategy."""
    print("🧪 Testing SMA Crossover Strategy...")
    
    registry = get_strategy_registry()
    
    # Load strategies
    loaded_count = load_strategies_from_directory()
    print(f"📚 Loaded {loaded_count} strategies")
    
    # List available strategies
    strategies = registry.list_strategies()
    print(f"📋 Available strategies: {strategies}")
    
    if "SMA_Crossover" not in strategies:
        print("❌ SMA_Crossover strategy not found!")
        return False
    
    # Create strategy configuration
    config = StrategyConfig(
        strategy_id="test_sma_001",
        name="Test SMA Strategy",
        symbols=["AAPL", "MSFT"],
        enabled=True,
        bar_timeframe="1 min",
        lookback_periods=60,
        parameters={
            "short_period": 10,
            "long_period": 20,
            "min_cross_strength": 0.005,
            "fixed_position_size": 100
        }
    )
    
    # Create strategy instance
    strategy = registry.create_strategy_instance("SMA_Crossover", config)
    print(f"✅ Created strategy instance: {strategy.config.strategy_id}")
    
    # Initialize strategy
    instruments = {"AAPL": {}, "MSFT": {}}
    await strategy.on_start(instruments)
    print(f"🚀 Strategy state: {strategy.get_state()}")
    
    # Test with sample data
    for symbol in ["AAPL", "MSFT"]:
        print(f"\n📊 Testing {symbol}...")
        
        # Create sample bars
        bars = create_sample_bars(symbol, 60)
        
        # Validate bars
        if not validate_bars_dataframe(bars):
            print(f"❌ Invalid bars dataframe for {symbol}")
            continue
        
        print(f"📈 Generated {len(bars)} bars for {symbol}")
        print(f"   Price range: ${bars['low'].min():.2f} - ${bars['high'].max():.2f}")
        print(f"   Latest close: ${bars['close'].iloc[-1]:.2f}")
        
        # Process bars through strategy
        signals = await strategy.on_bar(symbol, "1 min", bars)
        
        if signals:
            print(f"🎯 Generated {len(signals)} signals:")
            for signal in signals:
                print(f"   - {signal.signal_type.value} {signal.symbol} @ ${signal.price} "
                      f"(strength: {signal.strength:.3f}, qty: {signal.quantity})")
        else:
            print("   No signals generated")
    
    # Test strategy metrics
    metrics = strategy.get_metrics()
    print(f"\n📈 Strategy metrics:")
    print(f"   Total signals: {metrics.total_signals}")
    print(f"   Successful signals: {metrics.successful_signals}")
    print(f"   Win rate: {metrics.win_rate}")
    
    # Stop strategy
    await strategy.on_stop()
    print(f"🛑 Strategy stopped. Final state: {strategy.get_state()}")
    
    return True


async def test_mean_reversion_strategy():
    """Test Mean Reversion strategy."""
    print("\n🧪 Testing Mean Reversion Strategy...")
    
    registry = get_strategy_registry()
    
    if "Mean_Reversion" not in registry.list_strategies():
        print("❌ Mean_Reversion strategy not found!")
        return False
    
    # Create strategy configuration
    config = StrategyConfig(
        strategy_id="test_mr_001",
        name="Test Mean Reversion Strategy",
        symbols=["SPY"],
        enabled=True,
        bar_timeframe="1 min",
        lookback_periods=50,
        parameters={
            "bb_period": 20,
            "bb_std_dev": 2.0,
            "rsi_period": 14,
            "rsi_overbought": 70.0,
            "rsi_oversold": 30.0,
            "price_deviation_threshold": 0.02,
            "min_reversion_strength": 0.5,
            "profit_target_pct": 0.03,
            "stop_loss_pct": 0.02,
            "max_hold_bars": 50,
            "base_position_size": 50,
            "risk_per_trade_pct": 0.01
        }
    )
    
    # Create strategy instance
    strategy = registry.create_strategy_instance("Mean_Reversion", config)
    print(f"✅ Created strategy instance: {strategy.config.strategy_id}")
    
    # Initialize strategy
    instruments = {"SPY": {}}
    await strategy.on_start(instruments)
    
    # Create sample data with more volatility for mean reversion signals
    print(f"📊 Testing SPY with volatile data...")
    bars = create_sample_bars("SPY", 50)
    
    # Add some extreme moves to trigger signals
    bars.loc[bars.index[-10:], 'close'] *= 1.05  # 5% spike
    bars.loc[bars.index[-5:], 'close'] *= 0.95   # 5% drop
    
    print(f"📈 Generated {len(bars)} bars for SPY")
    print(f"   Latest close: ${bars['close'].iloc[-1]:.2f}")
    
    # Process bars
    signals = await strategy.on_bar("SPY", "1 min", bars)
    
    if signals:
        print(f"🎯 Generated {len(signals)} signals:")
        for signal in signals:
            print(f"   - {signal.signal_type.value} {signal.symbol} @ ${signal.price} "
                  f"(strength: {signal.strength:.3f}, qty: {signal.quantity})")
            print(f"     Metadata: {signal.metadata}")
    else:
        print("   No signals generated")
    
    # Stop strategy
    await strategy.on_stop()
    print(f"🛑 Strategy stopped. Final state: {strategy.get_state()}")
    
    return True


async def test_registry_functionality():
    """Test strategy registry functionality."""
    print("\n🧪 Testing Strategy Registry...")
    
    registry = get_strategy_registry()
    
    # Test registry info
    all_info = registry.get_all_strategy_info()
    print(f"📋 Registry contains {len(all_info)} strategies:")
    
    for name, info in all_info.items():
        print(f"   - {name}: {info.description}")
        if info.parameters_schema:
            print(f"     Parameters: {list(info.parameters_schema.keys())}")
    
    # Test strategy info lookup
    sma_info = registry.get_strategy_info("SMA_Crossover")
    if sma_info:
        print(f"\n📊 SMA Strategy Info:")
        print(f"   Class: {sma_info.class_name}")
        print(f"   Module: {sma_info.module_path}")
        print(f"   Parameters: {list(sma_info.parameters_schema.keys())}")
    
    return True


async def main():
    """Run all tests."""
    print("🚀 Starting Strategy Interface Tests...\n")
    
    try:
        # Test registry
        success1 = await test_registry_functionality()
        
        # Test SMA strategy
        success2 = await test_sma_strategy()
        
        # Test Mean Reversion strategy
        success3 = await test_mean_reversion_strategy()
        
        # Summary
        print("\n" + "="*60)
        print("📋 Test Results Summary:")
        print(f"   Registry functionality: {'✅ PASS' if success1 else '❌ FAIL'}")
        print(f"   SMA Crossover strategy: {'✅ PASS' if success2 else '❌ FAIL'}")
        print(f"   Mean Reversion strategy: {'✅ PASS' if success3 else '❌ FAIL'}")
        
        all_passed = success1 and success2 and success3
        print(f"\n🎯 Overall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
