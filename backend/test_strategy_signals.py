#!/usr/bin/env python3
"""
Quick test to see if SMA strategy generates signals on AAPL data
"""
import sys
import os
import asyncio
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from common.config import get_settings
from common.db import execute_with_retry
from common.models import Candle
from strategy_lib import StrategyConfig
from strategy_lib.sma_cross import SMAStrategy

async def test_signals():
    """Test if strategy generates signals."""
    
    # Load AAPL data
    def _load_data(session):
        candles = session.query(Candle).filter(
            Candle.symbol == 'AAPL',
            Candle.tf == '1 day'
        ).order_by(Candle.ts).all()
        
        return pd.DataFrame([
            {
                'timestamp': c.ts,
                'open': float(c.open),
                'high': float(c.high),
                'low': float(c.low),
                'close': float(c.close),
                'volume': c.volume
            }
            for c in candles
        ])
    
    df = execute_with_retry(_load_data)
    print(f"Loaded {len(df)} bars")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
    
    # Create strategy
    config = StrategyConfig(
        strategy_id='test_sma',
        name='SMA_Crossover',
        symbols=['AAPL'],
        enabled=True,
        bar_timeframe='1 day',
        lookback_periods=60,
        parameters={
            'short_period': 10,
            'long_period': 20,
            'fixed_position_size': 100
        }
    )
    
    # Manually add params to config
    config.short_period = 10
    config.long_period = 20
    config.fixed_position_size = 100
    
    strategy = SMAStrategy(config)
    
    # Initialize
    await strategy.on_start({'AAPL': {'symbol': 'AAPL'}})
    
    # Test on progressively more data
    print("\nTesting signal generation...")
    signal_count = 0
    
    for i in range(60, len(df)):
        bars = df.iloc[:i+1].tail(60)
        signals = await strategy.on_bar('AAPL', '1 day', bars)
        
        if signals:
            for signal in signals:
                signal_count += 1
                print(f"Bar {i}: {signal.signal_type} signal - strength={signal.strength}, "
                      f"price=${signal.price:.2f}, qty={signal.quantity}, "
                      f"date={df.iloc[i]['timestamp'].date()}")
    
    print(f"\nTotal signals generated: {signal_count}")
    
    await strategy.on_stop()

if __name__ == '__main__':
    asyncio.run(test_signals())

