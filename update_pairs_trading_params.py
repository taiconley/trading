#!/usr/bin/env python3
"""
Script to update the Pairs_Trading strategy parameters in the database.
"""

import sys
import os
import json

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from common.db import get_db_session
from common.models import Strategy
from sqlalchemy import text

# Default parameters from pairs_trade.py
DEFAULT_PARAMS = {
    "pairs": [
        ["AAPL", "MSFT"],   # Tech Large Cap
        ["JPM", "BAC"],     # Banks
        ["GS", "MS"],       # Investment Banks
        ["XOM", "CVX"],     # Energy
        ["V", "MA"],        # Payments
        ["KO", "PEP"],      # Beverages
        ["WMT", "TGT"],     # Retail
        ["PFE", "MRK"],     # Pharma
        ["DIS", "NFLX"]     # Media
    ],
    "lookback_window": 240,
    "entry_threshold": 2.0,
    "exit_threshold": 0.5,
    "position_size": 100,
    "max_hold_bars": 720,
    "stop_loss_zscore": 3.0,
    "market_close_hour": 16,
    "market_close_minute": 0,
    "close_before_eod_minutes": 5,
    "cooldown_bars": 60,
    "timezone": "US/Eastern",
    "spread_history_bars": 1000,
    "hedge_refresh_bars": 30,
    "min_hedge_lookback": 120,
    "stationarity_checks_enabled": True,
    "adf_pvalue_threshold": 0.05,
    "cointegration_pvalue_threshold": 0.05,
    "stationarity_check_interval": 60,
    "volatility_adaptation_enabled": True,
    "volatility_window": 240,
    "volatility_ema_alpha": 0.2,
    "min_volatility_ratio": 0.75,
    "max_volatility_ratio": 1.5,
    "min_exit_volatility_ratio": 0.8,
    "max_exit_volatility_ratio": 1.3
}


def list_strategies():
    """List all strategies in the database."""
    print("\n=== Current Strategies in Database ===\n")
    
    with get_db_session() as session:
        strategies = session.query(Strategy).all()
        
        if not strategies:
            print("No strategies found in database.")
            return
        
        for strat in strategies:
            print(f"Strategy ID: {strat.strategy_id}")
            print(f"  Name: {strat.name}")
            print(f"  Enabled: {strat.enabled}")
            print(f"  Created: {strat.created_at}")
            if strat.params_json:
                print(f"  Parameters: {json.dumps(strat.params_json, indent=4)}")
            print()


def update_pairs_trading_strategy():
    """Update or create the Pairs_Trading strategy with new parameters."""
    
    strategy_name = "Pairs_Trading"
    strategy_id = "pairs_trading_001"
    
    print(f"\n=== Updating {strategy_name} Strategy ===\n")
    
    with get_db_session() as session:
        # Try to find existing strategy by name
        existing = session.query(Strategy).filter(
            (Strategy.name == strategy_name) | (Strategy.strategy_id == strategy_id)
        ).first()
        
        if existing:
            print(f"Found existing strategy: {existing.strategy_id}")
            print(f"  Current parameters: {json.dumps(existing.params_json, indent=2) if existing.params_json else 'None'}")
            
            # Update parameters
            existing.params_json = DEFAULT_PARAMS
            session.commit()
            
            print(f"\n✓ Updated strategy '{existing.strategy_id}' with new parameters")
            print(f"  New parameters: {json.dumps(DEFAULT_PARAMS, indent=2)}")
            
        else:
            print(f"No existing {strategy_name} strategy found.")
            print(f"Creating new strategy with ID: {strategy_id}")
            
            new_strategy = Strategy(
                strategy_id=strategy_id,
                name=strategy_name,
                enabled=False,  # Start disabled for safety
                params_json=DEFAULT_PARAMS
            )
            
            session.add(new_strategy)
            session.commit()
            
            print(f"\n✓ Created new strategy '{strategy_id}'")
            print(f"  Enabled: False (enable manually when ready)")
            print(f"  Parameters: {json.dumps(DEFAULT_PARAMS, indent=2)}")


def verify_update():
    """Verify the strategy was updated correctly."""
    print("\n=== Verification ===\n")
    
    with get_db_session() as session:
        strategy = session.query(Strategy).filter(
            Strategy.name == "Pairs_Trading"
        ).first()
        
        if not strategy:
            print("❌ Strategy not found after update!")
            return False
        
        print(f"✓ Strategy found: {strategy.strategy_id}")
        print(f"  Name: {strategy.name}")
        print(f"  Enabled: {strategy.enabled}")
        print(f"  Parameters updated: {strategy.params_json is not None}")
        
        # Check key parameters
        if strategy.params_json:
            params = strategy.params_json
            print(f"\n  Key Parameters:")
            print(f"    - pairs: {len(params.get('pairs', []))} pairs")
            print(f"    - lookback_window: {params.get('lookback_window')}")
            print(f"    - entry_threshold: {params.get('entry_threshold')}")
            print(f"    - exit_threshold: {params.get('exit_threshold')}")
            print(f"    - position_size: {params.get('position_size')}")
            print(f"    - volatility_adaptation_enabled: {params.get('volatility_adaptation_enabled')}")
            print(f"    - stationarity_checks_enabled: {params.get('stationarity_checks_enabled')}")
        
        return True


def main():
    """Main function."""
    print("=" * 60)
    print("Pairs Trading Strategy Parameter Update Script")
    print("=" * 60)
    
    try:
        # List current strategies
        list_strategies()
        
        # Update the strategy
        update_pairs_trading_strategy()
        
        # Verify the update
        verify_update()
        
        print("\n" + "=" * 60)
        print("✓ Update completed successfully!")
        print("=" * 60)
        print("\nNote: If the strategy is enabled, you may need to reload it")
        print("      in the strategy service for changes to take effect.")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

