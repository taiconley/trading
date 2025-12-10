#!/usr/bin/env python3
"""
Script to update the Pairs_Trading strategy parameters in the database.

This script reads from the single source of truth:
    backend/src/strategy_lib/pairs_trading_config.py

To change pairs or parameters, edit that file and then run this script.
"""

import sys
import os
import json

# Add backend to path for imports
# Handle both cases: running from project root or from backend directory
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir.endswith('/backend'):
    # Running from backend directory
    src_path = os.path.join(script_dir, 'src')
else:
    # Running from project root (or container /app)
    src_path = os.path.join(script_dir, 'src') if os.path.exists(os.path.join(script_dir, 'src')) else '/app/src'
sys.path.insert(0, src_path)

from common.db import get_db_session
from common.models import Strategy
from strategy_lib.pairs_trading_config import PAIRS_TRADING_CONFIG

# Use the shared configuration as the single source of truth
DEFAULT_PARAMS = PAIRS_TRADING_CONFIG


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

