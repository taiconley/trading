#!/usr/bin/env python3
"""
Database testing script.
Tests basic database operations, connection retry logic, and round-trip data operations.
"""

import sys
import os
from decimal import Decimal
from datetime import datetime, timezone

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_database_operations():
    """Test basic database operations."""
    try:
        from common.db import get_db_session, check_database_health, execute_with_retry
        from common.models import Symbol, WatchlistEntry, Strategy, Candle
        
        print("✅ Database modules imported successfully")
        
        # Test database health
        health = check_database_health()
        print(f"✅ Database health check: {health['status']} (response time: {health.get('response_time_ms', 'N/A')}ms)")
        
        if health['status'] != 'healthy':
            print(f"❌ Database is not healthy: {health}")
            return False
        
        # Test basic CRUD operations
        with get_db_session() as session:
            # Insert test symbol
            test_symbol = Symbol(
                symbol='TEST',
                conid=12345,
                primary_exchange='NASDAQ',
                currency='USD',
                active=True
            )
            session.add(test_symbol)
            session.flush()  # Get the ID without committing
            
            print("✅ Symbol insert successful")
            
            # Insert watchlist entry
            watchlist_entry = WatchlistEntry(symbol='TEST')
            session.add(watchlist_entry)
            session.flush()
            
            print("✅ Watchlist entry insert successful")
            
            # Insert strategy
            strategy = Strategy(
                strategy_id='test_strategy',
                name='Test Strategy',
                enabled=True,
                params_json={'param1': 'value1', 'param2': 42}
            )
            session.add(strategy)
            session.flush()
            
            print("✅ Strategy insert successful")
            
            # Insert candle data
            candle = Candle(
                symbol='TEST',
                tf='1 min',
                ts=datetime.now(timezone.utc),
                open=Decimal('100.50'),
                high=Decimal('101.25'),
                low=Decimal('99.75'),
                close=Decimal('100.80'),
                volume=1000
            )
            session.add(candle)
            session.flush()
            
            print("✅ Candle insert successful")
            
            # Test queries
            symbol_count = session.query(Symbol).count()
            watchlist_count = session.query(WatchlistEntry).count()
            strategy_count = session.query(Strategy).count()
            candle_count = session.query(Candle).count()
            
            print(f"✅ Query test successful - Symbols: {symbol_count}, Watchlist: {watchlist_count}, Strategies: {strategy_count}, Candles: {candle_count}")
            
            # Test relationships
            symbol_with_watchlist = session.query(Symbol).filter(Symbol.symbol == 'TEST').first()
            if symbol_with_watchlist and len(symbol_with_watchlist.watchlist_entries) > 0:
                print("✅ Relationship test successful")
            else:
                print("❌ Relationship test failed")
                return False
            
            # Test updates
            symbol_with_watchlist.currency = 'CAD'
            session.flush()
            
            updated_symbol = session.query(Symbol).filter(Symbol.symbol == 'TEST').first()
            if updated_symbol.currency == 'CAD':
                print("✅ Update test successful")
            else:
                print("❌ Update test failed")
                return False
            
            # Test JSON field
            if strategy.params_json['param2'] == 42:
                print("✅ JSON field test successful")
            else:
                print("❌ JSON field test failed")
                return False
            
            # Rollback to clean up test data
            session.rollback()
            print("✅ Test data rolled back successfully")
        
        print("\n🎉 All database tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retry_logic():
    """Test database retry logic."""
    try:
        from common.db import execute_with_retry
        from common.models import Symbol
        
        print("\n--- Testing Retry Logic ---")
        
        def test_operation(session):
            # This should work normally
            return session.query(Symbol).count()
        
        result = execute_with_retry(test_operation, max_retries=2)
        print(f"✅ Retry logic test successful - Symbol count: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Retry logic test failed: {e}")
        return False


if __name__ == "__main__":
    print("🔧 Starting database tests...\n")
    
    test1_success = test_database_operations()
    test2_success = test_retry_logic()
    
    if test1_success and test2_success:
        print("\n✅ All database tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some database tests failed!")
        sys.exit(1)
