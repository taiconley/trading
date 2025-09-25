#!/usr/bin/env python3
"""
Test script for Market Data Service.

This script tests the key functionality of the market data service:
- Database connectivity
- Watchlist loading
- Subscription limits
- Health monitoring
"""

import asyncio
import sys
import os
from datetime import datetime, timezone

# Add the src directory to Python path for Docker compatibility
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from common.config import get_settings
from common.db import get_db_session, execute_with_retry, initialize_database
from common.models import WatchlistEntry, Symbol, HealthStatus
from common.logging import configure_service_logging
from tws_bridge.client_ids import allocate_service_client_id, release_service_client_id


def test_database_connectivity():
    """Test database connectivity."""
    print("🔍 Testing database connectivity...")
    
    try:
        initialize_database()
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


def test_client_id_allocation():
    """Test client ID allocation for marketdata service."""
    print("🔍 Testing client ID allocation...")
    
    try:
        client_id = allocate_service_client_id("marketdata")
        print(f"✅ Allocated client ID: {client_id}")
        
        # Should be 12 (base 10 + offset 2)
        expected_id = 12
        if client_id == expected_id:
            print(f"✅ Client ID matches expected value: {expected_id}")
        else:
            print(f"⚠️ Client ID {client_id} doesn't match expected {expected_id}")
        
        # Release the client ID
        release_service_client_id(client_id, "marketdata")
        print("✅ Released client ID")
        return True
        
    except Exception as e:
        print(f"❌ Client ID allocation failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading."""
    print("🔍 Testing configuration loading...")
    
    try:
        settings = get_settings()
        print(f"✅ Configuration loaded")
        print(f"   - TWS Host: {settings.tws.host}")
        print(f"   - TWS Port: {settings.tws.port}")
        print(f"   - Max Subscriptions: {settings.market_data.max_subscriptions}")
        print(f"   - Default Symbols: {settings.market_data.symbols_list}")
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False


def test_watchlist_operations():
    """Test watchlist database operations."""
    print("🔍 Testing watchlist operations...")
    
    try:
        def _test_watchlist(session):
            # Clear existing test symbols
            session.query(WatchlistEntry).filter(
                WatchlistEntry.symbol.in_(['TEST1', 'TEST2', 'TEST3'])
            ).delete()
            
            # Add test symbols to watchlist
            test_symbols = ['TEST1', 'TEST2', 'TEST3']
            for symbol in test_symbols:
                # Ensure symbol exists
                existing_symbol = session.query(Symbol).filter(Symbol.symbol == symbol).first()
                if not existing_symbol:
                    symbol_record = Symbol(symbol=symbol, currency='USD', active=True)
                    session.add(symbol_record)
                
                # Add to watchlist
                watchlist_entry = WatchlistEntry(symbol=symbol)
                session.add(watchlist_entry)
            
            session.commit()
            
            # Verify watchlist entries
            entries = session.query(WatchlistEntry).filter(
                WatchlistEntry.symbol.in_(test_symbols)
            ).all()
            
            return [entry.symbol for entry in entries]
        
        symbols = execute_with_retry(_test_watchlist)
        print(f"✅ Added {len(symbols)} symbols to watchlist: {symbols}")
        
        # Test subscription limit simulation
        settings = get_settings()
        max_subs = settings.market_data.max_subscriptions
        
        if len(symbols) <= max_subs:
            print(f"✅ Watchlist size ({len(symbols)}) within subscription limit ({max_subs})")
        else:
            print(f"⚠️ Watchlist size ({len(symbols)}) exceeds subscription limit ({max_subs})")
        
        return True
        
    except Exception as e:
        print(f"❌ Watchlist operations failed: {e}")
        return False


def test_health_status():
    """Test health status operations."""
    print("🔍 Testing health status operations...")
    
    try:
        def _test_health(session):
            # Update health status
            health_record = session.query(HealthStatus).filter(
                HealthStatus.service == "marketdata"
            ).first()
            
            if health_record:
                health_record.status = "healthy"
                health_record.updated_at = datetime.now(timezone.utc)
            else:
                health_record = HealthStatus(
                    service="marketdata",
                    status="healthy"
                )
                session.add(health_record)
            
            session.commit()
            return health_record
        
        health_record = execute_with_retry(_test_health)
        print(f"✅ Health status updated: {health_record.status}")
        return True
        
    except Exception as e:
        print(f"❌ Health status operations failed: {e}")
        return False


def test_logging():
    """Test logging configuration."""
    print("🔍 Testing logging configuration...")
    
    try:
        logger = configure_service_logging("marketdata_test")
        
        # Test different log levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        
        print("✅ Logging configuration successful")
        return True
        
    except Exception as e:
        print(f"❌ Logging configuration failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Market Data Service Tests")
    print("=" * 50)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Database Connectivity", test_database_connectivity),
        ("Client ID Allocation", test_client_id_allocation),
        ("Watchlist Operations", test_watchlist_operations),
        ("Health Status", test_health_status),
        ("Logging", test_logging),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Market Data Service is ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
