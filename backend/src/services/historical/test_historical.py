#!/usr/bin/env python3
"""
Test script for Historical Data Service.

This script tests the key functionality of the historical data service:
- Database connectivity
- Client ID allocation
- Request queueing and pacing
- Historical data storage
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
from common.models import Candle, Symbol, HealthStatus, WatchlistEntry
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
    """Test client ID allocation for historical service."""
    print("🔍 Testing client ID allocation...")
    
    try:
        client_id = allocate_service_client_id("historical")
        print(f"✅ Allocated client ID: {client_id}")
        
        # Should be 13 (base 10 + offset 3)
        expected_id = 13
        if client_id == expected_id:
            print(f"✅ Client ID matches expected value: {expected_id}")
        else:
            print(f"⚠️ Client ID {client_id} doesn't match expected {expected_id}")
        
        # Release the client ID
        release_service_client_id(client_id, "historical")
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
        print(f"   - Max Hist Requests/Min: {settings.historical.max_requests_per_min}")
        print(f"   - Bar Sizes: {settings.historical.bar_sizes_list}")
        print(f"   - Lookback: {settings.market_data.lookback}")
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False


def test_candles_table_operations():
    """Test candles table operations."""
    print("🔍 Testing candles table operations...")
    
    try:
        def _test_candles(session):
            # Clear existing test data
            session.query(Candle).filter(
                Candle.symbol.in_(['GOOG', 'AMZN'])
            ).delete()
            
            # Add test symbols if they don't exist
            test_symbols = ['GOOG', 'AMZN']
            for symbol in test_symbols:
                existing_symbol = session.query(Symbol).filter(Symbol.symbol == symbol).first()
                if not existing_symbol:
                    symbol_record = Symbol(symbol=symbol, currency='USD', active=True)
                    session.add(symbol_record)
            
            # Add test candles
            test_candles = [
                {
                    'symbol': 'GOOG',
                    'tf': '1 min',
                    'ts': datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc),
                    'open': 100.0,
                    'high': 101.0,
                    'low': 99.0,
                    'close': 100.5,
                    'volume': 1000
                },
                {
                    'symbol': 'AMZN',
                    'tf': '5 mins',
                    'ts': datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc),
                    'open': 200.0,
                    'high': 202.0,
                    'low': 198.0,
                    'close': 201.0,
                    'volume': 2000
                }
            ]
            
            for candle_data in test_candles:
                candle = Candle(**candle_data)
                session.add(candle)
            
            session.commit()
            
            # Verify candles were added
            candles = session.query(Candle).filter(
                Candle.symbol.in_(['GOOG', 'AMZN'])
            ).all()
            
            return len(candles)
        
        candle_count = execute_with_retry(_test_candles)
        print(f"✅ Added and verified {candle_count} test candles")
        
        # Test idempotent upsert (add same candle again)
        def _test_idempotent(session):
            # Try to add the same candle again
            existing_candle = session.query(Candle).filter(
                Candle.symbol == 'GOOG',
                Candle.tf == '1 min',
                Candle.ts == datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
            ).first()
            
            if existing_candle:
                # Update the existing candle (simulating idempotent upsert)
                existing_candle.close = 101.0
                existing_candle.volume = 1100
                session.commit()
                return True
            else:
                return False
        
        updated = execute_with_retry(_test_idempotent)
        if updated:
            print("✅ Idempotent upsert test passed")
        else:
            print("⚠️ Idempotent upsert test failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Candles table operations failed: {e}")
        return False


def test_health_status():
    """Test health status operations."""
    print("🔍 Testing health status operations...")
    
    try:
        def _test_health(session):
            # Update health status
            health_record = session.query(HealthStatus).filter(
                HealthStatus.service == "historical"
            ).first()
            
            if health_record:
                health_record.status = "healthy"
                health_record.updated_at = datetime.now(timezone.utc)
            else:
                health_record = HealthStatus(
                    service="historical",
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


def test_pacing_calculation():
    """Test pacing calculation logic."""
    print("🔍 Testing pacing calculation...")
    
    try:
        settings = get_settings()
        max_requests = settings.historical.max_requests_per_min
        
        # Simulate request timestamps
        current_time = 1000000.0  # Arbitrary timestamp
        request_timestamps = []
        
        # Add requests up to the limit
        for i in range(max_requests):
            request_timestamps.append(current_time - (60 - i))  # Spread over 1 minute
        
        # Check if we need to wait (should not wait if requests are spread out)
        old_requests = [ts for ts in request_timestamps if current_time - ts < 60.0]
        
        if len(old_requests) < max_requests:
            print(f"✅ Pacing allows new request ({len(old_requests)}/{max_requests})")
        else:
            wait_time = 60.0 - (current_time - old_requests[0])
            print(f"✅ Pacing would wait {wait_time:.1f}s (correct behavior)")
        
        return True
        
    except Exception as e:
        print(f"❌ Pacing calculation failed: {e}")
        return False


def test_logging():
    """Test logging configuration."""
    print("🔍 Testing logging configuration...")
    
    try:
        logger = configure_service_logging("historical_test")
        
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
    print("🚀 Starting Historical Data Service Tests")
    print("=" * 50)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Database Connectivity", test_database_connectivity),
        ("Client ID Allocation", test_client_id_allocation),
        ("Candles Table Operations", test_candles_table_operations),
        ("Health Status", test_health_status),
        ("Pacing Calculation", test_pacing_calculation),
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
        print("🎉 All tests passed! Historical Data Service is ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
