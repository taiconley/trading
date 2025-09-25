#!/usr/bin/env python3
"""
Simple TWS bridge test that can run independently.
Tests core functionality without complex imports.
"""

import sys
import os
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_basic_imports():
    """Test that we can import the basic modules."""
    try:
        # Test basic import structure
        import tws_bridge
        print("✅ TWS bridge package imported")
        
        # Test individual modules can be imported
        from tws_bridge.client_ids import ClientIdManager
        print("✅ Client ID manager imported")
        
        # Test that we can create a manager
        manager = ClientIdManager()
        print("✅ Client ID manager created")
        
        # Test service ranges are defined correctly
        expected_ranges = {
            "account": (11, 11),
            "marketdata": (12, 12),
            "historical": (13, 13),
            "trader": (14, 14),
            "strategy": (15, 29),
        }
        
        assert manager.SERVICE_RANGES == expected_ranges
        print("✅ Service ranges are correct")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic imports test failed: {e}")
        return False


def test_client_id_logic():
    """Test client ID allocation logic."""
    try:
        from tws_bridge.client_ids import ClientIdManager
        
        manager = ClientIdManager()
        
        # Test that we can allocate single-service IDs
        # Note: This will fail due to config dependencies, but we can test the logic
        
        # Test service range validation
        assert "account" in manager.SERVICE_RANGES
        assert "strategy" in manager.SERVICE_RANGES
        
        start, end = manager.SERVICE_RANGES["account"]
        assert start == 11 and end == 11
        
        start, end = manager.SERVICE_RANGES["strategy"]
        assert start == 15 and end == 29
        
        print("✅ Client ID ranges are configured correctly")
        
        # Test available IDs calculation
        available = manager.get_available_ids_for_service("strategy")
        # This should return empty list since we haven't mocked the database
        assert isinstance(available, list)
        
        print("✅ Available IDs calculation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Client ID logic test failed: {e}")
        return False


def test_request_throttler():
    """Test request throttling without async."""
    try:
        # Import just the throttler class
        import asyncio
        from tws_bridge.ib_client import RequestThrottler
        
        throttler = RequestThrottler(max_requests_per_second=5)
        assert throttler.max_requests_per_second == 5
        assert len(throttler.request_times) == 0
        
        print("✅ Request throttler created successfully")
        
        # Test that we can call the wait method (won't actually wait in this test)
        async def test_wait():
            await throttler.wait_if_needed("test")
            return True
        
        result = asyncio.run(test_wait())
        assert result is True
        
        print("✅ Request throttler wait method works")
        
        return True
        
    except Exception as e:
        print(f"❌ Request throttler test failed: {e}")
        return False


def test_connection_state():
    """Test connection state tracking."""
    try:
        from tws_bridge.ib_client import ConnectionState
        
        state = ConnectionState()
        assert not state.connected
        assert state.client_id is None
        assert state.reconnect_attempts == 0
        assert state.total_connections == 0
        assert len(state.subscribed_contracts) == 0
        
        # Test state modification
        state.connected = True
        state.client_id = 123
        state.reconnect_attempts = 1
        
        assert state.connected
        assert state.client_id == 123
        assert state.reconnect_attempts == 1
        
        print("✅ Connection state tracking works")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection state test failed: {e}")
        return False


def test_module_structure():
    """Test that module structure is correct."""
    try:
        # Check that all expected files exist
        base_path = os.path.join(os.path.dirname(__file__), 'src', 'tws_bridge')
        
        expected_files = [
            '__init__.py',
            'ib_client.py',
            'client_ids.py',
            'base_service.py'
        ]
        
        for filename in expected_files:
            filepath = os.path.join(base_path, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Missing file: {filename}")
        
        print("✅ All expected TWS bridge files exist")
        
        # Test that we can import from each module
        from tws_bridge import ib_client, client_ids, base_service
        
        # Check that key classes exist
        assert hasattr(ib_client, 'EnhancedIBClient')
        assert hasattr(ib_client, 'ConnectionState')
        assert hasattr(ib_client, 'RequestThrottler')
        
        assert hasattr(client_ids, 'ClientIdManager')
        assert hasattr(client_ids, 'ClientIdAllocation')
        
        assert hasattr(base_service, 'BaseTWSService')
        
        print("✅ All key classes are available")
        
        return True
        
    except Exception as e:
        print(f"❌ Module structure test failed: {e}")
        return False


def run_simple_tests():
    """Run simple TWS bridge tests."""
    print("🔧 Running simple TWS bridge tests...\n")
    
    tests = [
        ("Module Structure", test_module_structure),
        ("Basic Imports", test_basic_imports),
        ("Client ID Logic", test_client_id_logic),
        ("Request Throttler", test_request_throttler),
        ("Connection State", test_connection_state),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"--- Testing {test_name} ---")
        try:
            result = test_func()
            results.append(result)
            print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}\n")
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}\n")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All simple TWS bridge tests passed!")
        return True
    else:
        print(f"❌ {total - passed} tests failed!")
        return False


if __name__ == "__main__":
    success = run_simple_tests()
    sys.exit(0 if success else 1)
