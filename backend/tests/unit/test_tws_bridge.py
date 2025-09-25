"""
Tests for TWS bridge components.

Tests client ID management, IB client wrapper functionality, and base service class
without requiring actual TWS connection.
"""

import sys
import os
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Mock the common modules for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def test_client_id_manager():
    """Test client ID management functionality."""
    try:
        from tws_bridge.client_ids import (
            ClientIdManager,
            allocate_service_client_id,
            release_service_client_id,
            get_client_id_usage_stats,
            detect_client_id_conflicts,
            reclaim_service_client_id
        )
        
        print("✅ Client ID modules imported successfully")
        
        # Test manager creation
        manager = ClientIdManager()
        assert manager.base_id == 10  # Default from config
        print("✅ Client ID manager created")
        
        # Test service client ID allocation
        account_id = manager.get_service_client_id("account")
        assert account_id == 21  # base(10) + offset(11)
        
        marketdata_id = manager.get_service_client_id("marketdata")
        assert marketdata_id == 22  # base(10) + offset(12)
        
        print("✅ Single-service client ID allocation works")
        
        # Test strategy client ID allocation (multi-instance)
        strategy_id1 = manager.get_service_client_id("strategy", "strategy1")
        strategy_id2 = manager.get_service_client_id("strategy", "strategy2")
        
        assert strategy_id1 == 25  # base(10) + offset(15)
        assert strategy_id2 == 26  # base(10) + offset(16)
        assert strategy_id1 != strategy_id2
        
        print("✅ Multi-instance client ID allocation works")
        
        # Test client ID release
        manager.release_client_id(account_id, "account")
        
        # Should be able to allocate again
        new_account_id = manager.get_service_client_id("account")
        assert new_account_id == account_id
        
        print("✅ Client ID release and reallocation works")
        
        # Test usage stats
        stats = get_client_id_usage_stats()
        assert isinstance(stats, dict)
        assert "total_allocated" in stats
        assert "by_service" in stats
        
        print("✅ Usage statistics work")
        
        # Test conflict detection
        conflicts = detect_client_id_conflicts()
        assert isinstance(conflicts, list)
        
        print("✅ Conflict detection works")
        
        return True
        
    except Exception as e:
        print(f"❌ Client ID manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ib_client_wrapper():
    """Test IB client wrapper (without actual connection)."""
    try:
        from tws_bridge.ib_client import (
            EnhancedIBClient,
            ConnectionState,
            RequestThrottler,
            create_ib_client
        )
        
        print("✅ IB client modules imported successfully")
        
        # Test connection state
        state = ConnectionState()
        assert not state.connected
        assert state.client_id is None
        
        print("✅ Connection state works")
        
        # Test request throttler
        throttler = RequestThrottler(max_requests_per_second=10)
        assert throttler.max_requests_per_second == 10
        
        print("✅ Request throttler created")
        
        # Test client creation (without connecting)
        client = EnhancedIBClient(
            client_id=100,
            host="localhost",
            port=7497
        )
        
        assert client.client_id == 100
        assert client.host == "localhost"
        assert client.port == 7497
        assert not client.state.connected
        
        print("✅ Enhanced IB client creation works")
        
        # Test event handler registration
        def dummy_handler():
            pass
        
        client.add_connection_handler(dummy_handler)
        client.add_disconnection_handler(dummy_handler)
        client.add_error_handler(dummy_handler)
        
        assert len(client.connection_handlers) == 1
        assert len(client.disconnection_handlers) == 1
        assert len(client.error_handlers) == 1
        
        print("✅ Event handler registration works")
        
        # Test factory function
        factory_client = create_ib_client(200)
        assert factory_client.client_id == 200
        
        print("✅ Factory function works")
        
        return True
        
    except Exception as e:
        print(f"❌ IB client wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_base_service_class():
    """Test base TWS service class."""
    try:
        from tws_bridge.base_service import BaseTWSService
        
        print("✅ Base service class imported successfully")
        
        # Create a concrete implementation for testing
        class TestService(BaseTWSService):
            def __init__(self):
                super().__init__("test_service", "test_instance")
                self.initialized = False
                self.tasks_started = False
                self.cleaned_up = False
            
            async def on_initialize(self):
                self.initialized = True
            
            async def start_service_tasks(self):
                self.tasks_started = True
                return []
            
            async def on_cleanup(self):
                self.cleaned_up = True
            
            def should_create_api(self):
                return False  # Don't create API for testing
        
        # Test service creation
        service = TestService()
        assert service.service_name == "test_service"
        assert service.instance_id == "test_instance"
        assert not service.running
        
        print("✅ Service creation works")
        
        # Test health status
        health = asyncio.run(service.get_health_status())
        assert isinstance(health, dict)
        assert "status" in health
        assert "service" in health
        assert health["service"] == "test_service"
        
        print("✅ Health status works")
        
        # Test FastAPI app creation
        service_with_api = TestService()
        service_with_api.should_create_api = lambda: True
        
        app = service_with_api.create_fastapi_app()
        assert app is not None
        assert app.title == "Test_service Service"
        
        print("✅ FastAPI app creation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Base service class test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration between components."""
    try:
        from tws_bridge.client_ids import allocate_service_client_id
        from tws_bridge.ib_client import create_ib_client
        from tws_bridge.base_service import BaseTWSService
        
        print("✅ Integration imports successful")
        
        # Test that client ID allocation works with IB client creation
        client_id = allocate_service_client_id("test_service")
        ib_client = create_ib_client(client_id)
        
        assert ib_client.client_id == client_id
        print("✅ Client ID allocation + IB client creation integration works")
        
        # Test that base service can use both components
        class IntegrationTestService(BaseTWSService):
            async def on_initialize(self):
                pass
            
            async def start_service_tasks(self):
                return []
            
            async def on_cleanup(self):
                pass
        
        service = IntegrationTestService("integration_test")
        assert service.service_name == "integration_test"
        
        print("✅ Base service integration works")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_client_id_reclaim_logic():
    """Test client ID reclaim functionality."""
    try:
        from tws_bridge.client_ids import (
            ClientIdManager,
            reclaim_service_client_id,
            allocate_service_client_id,
            release_service_client_id
        )
        
        print("✅ Testing client ID reclaim logic")
        
        # Allocate a client ID for account service
        client_id1 = allocate_service_client_id("account")
        assert client_id1 == 21  # base(10) + offset(11)
        
        # Release it
        release_service_client_id(client_id1, "account")
        
        # Reclaim should get the same ID
        client_id2 = reclaim_service_client_id("account")
        assert client_id2 == client_id1
        
        print("✅ Client ID reclaim works for single-ID services")
        
        # Test strategy service reclaim
        strategy_id1 = allocate_service_client_id("strategy", "strategy1")
        release_service_client_id(strategy_id1, "strategy")
        
        # Reclaim should work (might get same or different ID)
        strategy_id2 = reclaim_service_client_id("strategy", "strategy1")
        assert isinstance(strategy_id2, int)
        assert 25 <= strategy_id2 <= 39  # Within strategy range
        
        print("✅ Client ID reclaim works for multi-instance services")
        
        return True
        
    except Exception as e:
        print(f"❌ Client ID reclaim test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_request_throttling():
    """Test request throttling functionality."""
    try:
        from tws_bridge.ib_client import RequestThrottler
        import time
        
        print("✅ Testing request throttling")
        
        # Create throttler with low limit for testing
        throttler = RequestThrottler(max_requests_per_second=2)
        
        # Make requests quickly
        start_time = time.time()
        
        async def make_requests():
            for i in range(3):
                await throttler.wait_if_needed(f"request_{i}")
        
        asyncio.run(make_requests())
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should take at least 0.5 seconds due to throttling
        # (3 requests, max 2 per second)
        assert duration >= 0.4  # Allow some margin
        
        print(f"✅ Request throttling works (took {duration:.2f}s for 3 requests)")
        
        return True
        
    except Exception as e:
        print(f"❌ Request throttling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all TWS bridge tests."""
    print("🔧 Testing TWS bridge components...\n")
    
    tests = [
        ("Client ID Manager", test_client_id_manager),
        ("IB Client Wrapper", test_ib_client_wrapper),
        ("Base Service Class", test_base_service_class),
        ("Component Integration", test_integration),
        ("Client ID Reclaim Logic", test_client_id_reclaim_logic),
        ("Request Throttling", test_request_throttling),
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
        print("🎉 All TWS bridge tests passed!")
        return True
    else:
        print(f"❌ {total - passed} tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
