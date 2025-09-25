"""
Tests for common libraries functionality.

Tests that all common modules can be imported and basic functionality works.
"""

import sys
import os
import json
from decimal import Decimal
from datetime import datetime, timezone

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def test_config_import_and_usage():
    """Test that config module can be imported and used."""
    try:
        from common.config import (
            TradingBotSettings,
            get_settings,
            get_client_id_for_service,
            get_strategy_client_id
        )
        
        print("✅ Config module imported successfully")
        
        # Test settings creation
        settings = get_settings()
        assert settings is not None
        assert hasattr(settings, 'database')
        assert hasattr(settings, 'tws')
        print("✅ Settings object created and has expected attributes")
        
        # Test client ID functions
        account_id = get_client_id_for_service("account")
        assert account_id == 11
        
        strategy_id = get_strategy_client_id(5)
        assert strategy_id == 20
        print("✅ Client ID functions work correctly")
        
        # Test safety validation
        is_safe, violations = settings.validate_live_trading_safety()
        assert isinstance(is_safe, bool)
        assert isinstance(violations, list)
        print("✅ Safety validation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_models_import_and_usage():
    """Test that models module can be imported and used."""
    try:
        from common.models import (
            Symbol, WatchlistEntry, Strategy, Order, Execution,
            Tick, Candle, Signal, BacktestRun, get_utc_now
        )
        
        print("✅ Models module imported successfully")
        
        # Test model creation (without database)
        symbol = Symbol(
            symbol='TEST',
            conid=12345,
            primary_exchange='NASDAQ',
            currency='USD'
        )
        assert symbol.symbol == 'TEST'
        assert symbol.conid == 12345
        print("✅ Model objects can be created")
        
        # Test utility functions
        now = get_utc_now()
        assert isinstance(now, datetime)
        print("✅ Utility functions work")
        
        return True
        
    except Exception as e:
        print(f"❌ Models test failed: {e}")
        return False


def test_logging_import_and_usage():
    """Test that logging module can be imported and used."""
    try:
        from common.logging import (
            setup_logging,
            get_logger,
            log_execution_time,
            JSONFormatter,
            log_trade_event,
            log_system_event
        )
        
        print("✅ Logging module imported successfully")
        
        # Test logger setup
        logger = setup_logging("test_service", log_level="INFO", enable_db_logging=False)
        assert logger is not None
        print("✅ Logger setup works")
        
        # Test JSON formatter
        formatter = JSONFormatter("test_service")
        assert formatter is not None
        print("✅ JSON formatter created")
        
        # Test logging functions
        test_logger = get_logger(__name__)
        log_trade_event(test_logger, "test_event", "TEST", qty=100)
        log_system_event(test_logger, "startup", service="test")
        print("✅ Logging functions work")
        
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False


def test_schemas_import_and_usage():
    """Test that schemas module can be imported and used."""
    try:
        from common.schemas import (
            OrderRequest, OrderResponse, TickSchema, CandleSchema,
            SignalSchema, StrategySchema, BacktestRequest, APIResponse,
            OrderSide, OrderType, SignalType, create_success_response,
            validate_symbol_format, validate_price
        )
        
        print("✅ Schemas module imported successfully")
        
        # Test schema creation and validation
        order_req = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=100,
            order_type=OrderType.MARKET
        )
        assert order_req.symbol == "AAPL"
        assert order_req.side == OrderSide.BUY
        print("✅ Order request schema validation works")
        
        # Test tick schema
        tick = TickSchema(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            bid=Decimal("150.00"),
            ask=Decimal("150.05"),
            last=Decimal("150.02")
        )
        assert tick.symbol == "AAPL"
        print("✅ Tick schema validation works")
        
        # Test candle schema with validation
        candle = CandleSchema(
            symbol="AAPL",
            tf="1 min",
            ts=datetime.now(timezone.utc),
            open=Decimal("150.00"),
            high=Decimal("150.50"),
            low=Decimal("149.50"),
            close=Decimal("150.25"),
            volume=1000
        )
        assert candle.high >= candle.open
        assert candle.low <= candle.close
        print("✅ Candle schema validation works")
        
        # Test utility functions
        response = create_success_response({"test": "data"})
        assert response.success is True
        
        normalized_symbol = validate_symbol_format("  aapl  ")
        assert normalized_symbol == "AAPL"
        
        validated_price = validate_price(150.50)
        assert validated_price == Decimal("150.50")
        print("✅ Schema utility functions work")
        
        return True
        
    except Exception as e:
        print(f"❌ Schemas test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_notify_import_and_basic_usage():
    """Test that notify module can be imported (without database connection)."""
    try:
        from common.notify import (
            NotificationManager,
            get_notification_manager,
            notify_new_signal,
            notify_new_order,
            notify_watchlist_update,
            AsyncNotificationManager,
            create_notification_manager
        )
        
        print("✅ Notify module imported successfully")
        
        # Test manager creation (without connecting)
        manager = create_notification_manager(async_mode=False)
        assert isinstance(manager, NotificationManager)
        
        async_manager = create_notification_manager(async_mode=True)
        assert isinstance(async_manager, AsyncNotificationManager)
        print("✅ Notification managers can be created")
        
        # Test handler management (without starting)
        def dummy_handler(channel, data):
            pass
        
        manager.add_handler("test_channel", dummy_handler)
        assert "test_channel" in manager.handlers
        
        manager.remove_handler("test_channel", dummy_handler)
        assert "test_channel" not in manager.handlers
        print("✅ Handler management works")
        
        return True
        
    except Exception as e:
        print(f"❌ Notify test failed: {e}")
        return False


def test_db_import_and_basic_usage():
    """Test that db module can be imported (without actual database connection)."""
    try:
        from common.db import (
            create_database_engine,
            get_db_session,
            execute_with_retry,
            check_database_health,
            get_database_info,
            upsert_health_status
        )
        
        print("✅ Database module imported successfully")
        
        # Test that functions exist and are callable
        assert callable(create_database_engine)
        assert callable(get_db_session)
        assert callable(execute_with_retry)
        assert callable(check_database_health)
        print("✅ Database functions are available")
        
        return True
        
    except Exception as e:
        print(f"❌ Database module test failed: {e}")
        return False


def test_cross_module_integration():
    """Test that modules can work together."""
    try:
        from common.config import get_settings
        from common.logging import setup_logging
        from common.schemas import OrderRequest, OrderSide, OrderType
        from common.models import Symbol
        
        print("✅ Cross-module imports successful")
        
        # Test config + logging integration
        settings = get_settings()
        logger = setup_logging(
            "integration_test",
            log_level=settings.logging.level,
            enable_db_logging=False
        )
        logger.info("Integration test message")
        print("✅ Config + Logging integration works")
        
        # Test schemas + models integration
        order_req = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=100,
            order_type=OrderType.MARKET
        )
        
        symbol = Symbol(
            symbol=order_req.symbol,
            currency="USD"
        )
        assert symbol.symbol == order_req.symbol
        print("✅ Schemas + Models integration works")
        
        return True
        
    except Exception as e:
        print(f"❌ Cross-module integration test failed: {e}")
        return False


def run_all_tests():
    """Run all common library tests."""
    print("🔧 Testing common libraries...\n")
    
    tests = [
        ("Config Module", test_config_import_and_usage),
        ("Models Module", test_models_import_and_usage),
        ("Logging Module", test_logging_import_and_usage),
        ("Schemas Module", test_schemas_import_and_usage),
        ("Notify Module", test_notify_import_and_basic_usage),
        ("Database Module", test_db_import_and_basic_usage),
        ("Cross-Module Integration", test_cross_module_integration)
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
        print("🎉 All common library tests passed!")
        return True
    else:
        print(f"❌ {total - passed} tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
