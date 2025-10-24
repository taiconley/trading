"""Tests for configuration management."""

import os
import pytest
from pydantic import ValidationError
from src.common.config import (
    TradingBotSettings,
    DatabaseSettings,
    TWSSettings,
    MarketDataSettings,
    HistoricalDataSettings,
    BacktestSettings,
    get_client_id_for_service,
    get_strategy_client_id,
    reload_settings
)


class TestDatabaseSettings:
    """Test database configuration."""
    
    def test_default_values(self):
        """Test default database configuration values."""
        db_config = DatabaseSettings()
        assert db_config.user == "bot"
        assert db_config.password == "botpw"
        assert db_config.db == "trading"
        assert db_config.host == "postgres"
        assert db_config.port == 5432
        
    def test_database_url(self):
        """Test database URL generation."""
        db_config = DatabaseSettings()
        expected_url = "postgresql://bot:botpw@postgres:5432/trading"
        assert db_config.url == expected_url
        
    def test_async_database_url(self):
        """Test async database URL generation."""
        db_config = DatabaseSettings()
        expected_url = "postgresql+asyncpg://bot:botpw@postgres:5432/trading"
        assert db_config.async_url == expected_url


class TestTWSSettings:
    """Test TWS configuration."""
    
    def test_default_values(self):
        """Test default TWS configuration values."""
        tws_config = TWSSettings()
        assert tws_config.host == "172.25.0.100"
        assert tws_config.port == 7497  # Paper trading default
        assert tws_config.use_paper is True
        assert tws_config.enable_live is False
        assert tws_config.dry_run is True
        
    def test_paper_trading_port_validation(self):
        """Test that paper trading requires port 7497."""
        # This should work
        TWSSettings(use_paper=True, port=7497)
        
        # This should fail
        with pytest.raises(ValidationError):
            TWSSettings(use_paper=True, port=7496)
            
    def test_live_trading_port_validation(self):
        """Test that live trading requires port 7496."""
        # This should work
        TWSSettings(use_paper=False, port=7496)
        
        # This should fail  
        with pytest.raises(ValidationError):
            TWSSettings(use_paper=False, port=7497)
            
    def test_is_live_mode(self):
        """Test live mode detection."""
        # Paper mode
        paper_config = TWSSettings(use_paper=True, enable_live=False, dry_run=True, port=7497)
        assert paper_config.is_live_mode is False
        
        # Live mode - all conditions met
        live_config = TWSSettings(use_paper=False, enable_live=True, dry_run=False, port=7496)
        assert live_config.is_live_mode is True
        
        # Live mode - missing enable_live
        not_live_config = TWSSettings(use_paper=False, enable_live=False, dry_run=False, port=7496)
        assert not_live_config.is_live_mode is False


class TestMarketDataSettings:
    """Test market data configuration."""
    
    def test_default_values(self):
        """Test default market data configuration."""
        md_config = MarketDataSettings()
        assert md_config.default_symbols == "AAPL,MSFT,SPY"
        assert md_config.max_subscriptions == 50
        assert md_config.bar_size == "1 min"
        
    def test_symbols_list(self):
        """Test symbols list parsing."""
        md_config = MarketDataSettings(default_symbols="AAPL, msft,  spy  ")
        assert md_config.symbols_list == ["AAPL", "MSFT", "SPY"]
        
    def test_max_subscriptions_validation(self):
        """Test subscription limit validation."""
        # Valid values
        MarketDataSettings(max_subscriptions=1)
        MarketDataSettings(max_subscriptions=100)
        MarketDataSettings(max_subscriptions=1000)
        
        # Invalid values
        with pytest.raises(ValidationError):
            MarketDataSettings(max_subscriptions=0)
        with pytest.raises(ValidationError):
            MarketDataSettings(max_subscriptions=1001)


class TestHistoricalDataSettings:
    """Test historical data configuration."""
    
    def test_default_values(self):
        """Test default historical data configuration."""
        hist_config = HistoricalDataSettings()
        assert hist_config.max_requests_per_min == 30
        assert hist_config.bar_sizes == "1 min,5 mins,1 day"
        
    def test_bar_sizes_list(self):
        """Test bar sizes list parsing."""
        hist_config = HistoricalDataSettings(bar_sizes="1 min, 5 mins,  1 day  ")
        assert hist_config.bar_sizes_list == ["1 min", "5 mins", "1 day"]
        
    def test_request_rate_validation(self):
        """Test request rate validation."""
        # Valid values
        HistoricalDataSettings(max_requests_per_min=1)
        HistoricalDataSettings(max_requests_per_min=50)
        HistoricalDataSettings(max_requests_per_min=100)
        
        # Invalid values
        with pytest.raises(ValidationError):
            HistoricalDataSettings(max_requests_per_min=0)
        with pytest.raises(ValidationError):
            HistoricalDataSettings(max_requests_per_min=101)


class TestBacktestSettings:
    """Test backtest configuration."""
    
    def test_default_values(self):
        """Test default backtest configuration."""
        bt_config = BacktestSettings()
        assert bt_config.comm_per_share == 0.005
        assert bt_config.min_comm_per_order == 1.0
        assert bt_config.default_slippage_ticks == 1
        assert bt_config.tick_size_us_equity == 0.01
        
    def test_commission_validation(self):
        """Test commission validation."""
        # Valid values
        BacktestSettings(comm_per_share=0.0, min_comm_per_order=0.0)
        BacktestSettings(comm_per_share=0.01, min_comm_per_order=2.0)
        
        # Invalid values
        with pytest.raises(ValidationError):
            BacktestSettings(comm_per_share=-0.001)
        with pytest.raises(ValidationError):
            BacktestSettings(min_comm_per_order=-1.0)


class TestTradingBotSettings:
    """Test main configuration class."""
    
    def test_default_configuration(self):
        """Test that default configuration loads without errors."""
        config = TradingBotSettings()
        assert config.database is not None
        assert config.tws is not None
        assert config.market_data is not None
        assert config.historical is not None
        assert config.backtest is not None
        assert config.logging is not None
        assert config.development is not None
        
    def test_live_trading_safety_validation(self):
        """Test live trading safety validation."""
        # Paper trading configuration (safe)
        paper_config = TradingBotSettings()
        is_safe, violations = paper_config.validate_live_trading_safety()
        assert is_safe is False
        assert len(violations) > 0
        
        # Mock live trading configuration (unsafe due to missing conditions)
        live_config = TradingBotSettings()
        live_config.tws.enable_live = True
        live_config.tws.use_paper = False
        live_config.tws.dry_run = False
        live_config.tws.port = 7496
        
        is_safe, violations = live_config.validate_live_trading_safety()
        assert is_safe is True
        assert len(violations) == 0
        
    def test_is_production(self):
        """Test production mode detection."""
        config = TradingBotSettings()
        config.development.debug = False
        config.development.development_mode = False
        assert config.is_production is True
        
        config.development.debug = True
        assert config.is_production is False


class TestClientIdAllocation:
    """Test client ID allocation functions."""
    
    def test_get_client_id_for_service(self):
        """Test service client ID allocation."""
        # Test known services
        assert get_client_id_for_service("account") == 11
        assert get_client_id_for_service("marketdata") == 12
        assert get_client_id_for_service("historical") == 13
        assert get_client_id_for_service("trader") == 0  # Trader must use client ID 0
        assert get_client_id_for_service("strategy") == 15
        
        # Test unknown service
        with pytest.raises(ValueError):
            get_client_id_for_service("unknown")
            
    def test_get_strategy_client_id(self):
        """Test strategy client ID allocation."""
        # Test valid instance IDs
        assert get_strategy_client_id(0) == 15
        assert get_strategy_client_id(5) == 20
        assert get_strategy_client_id(14) == 29
        
        # Test invalid instance IDs
        with pytest.raises(ValueError):
            get_strategy_client_id(-1)
        with pytest.raises(ValueError):
            get_strategy_client_id(15)


class TestEnvironmentIntegration:
    """Test environment variable integration."""
    
    def test_environment_override(self, monkeypatch):
        """Test that environment variables override defaults."""
        # Set environment variables
        monkeypatch.setenv("POSTGRES_USER", "test_user")
        monkeypatch.setenv("TWS_PORT", "7496")
        monkeypatch.setenv("MAX_SUBSCRIPTIONS", "25")
        
        # Reload settings to pick up environment changes
        config = reload_settings()
        
        assert config.database.user == "test_user"
        assert config.tws.port == 7496
        assert config.market_data.max_subscriptions == 25
        
    def test_invalid_environment_values(self, monkeypatch):
        """Test that invalid environment values raise validation errors."""
        # Set invalid environment variable
        monkeypatch.setenv("MAX_SUBSCRIPTIONS", "0")
        
        with pytest.raises(ValidationError):
            reload_settings()
