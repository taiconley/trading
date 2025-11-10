"""
Pairs Trading Strategy Configuration

This is the SINGLE SOURCE OF TRUTH for pairs trading configuration.
Modify the pairs and parameters here, then:
1. Run `update_pairs_trading_params.py` to sync to database
2. Strategy service will automatically reload from database

To change pairs: Edit the `PAIRS` list below.
"""

# Pairs definition - list of [stock_a, stock_b] pairs
# Modify this list to change which pairs are tracked
PAIRS = [
    ["AMN", "CORT"],   
    ["ACA", "KTB"],     
    ["ANET", "AORT"],           
]



# Complete strategy configuration
# Optimized parameters from genetic algorithm (Run ID: 33, Sharpe: 0.7863)
PAIRS_TRADING_CONFIG = {
    "pairs": PAIRS,
    "lookback_periods": 250,        # Number of bars to fetch for strategy
    "lookback_window": 120,         # Optimized: 240 → 120
    "entry_threshold": 1.5,         # Optimized: 2.0 → 1.5
    "exit_threshold": 1.0,          # Optimized: 0.5 → 1.0
    "position_size": 25,            # Optimized: 100 → 25
    "max_hold_bars": 360,           # Optimized: 720 → 360
    "stop_loss_zscore": 4.0,        # Optimized: 3.0 → 4.0
    "market_close_hour": 16,
    "market_close_minute": 0,
    "close_before_eod_minutes": 5,
    "cooldown_bars": 90,            # Optimized: 60 → 90
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
    "max_exit_volatility_ratio": 1.3,
    # Order execution settings
    "execution_type": "ADAPTIVE",  # Options: "MKT", "LMT", "ADAPTIVE", "PEG BEST", "PEG MID"
    "adaptive_priority": "Normal"  # Options: "Patient", "Normal", "Urgent" (for ADAPTIVE orders)
}

