#!/bin/bash
# Register the Pairs_Trading_Adaptive_Kalman strategy in the strategies table
#
# Data Requirements Calculation:
# - stats_aggregation_seconds: 1800s (30 minutes per spread bar)
# - Required lookback = max(lookback_window, spread_history_bars, min_hedge_lookback)
# - Current: max(30, 40, 40) = 40 bars
# - Total seconds: 40 bars × 1800s = 72,000 seconds
# - Market time: 72,000s ÷ 3600 = 20 hours = ~3 days (at 6.5 hrs/day)
# - TWS requests: ~10 chunks per symbol × 4 symbols = ~40 total requests

set -euo pipefail

# Change to project root directory (where compose.yaml is located)
cd "$(dirname "$0")/.." || exit 1

STRATEGY_ID="pairs_trading_kalman_v1"
STRATEGY_NAME="Pairs_Trading_Adaptive_Kalman"
ENABLE_STRATEGY=false  # flip to true when you're ready to let the live service load it

# Selected pairs from potential pairs analysis
SYMBOLS_JSON='["PG","MUSA","AMN","CORT","ACA","KTB","FITB","TMHC","D","SPXC","RTX","AEP"]'

# Selected pair combinations
PAIRS_JSON='[
  ["PG","MUSA"],
  ["AMN","CORT"],
  ["ACA","KTB"],
  ["FITB","TMHC"],
  ["D","SPXC"],
  ["RTX","AEP"]
]'

# Configure lookback_window (edit this value as needed)
LOOKBACK_WINDOW=200
STATS_AGGREGATION_SECONDS=1800
# Automatically calculate spread_history_bars as lookback_window + 100 for buffer
SPREAD_HISTORY_BARS=300

# Strategy parameters persisted to strategies.params_json
PARAMS_JSON=$(cat <<JSON
{
  "strategy_id": "$STRATEGY_ID",
  "name": "$STRATEGY_NAME",
  "symbols": $SYMBOLS_JSON,
  "pairs": $PAIRS_JSON,
  "bar_timeframe": "5 secs",
  "lookback_window": $LOOKBACK_WINDOW,
  "stats_aggregation_seconds": $STATS_AGGREGATION_SECONDS,
  "spread_history_bars": $SPREAD_HISTORY_BARS,
  "hedge_refresh_bars": 1000000,
  "min_hedge_lookback": 100000,
  "entry_threshold": 2.0,
  "exit_threshold": 0.5,
  "position_size": 48,
  "max_hold_bars": 200000,
  "stop_loss_zscore": 3.5,
  "market_close_hour": 16,
  "market_close_minute": 0,
  "close_before_eod_minutes": -60,
  "cooldown_bars": 720,
  "cooldown_after_all_exits": true,
  "timezone": "US/Eastern",
  "use_kalman": true,
  "kalman_delta": 0.01,
  "kalman_R": 0.1,
  "stationarity_checks_enabled": true,
  "adf_pvalue_threshold": 0.05,
  "cointegration_pvalue_threshold": 0.05,
  "stationarity_check_interval": 60,
  "volatility_adaptation_enabled": true,
  "volatility_window": 15,
  "volatility_ema_alpha": 0.2,
  "min_volatility_ratio": 0.75,
  "max_volatility_ratio": 1.5,
  "min_exit_volatility_ratio": 0.8,
  "max_exit_volatility_ratio": 1.3,
  "execution_type": "MKT",
  "base_pair_notional": 25000,
  "min_pair_notional": 5000,
  "max_pair_notional": 100000,
  "max_portfolio_notional": 300000,
  "volatility_positioning_enabled": true,
  "volatility_position_power": 1.0,
  "risk_budget_per_pair": 1000,
  "halflife_weight_bars": 240,
  "max_halflife_bars": 720,
  "require_half_life": true,
  "max_pair_loss_pct": 0.02,
  "volatility_stop_multiplier": 2.5

}
JSON
)

SQL=$(cat <<SQL
INSERT INTO strategies (strategy_id, name, enabled, params_json, created_at)
VALUES (
  '$STRATEGY_ID',
  '$STRATEGY_NAME',
  $ENABLE_STRATEGY,
  '$PARAMS_JSON',
  NOW()
)
ON CONFLICT (strategy_id) DO UPDATE
SET
  name = EXCLUDED.name,
  enabled = EXCLUDED.enabled,
  params_json = EXCLUDED.params_json;
SQL
)

echo "Registering strategy '$STRATEGY_ID' ($STRATEGY_NAME) ..."
docker compose exec postgres psql -U bot -d trading -c "$SQL"
echo "Done. Use \"SELECT strategy_id, enabled FROM strategies WHERE strategy_id = '$STRATEGY_ID';\" to verify."

