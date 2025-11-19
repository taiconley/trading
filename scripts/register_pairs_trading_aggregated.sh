#!/bin/bash
# Register the Pairs_Trading_Adaptive_Aggregated strategy in the strategies table

set -euo pipefail

# Change to project root directory (where compose.yaml is located)
cd "$(dirname "$0")/.." || exit 1

STRATEGY_ID="pairs_trading_agg_stats_v1"
STRATEGY_NAME="Pairs_Trading_Adaptive_Aggregated"
ENABLE_STRATEGY=false  # flip to true when you're ready to let the live service load it

# Unique symbols extracted from potential_pairs.md
SYMBOLS_JSON='["AIN","ESE","AHCO","SIBN","ALLO","ANGO","CTRA","ORA","HUM","CINF","ENPH","DOW","DE","CME","D","JXN"]'

# Candidate pairs sourced from potential_pairs.md
PAIRS_JSON='[
  ["AIN","ESE"],
  ["AHCO","SIBN"],
  ["ALLO","ANGO"],
  ["CTRA","ORA"],
  ["HUM","CINF"],
  ["ENPH","DOW"],
  ["DE","CME"],
  ["D","JXN"]
]'

# Strategy parameters persisted to strategies.params_json
PARAMS_JSON=$(cat <<JSON
{
  "strategy_id": "$STRATEGY_ID",
  "name": "$STRATEGY_NAME",
  "symbols": $SYMBOLS_JSON,
  "pairs": $PAIRS_JSON,
  "bar_timeframe": "5 secs",
  "lookback_periods": 3600,
  "lookback_window": 120,
  "entry_threshold": 2.7,
  "exit_threshold": 0.4,
  "position_size": 500,
  "base_pair_notional": 100000,
  "min_pair_notional": 5000,
  "max_pair_notional": 100000,
  "max_portfolio_notional": 300000,
  "max_hold_bars": 1800,
  "stop_loss_zscore": 3.5,
  "cooldown_bars": 180,
  "hedge_refresh_bars": 90,
  "min_hedge_lookback": 120,
  "spread_history_bars": 1200,
  "volatility_adaptation_enabled": true,
  "volatility_window": 240,
  "volatility_ema_alpha": 0.2,
  "min_volatility_ratio": 0.75,
  "max_volatility_ratio": 1.5,
  "min_exit_volatility_ratio": 0.8,
  "max_exit_volatility_ratio": 1.3,
  "stats_aggregation_seconds": 60,
  "market_close_hour": 16,
  "market_close_minute": 0,
  "close_before_eod_minutes": 5,
  "timezone": "US/Eastern"
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
