#!/bin/bash
# Run the backtester against the Pairs_Trading_Adaptive_Aggregated strategy

set -euo pipefail

# Change to project root directory (where compose.yaml is located)
cd "$(dirname "$0")/.." || exit 1

STRATEGY_NAME="Pairs_Trading_Adaptive_Aggregated"
LOOKBACK=600
TIMEFRAME="5 secs"
INITIAL_CAPITAL=250000

# Optional overrides (export before running)
START_DATE=${START_DATE:-""}  # e.g. 2024-08-01
END_DATE=${END_DATE:-""}      # e.g. 2024-12-31

SYMBOLS=(AIN ESE AHCO SIBN ALLO ANGO CTRA ORA HUM CINF ENPH DOW DE CME D JXN)
SYMBOLS_JSON='["AIN","ESE","AHCO","SIBN","ALLO","ANGO","CTRA","ORA","HUM","CINF","ENPH","DOW","DE","CME","D","JXN"]'
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

PARAMS_JSON=$(cat <<JSON
{
  "symbols": $SYMBOLS_JSON,
  "pairs": $PAIRS_JSON,
  "bar_timeframe": "$TIMEFRAME",
  "lookback_periods": $LOOKBACK,
  "lookback_window": 240,
  "entry_threshold": 1.8,
  "exit_threshold": 0.4,
  "position_size": 50,
  "base_pair_notional": 25000,
  "max_hold_bars": 900,
  "stop_loss_zscore": 3.0,
  "cooldown_bars": 60,
  "hedge_refresh_bars": 30,
  "min_hedge_lookback": 120,
  "spread_history_bars": 1200,
  "volatility_adaptation_enabled": true,
  "volatility_window": 240,
  "stats_aggregation_seconds": 60,
  "market_close_hour": 16,
  "market_close_minute": 0,
  "close_before_eod_minutes": 5,
  "timezone": "US/Eastern"
}
JSON
)

echo "Running backtest for $STRATEGY_NAME ..."
CMD=(docker compose exec backend-api python -m src.services.backtester.main cli
  --strategy "$STRATEGY_NAME"
  --timeframe "$TIMEFRAME"
  --lookback "$LOOKBACK"
  --initial-capital "$INITIAL_CAPITAL"
  --params "$PARAMS_JSON"
)

CMD+=(--symbols "${SYMBOLS[@]}")

if [[ -n "$START_DATE" ]]; then
  CMD+=(--start-date "$START_DATE")
fi

if [[ -n "$END_DATE" ]]; then
  CMD+=(--end-date "$END_DATE")
fi

echo ""
echo "Command:"
printf ' %q' "${CMD[@]}"
echo ""
echo ""

"${CMD[@]}"
