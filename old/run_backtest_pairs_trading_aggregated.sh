#!/bin/bash
# Run the backtester against the Pairs_Trading_Adaptive_Aggregated strategy
#
# Multi-threading support:
# - Auto-detects available CPU cores and uses them for NumPy/Pandas operations
# - Override with: WORKERS=8 ./run_backtest_pairs_trading_aggregated.sh
# - This speeds up statistical calculations (correlations, rolling windows, etc.)

set -euo pipefail

# Change to project root directory (where compose.yaml is located)
cd "$(dirname "$0")/.." || exit 1

STRATEGY_NAME="Pairs_Trading_Adaptive_Aggregated"
LOOKBACK=200
TIMEFRAME="5 secs"
INITIAL_CAPITAL=250000

# Number of CPU cores to use for parallel processing
# Set to number of available cores, or override with WORKERS env var
WORKERS=${WORKERS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}

# Optional overrides (export before running)
START_DATE=${START_DATE:-""}  # e.g. 2024-08-01
END_DATE=${END_DATE:-""}      # e.g. 2024-12-31
# WORKERS can be set to control CPU cores used (default: auto-detect)

SYMBOLS=(AMN CORT FITB TMHC ACT DUK)
SYMBOLS_JSON='["AMN","CORT","FITB","TMHC","ACT","DUK"]'
PAIRS_JSON='[
  ["AMN","CORT"],
  ["FITB","TMHC"],
  ["ACT","DUK"]
]'

PARAMS_JSON=$(cat <<JSON
{
  "symbols": $SYMBOLS_JSON,
  "pairs": $PAIRS_JSON,
  "bar_timeframe": "$TIMEFRAME",
  "lookback_periods": $LOOKBACK,
  "lookback_window": 20,
  "entry_threshold": 1.5,
  "exit_threshold": 0.5,
  "position_size": 100,
  "base_pair_notional": 50000,
  "max_hold_bars": 7200,
  "stop_loss_zscore": 3.0,
  "cooldown_bars": 300,
  "hedge_refresh_bars": 120,
  "min_hedge_lookback": 120,
  "spread_history_bars": 100,
  "volatility_adaptation_enabled": true,
  "volatility_window": 10,
  "stats_aggregation_seconds": 60,
  "market_close_hour": 16,
  "market_close_minute": 0,
  "close_before_eod_minutes": 5,
  "timezone": "US/Eastern"
}
JSON
)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Backtest Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Strategy:      $STRATEGY_NAME"
echo "Timeframe:     $TIMEFRAME"
echo "Initial Capital: \$$INITIAL_CAPITAL"
echo "CPU Workers:   $WORKERS cores (NumPy/Pandas parallel processing)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

CMD=(docker compose exec 
  -e "OMP_NUM_THREADS=$WORKERS"
  -e "OPENBLAS_NUM_THREADS=$WORKERS"
  -e "MKL_NUM_THREADS=$WORKERS"
  -e "VECLIB_MAXIMUM_THREADS=$WORKERS"
  -e "NUMEXPR_NUM_THREADS=$WORKERS"
  backend-api python -m src.services.backtester.main cli
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
