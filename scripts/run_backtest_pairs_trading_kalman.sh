#!/bin/bash
# Run the backtester against the Pairs_Trading_Adaptive_Kalman strategy
#
# Multi-threading support:
# - Auto-detects available CPU cores and uses them for NumPy/Pandas operations
# - Override with: WORKERS=8 ./run_backtest_pairs_trading_kalman.sh
# - This speeds up statistical calculations (correlations, rolling windows, etc.)

set -euo pipefail

# Change to project root directory (where compose.yaml is located)
cd "$(dirname "$0")/.." || exit 1

STRATEGY_NAME="Pairs_Trading_Adaptive_Kalman"
# LOOKBACK = raw 5-sec bars to pass on initial warmup
# Need: max(lookback_window) * max(stats_aggregation_seconds) / 5 + min_hedge_lookback
# = 40 * 1800 / 5 + 120 = 14400 + 120 = 14520
# Set to 20000 for safety (matching optimizer)
LOOKBACK=15000
TIMEFRAME="5 secs"
INITIAL_CAPITAL=50000

# Number of CPU cores to use for parallel processing
# Set to number of available cores, or override with WORKERS env var
WORKERS=${WORKERS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}

# Optional overrides (export before running)
START_DATE=${START_DATE:-""}  # e.g. 2024-08-01
END_DATE=${END_DATE:-""}      # e.g. 2024-12-31
# WORKERS can be set to control CPU cores used (default: auto-detect)

# Selected pairs based on pair_selection parameter
# ITW/PKG: true
SYMBOLS=(ITW PKG)
SYMBOLS_JSON='["ITW","PKG"]'
PAIRS_JSON='[
  ["ITW","PKG"]
]'

PARAMS_JSON=$(cat <<JSON
{
  "symbols": $SYMBOLS_JSON,
  "pairs": $PAIRS_JSON,
  "bar_timeframe": "$TIMEFRAME",
  "lookback_periods": $LOOKBACK,
  "lookback_window": 40,
  "entry_threshold": 1.8,
  "exit_threshold": 0.7,
  "position_size": 100,
  "base_pair_notional": 50000,
  "min_pair_notional": 10000,
  "max_pair_notional": 100000,
  "max_portfolio_notional": 300000,
  "max_hold_bars": 21600,
  "stop_loss_zscore": 3.0,
  "cooldown_bars": 240,
  "hedge_refresh_bars": 480,
  "min_hedge_lookback": 120,
  "spread_history_bars": 150,
  "volatility_adaptation_enabled": true,
  "volatility_window": 15,
  "stats_aggregation_seconds": 1800,
  "market_close_hour": 16,
  "market_close_minute": 0,
  "close_before_eod_minutes": 5,
  "timezone": "US/Eastern",
  "use_kalman": true,
  "kalman_delta": 0.00001,
  "kalman_R": 0.1,
  "stationarity_checks_enabled": true
}
JSON
)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Backtest Configuration (Kalman Filter)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Strategy:      $STRATEGY_NAME"
echo "Timeframe:     $TIMEFRAME"
echo "Initial Capital: \$$INITIAL_CAPITAL"
echo "CPU Workers:   $WORKERS cores (NumPy/Pandas parallel processing)"
echo ""
echo "Trading Pairs:"
echo "  • ITW/PKG"
echo ""
echo "Key Parameters:"
echo "  • lookback_window:        40 spread entries"
echo "  • entry_threshold:        1.8 z-score"
echo "  • exit_threshold:         0.7 z-score"
echo "  • max_hold_bars:          21,600 (30 hrs raw bars)"
echo "  • stop_loss_zscore:       3.0"
echo "  • cooldown_bars:          240 (20 min raw bars)"
echo "  • hedge_refresh_bars:     480 (40 min raw bars)"
echo "  • volatility_window:      15 spread entries"
echo "  • stats_aggregation:      1800 sec (30 min z-score sampling)"
echo "  • kalman_delta:           0.00001"
echo "  • kalman_R:               0.1"
echo ""
echo "Position Sizing:"
echo "  • base_pair_notional:     \$50,000"
echo "  • min_pair_notional:      \$10,000"
echo "  • max_pair_notional:      \$100,000"
echo "  • max_portfolio:          \$300,000"
if [[ -n "$START_DATE" ]]; then
  echo ""
  echo "Date Range:"
  echo "  Start: $START_DATE"
  [[ -n "$END_DATE" ]] && echo "  End:   $END_DATE"
fi
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

