#!/bin/bash
# Run the optimizer for the Pairs_Trading_Adaptive_Aggregated strategy
#
# Multi-threading support:
# - Auto-detects available CPU cores and uses them for parallel backtesting
# - Override with: WORKERS=8 ./run_optimizer_pairs_aggregated.sh
# - Each worker runs a separate backtest in parallel during optimization
# ALLO and ANGO are not included in the optimizer because they have no data for the date range

set -euo pipefail

# Change to project root directory (where compose.yaml is located)
cd "$(dirname "$0")/.." || exit 1

STRATEGY_NAME="Pairs_Trading_Adaptive_Aggregated"
LOOKBACK=600
TIMEFRAME="5 secs"
INITIAL_CAPITAL=250000

# Number of CPU cores to use for parallel optimization
# Set to number of available cores, or override with WORKERS env var
WORKERS=${WORKERS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}

# Optimization settings
ALGORITHM="bayesian"  # bayesian, grid_search, random_search, genetic
OBJECTIVE="sharpe_ratio"  # sharpe_ratio, sortino_ratio, total_return, profit_factor
MAX_ITERATIONS=50  # With early stopping, will exit when converged (typically 20-35 iterations)
BATCH_SIZE=${BATCH_SIZE:-$WORKERS}  # Match worker count for full CPU utilization (default: auto-detect)

# Optional overrides (export before running)
START_DATE=${START_DATE:-""}  # e.g. 2024-08-01
END_DATE=${END_DATE:-""}      # e.g. 2024-12-31
# WORKERS can be set to control CPU cores used (default: auto-detect)

SYMBOLS=(AIN ESE AHCO SIBN CTRA ORA HUM CINF ENPH DOW DE CME D JXN)
SYMBOLS_CSV="AIN,ESE,AHCO,SIBN,CTRA,ORA,HUM,CINF,ENPH,DOW,DE,CME,D,JXN"
SYMBOLS_JSON='["AIN","ESE","AHCO","SIBN","CTRA","ORA","HUM","CINF","ENPH","DOW","DE","CME","D","JXN"]'
PAIRS_JSON='[
  ["AIN","ESE"],
  ["AHCO","SIBN"],
  ["CTRA","ORA"],
  ["HUM","CINF"],
  ["ENPH","DOW"],
  ["DE","CME"],
  ["D","JXN"]
]'

# Parameter ranges for optimization
# Format: Lists of values to test for each parameter
# CONSERVATIVE SETTINGS: Trade less, trade bigger, higher quality signals
PARAM_RANGES=$(cat <<JSON
{
  "lookback_window": [360, 420, 480, 600, 720],
  "entry_threshold": [2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
  "exit_threshold": [0.2, 0.3, 0.4, 0.5],
  "max_hold_bars": [1200, 1800, 2400, 3600],
  "stop_loss_zscore": [3.0, 3.5, 4.0, 4.5],
  "cooldown_bars": [120, 180, 240, 300],
  "hedge_refresh_bars": [60, 90, 120],
  "volatility_window": [360, 420, 480, 600],
  "pair_selection": {
    "AIN/ESE": [true, false],
    "AHCO/SIBN": [true, false],
    "CTRA/ORA": [true, false],
    "HUM/CINF": [true, false],
    "ENPH/DOW": [true, false],
    "DE/CME": [true, false],
    "D/JXN": [true, false]
  }
}
JSON
)

# Additional configuration for optimizer
# Include fixed strategy parameters that don't change during optimization
# REALISTIC COMMISSIONS: IBKR Tiered pricing + bigger positions to avoid minimum commission drag
CONFIG_JSON=$(cat <<JSON
{
  "initial_capital": $INITIAL_CAPITAL,
  "commission_per_share": 0.0035,
  "min_commission": 0.35,
  "slippage_ticks": 2,
  "fixed_params": {
    "symbols": $SYMBOLS_JSON,
    "pairs": $PAIRS_JSON,
    "bar_timeframe": "$TIMEFRAME",
    "lookback_periods": $LOOKBACK,
    "position_size": 500,
    "base_pair_notional": 100000,
    "min_hedge_lookback": 120,
    "spread_history_bars": 1500,
    "volatility_adaptation_enabled": true,
    "stats_aggregation_seconds": 300,
    "market_close_hour": 16,
    "market_close_minute": 0,
    "close_before_eod_minutes": 5,
    "timezone": "US/Eastern"
  }
}
JSON
)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Optimization Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Strategy:       $STRATEGY_NAME"
echo "Algorithm:      $ALGORITHM"
echo "Objective:      $OBJECTIVE"
echo "Timeframe:      $TIMEFRAME"
echo "Max Iterations: $MAX_ITERATIONS (with early stopping)"
echo "Batch Size:     $BATCH_SIZE (progress updates every $BATCH_SIZE tests)"
echo "Initial Capital: \$$INITIAL_CAPITAL"
echo "CPU Workers:    $WORKERS cores (100% utilization per batch)"
echo ""
echo "Optimizing Parameters (CONSERVATIVE - Quality over Quantity):"
echo "  • entry_threshold:    2.5 - 5.0 (wait for extreme deviations)"
echo "  • exit_threshold:     0.2 - 0.5 (take profits near mean)"
echo "  • lookback_window:    360 - 720 (longer history, more stable)"
echo "  • max_hold_bars:      1200 - 3600 (1-5 hours)"
echo "  • stop_loss_zscore:   3.0 - 4.5 (wider stops)"
echo "  • cooldown_bars:      120 - 300 (10-25 min between trades)"
echo "  • hedge_refresh_bars: 60 - 120 (refresh less often)"
echo "  • volatility_window:  360 - 600 (longer volatility calc)"
echo "  • pair_selection:     Enable/disable each of 7 pairs"
echo ""
echo "Position Sizing:"
echo "  • position_size:      500 shares (10x bigger to beat min commission)"
echo "  • base_pair_notional: \$100,000 (4x bigger for better scaling)"
echo "  • stats_aggregation:  300 seconds (5-min bars for stability)"
echo ""
echo "Commission Model (IBKR Tiered):"
echo "  • per share:          \$0.0035 (vs \$0.005 Fixed)"
echo "  • minimum:            \$0.35 (vs \$1.00 Fixed)"
echo "  • 500 shares =        \$1.75 (real cost, no minimum bump)"
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
  backend-api python -m src.services.optimizer.main optimize
  --strategy "$STRATEGY_NAME"
  --symbols "$SYMBOLS_CSV"
  --timeframe "$TIMEFRAME"
  --lookback "$LOOKBACK"
  --params "$PARAM_RANGES"
  --algorithm "$ALGORITHM"
  --objective "$OBJECTIVE"
  --workers "$WORKERS"
  --max-iterations "$MAX_ITERATIONS"
  --batch-size "$BATCH_SIZE"
  --config "$CONFIG_JSON"
)

# Add date filters if provided
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

