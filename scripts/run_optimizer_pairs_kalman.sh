#!/bin/bash
# Run the optimizer for the Pairs_Trading_Adaptive_Kalman strategy
#
# Multi-threading support:
# - Auto-detects available CPU cores and uses them for parallel backtesting
# - Override with: WORKERS=8 ./run_optimizer_pairs_kalman.sh
# - Each worker runs a separate backtest in parallel during optimization

set -euo pipefail

# Change to project root directory (where compose.yaml is located)
cd "$(dirname "$0")/.." || exit 1

STRATEGY_NAME="Pairs_Trading_Adaptive_Kalman"
# LOOKBACK = raw 5-sec bars to pass on initial warmup
# Need: max(lookback_window) * max(stats_aggregation_seconds) / 5 + min_hedge_lookback
# = 50 * 1800 / 5 + 120 = 18,000 + 120 = 18,120
# Set to 20,000 for safety
LOOKBACK=20000
TIMEFRAME="5 secs"
INITIAL_CAPITAL=250000

# Number of CPU cores to use for parallel optimization
# Set to number of available cores, or override with WORKERS env var
WORKERS=${WORKERS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}

# Optimization settings
ALGORITHM="random_search"  
OBJECTIVE="total_return"  # sharpe_ratio, sortino_ratio, total_return, profit_factor
MAX_ITERATIONS=2500  # With early stopping, will converge when no improvement
BATCH_SIZE=${BATCH_SIZE:-$WORKERS}  # Match worker count for full CPU utilization (default: auto-detect)

# Optional overrides (export before running)
START_DATE=${START_DATE:-""}  # e.g. 2024-08-01
END_DATE=${END_DATE:-""}      # e.g. 2024-12-31
# WORKERS can be set to control CPU cores used (default: auto-detect)

# Multi-pair optimization: AMN/CORT, FITB/TMHC, ACT/DUK
# Optimizer will test each pair on/off via boolean flags
# SYMBOLS=(AMN CORT FITB TMHC ACT DUK)
# SYMBOLS_CSV="AMN,CORT,FITB,TMHC,ACT,DUK"
# SYMBOLS_JSON='["AMN","CORT","FITB","TMHC","ACT","DUK"]'
# PAIRS_JSON='[
#   ["AMN","CORT"],
#   ["FITB","TMHC"],
#   ["ACT","DUK"]
SYMBOLS=(ITW PKG)
SYMBOLS_CSV="ITW,PKG"
SYMBOLS_JSON='["ITW","PKG"]'
PAIRS_JSON='[
  ["ITW","PKG"]
]'

# Parameter ranges for optimization
# Format: Lists of values to test for each parameter
# Kalman-specific parameters: kalman_delta, kalman_R
PARAM_RANGES=$(cat <<JSON
{
  "lookback_window": [10, 20, 30, 40, 50],
  "entry_threshold": [1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5],
  "exit_threshold": [0.3, 0.5, 0.7, 1.0, 1.2],
  "max_hold_bars": [5400, 7200, 10800, 14400, 21600, 28800],
  "stop_loss_zscore": [2.5, 3.0, 3.5],
  "cooldown_bars": [180, 240, 300, 600, 900],
  "hedge_refresh_bars": [90, 120, 180, 240, 360, 480],
  "volatility_window": [8, 10, 15, 20, 25, 30],
  "stats_aggregation_seconds": [60, 120, 180, 300, 600, 900, 1800],
  "kalman_delta": [1e-5, 1e-4, 1e-3, 1e-2],
  "kalman_R": [1e-4, 1e-3, 1e-2, 1e-1],
  "pair_selection": {
    "ITW/PKG": [true, false]
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
    "position_size": 100,
    "base_pair_notional": 50000,
    "min_pair_notional": 5000,
    "max_pair_notional": 100000,
    "max_portfolio_notional": 300000,
    "min_hedge_lookback": 120,
    "spread_history_bars": 150,
    "volatility_adaptation_enabled": true,
    "market_close_hour": 16,
    "market_close_minute": 0,
    "close_before_eod_minutes": 5,
    "timezone": "US/Eastern",
    "use_kalman": true
  }
}
JSON
)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Optimization Configuration (Kalman Filter)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Strategy:       $STRATEGY_NAME"
echo "Algorithm:      $ALGORITHM"
echo "Objective:      $OBJECTIVE"
echo "Timeframe:      $TIMEFRAME"
echo "Max Iterations: $MAX_ITERATIONS (bayesian with early stopping)"
echo "Batch Size:     $BATCH_SIZE (progress updates every $BATCH_SIZE tests)"
echo "Initial Capital: \$$INITIAL_CAPITAL"
echo "CPU Workers:    $WORKERS cores (100% utilization per batch)"
echo ""
echo "Optimizing Parameters (INCLUDES WORKING BASELINE + EXPLORATION):"
echo "  • stats_aggregation:  60 - 1800 sec (1m - 30m z-score sampling) 🔑 KEY!"
echo "  • lookback_window:    10 - 50 SPREAD entries"
echo "  • volatility_window:  8 - 30 SPREAD entries"
echo "  • entry_threshold:    1.2 - 3.5 z-score (includes working 1.5)"
echo "  • exit_threshold:     0.3 - 1.2"
echo "  • max_hold_bars:      5,400 - 28,800 (7.5 hrs - 40 hrs RAW bars)"
echo "  • stop_loss_zscore:   2.5 - 5.0"
echo "  • cooldown_bars:      180 - 900 (15 min - 75 min RAW bars)"
echo "  • hedge_refresh_bars: 90 - 480 (7.5 min - 40 min RAW bars)"
echo "  • kalman_delta:       1e-5 - 1e-2"
echo "  • kalman_R:           1e-4 - 1e-1"
echo "  • pair_selection:     AMN/CORT, FITB/TMHC, ACT/DUK (8 combos)"
echo ""
echo "Position Sizing (Notional-Based):"
echo "  • base_pair_notional: \$50,000 (target capital per pair)"
echo "  • min_pair_notional:  \$5,000 (minimum to enter)"
echo "  • max_pair_notional:  \$100,000 (max per pair)"
echo "  • max_portfolio:      \$300,000 (total exposure cap)"
echo "  • spread_history:     150 entries (buffer for calculations)"
echo "  • lookback_periods:   20,000 raw bars (covers up to 30m aggregations)"
echo ""
echo "Parameter Ranges Being Tested:"
echo "  • lookback_window:    10-50 spread entries"
echo "  • entry_threshold:    1.2-3.5 z-score"
echo "  • exit_threshold:     0.3-1.2 z-score"
echo "  • max_hold_bars:      5,400-28,800 raw bars (7.5 hrs - 40 hrs)"
echo "  • stop_loss_zscore:   2.5-5.0"
echo "  • cooldown_bars:      180-900 raw bars (15 min - 75 min)"
echo "  • hedge_refresh_bars: 90-480 raw bars (7.5 min - 40 min)"
echo "  • volatility_window:  8-30 spread entries"
echo "  • kalman_delta:       1e-5 - 1e-2"
echo "  • kalman_R:           1e-4 - 1e-1"
echo "  • pair_selection:     8 combinations (2^3) of AMN/CORT, FITB/TMHC, ACT/DUK"
echo ""
echo "Z-Score Aggregation Periods Being Tested:"
echo "  • 60 sec   (1 min)   - Fast intraday (WORKING BASELINE)"
echo "  • 120 sec  (2 min)   - Slightly smoother"
echo "  • 180 sec  (3 min)   - Balanced"
echo "  • 300 sec  (5 min)   - Medium-term"
echo "  • 600 sec  (10 min)  - Slower swing"
echo "  • 900 sec  (15 min)  - Conservative"
echo "  • 1800 sec (30 min)  - Long-cycle intraday"
echo ""
echo "Commission Model (IBKR Tiered):"
echo "  • per share:          \$0.0035 (vs \$0.005 Fixed)"
echo "  • minimum:            \$0.35 (vs \$1.00 Fixed)"
echo "  • \$50k notional ≈     \$3-5 per trade (varies by price/hedge)"
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
