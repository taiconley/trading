#!/bin/bash
# Run the optimizer for the Pairs_Trading_Adaptive_Aggregated strategy
#
# Multi-threading support:
# - Auto-detects available CPU cores and uses them for parallel backtesting
# - Override with: WORKERS=8 ./run_optimizer_pairs_aggregated.sh
# - Each worker runs a separate backtest in parallel during optimization
# New pairs from daily bar cointegration analysis
# These pairs show 6-9 day mean-reversion cycles on daily bars
# Optimization will test if they also mean-revert intraday on 5-sec bars

set -euo pipefail

# Change to project root directory (where compose.yaml is located)
cd "$(dirname "$0")/.." || exit 1

STRATEGY_NAME="Pairs_Trading_Adaptive_Aggregated"
# LOOKBACK = raw 5-sec bars to pass on initial warmup
# Need: max(lookback_window) * max(stats_aggregation_seconds) / 5 + min_hedge_lookback
# = 40 * 600 / 5 + 120 = 4,800 + 120 = 4,920
# Set to 5,000 for safety
LOOKBACK=5000
TIMEFRAME="5 secs"
INITIAL_CAPITAL=250000

# Number of CPU cores to use for parallel optimization
# Set to number of available cores, or override with WORKERS env var
WORKERS=${WORKERS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}

# Optimization settings
ALGORITHM="bayesian"  # bayesian for intelligent parameter search
OBJECTIVE="sharpe_ratio"  # sharpe_ratio, sortino_ratio, total_return, profit_factor
MAX_ITERATIONS=50  # With early stopping, will exit when converged
BATCH_SIZE=${BATCH_SIZE:-$WORKERS}  # Match worker count for full CPU utilization (default: auto-detect)

# Optional overrides (export before running)
START_DATE=${START_DATE:-""}  # e.g. 2024-08-01
END_DATE=${END_DATE:-""}      # e.g. 2024-12-31
# WORKERS can be set to control CPU cores used (default: auto-detect)

# New pairs from daily bar analysis (Top 3 from cointegration study)
# AMN/CORT: Healthcare sector (Sharpe 4.03, 7.5d half-life, p=0.000045)
# FITB/TMHC: Housing market (Sharpe 3.67, 6.3d half-life, p=0.001225)
# ACT/DUK: Defensive stocks (Sharpe 4.08, 9.4d half-life, p=0.000994)
SYMBOLS=(AMN CORT FITB TMHC ACT DUK)
SYMBOLS_CSV="AMN,CORT,FITB,TMHC,ACT,DUK"
SYMBOLS_JSON='["AMN","CORT","FITB","TMHC","ACT","DUK"]'
PAIRS_JSON='[
  ["AMN","CORT"],
  ["FITB","TMHC"],
  ["ACT","DUK"]
]'

# Parameter ranges for optimization
# Format: Lists of values to test for each parameter
# LONG-CYCLE SETTINGS: These pairs revert over 6-9 DAYS on daily bars
# Testing if they also show intraday mean-reversion patterns on 5-sec bars
# KEY PARAMETER: stats_aggregation_seconds - determines z-score calculation timeframe
# With 7 days × 6.5 hrs/day × 720 bars/hr = ~31,000 bars available:
# Max spreads at 3600s agg: 31,000 / 720 = 43 spreads
# Safe lookback_window max: 40 spreads (works for all aggregations)
PARAM_RANGES=$(cat <<JSON
{
  "lookback_window": [10, 15, 20, 25, 30, 40],
  "entry_threshold": [1.5, 1.8, 2.0, 2.2, 2.5, 3.0],
  "exit_threshold": [0.2, 0.3, 0.5, 0.7, 1.0],
  "max_hold_bars": [3600, 5400, 7200, 10800, 14400],
  "stop_loss_zscore": [2.5, 3.0, 3.5, 4.0],
  "cooldown_bars": [120, 180, 240, 300, 420, 600],
  "hedge_refresh_bars": [60, 90, 120, 180, 240],
  "volatility_window": [5, 8, 10, 12, 15, 20],
  "stats_aggregation_seconds": [60, 120, 300, 600, 1200, 1800, 3600],
  "pair_selection": {
    "AMN/CORT": [true, false],
    "FITB/TMHC": [true, false],
    "ACT/DUK": [true, false]
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
    "spread_history_bars": 100,
    "volatility_adaptation_enabled": true,
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
echo "Optimizing Parameters (LONG-CYCLE - Multi-day mean reversion):"
echo "  • stats_aggregation:  60 - 3600 sec (1m - 1hr for z-score calc) 🔑 KEY!"
echo "  • lookback_window:    20 - 40 SPREAD entries (safe for all aggs)"
echo "  •   @ 60s agg:        20min - 40min of history"
echo "  •   @ 300s agg:       1.7hrs - 3.3hrs of history"
echo "  •   @ 3600s agg:      0.8days - 1.7days of history"
echo "  • volatility_window:  10 - 20 SPREAD entries"
echo "  • entry_threshold:    1.5 - 3.0 (lower for intraday noise)"
echo "  • exit_threshold:     0.3 - 1.0 (wider exit range)"
echo "  • max_hold_bars:      3600 - 21600 (5 hours - 3 days RAW bars)"
echo "  • stop_loss_zscore:   3.0 - 5.0 (very wide stops)"
echo "  • cooldown_bars:      300 - 1200 (25-100 min RAW bars)"
echo "  • hedge_refresh_bars: 120 - 360 (10-30 min RAW bars)"
echo "  • pair_selection:     Enable/disable each of 3 pairs"
echo ""
echo "Position Sizing (Notional-Based):"
echo "  • base_pair_notional: \$50,000 (target capital per pair)"
echo "  • min_pair_notional:  \$5,000 (minimum to enter)"
echo "  • max_pair_notional:  \$100,000 (max per pair)"
echo "  • max_portfolio:      \$300,000 (total exposure cap)"
echo "  • spread_history:     100 entries (buffer for calculations)"
echo "  • lookback_periods:   5,000 raw bars (sufficient for all aggregations)"
echo ""
echo "Parameter Ranges Being Tested:"
echo "  • lookback_window:    10-40 spread entries"
echo "  • entry_threshold:    1.2-2.5 z-score"
echo "  • exit_threshold:     0.2-1.0 z-score"
echo "  • max_hold_bars:      3600-14400 raw bars (5 hrs - 20 hrs)"
echo "  • stop_loss_zscore:   2.5-4.0"
echo "  • cooldown_bars:      120-600 raw bars (10 min - 50 min)"
echo "  • hedge_refresh_bars: 60-240 raw bars (5 min - 20 min)"
echo "  • volatility_window:  5-20 spread entries"
echo ""
echo "Z-Score Aggregation Periods Being Tested:"
echo "  • 60 sec  (1 min)   - Fast reaction"
echo "  • 120 sec (2 min)   - Moderate"
echo "  • 180 sec (3 min)   - Balanced"
echo "  • 300 sec (5 min)   - Smoother"
echo "  • 600 sec (10 min)  - Slow mean reversion"
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

