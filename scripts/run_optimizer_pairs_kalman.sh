#!/bin/bash
# Run the optimizer for the Pairs_Trading_Adaptive_Kalman strategy
#
# Multi-threading support:
# - Auto-detects available CPU cores and uses them for parallel backtesting
# - Override with: WORKERS=8 ./run_optimizer_pairs_kalman.sh
# - Each worker runs a separate backtest in parallel during optimization
# START_DATE=2025-09-04 END_DATE=2025-12-12 WORKERS=16 ./scripts/run_optimizer_pairs_kalman.sh


set -euo pipefail

# Change to project root directory (where compose.yaml is located)
cd "$(dirname "$0")/.." || exit 1

STRATEGY_NAME="Pairs_Trading_Adaptive_Kalman"
# LOOKBACK = raw 5-sec bars to pass on initial warmup
# Need: max(lookback_window) * max(stats_aggregation_seconds) / 5 + min_hedge_lookback
# = 200 * 1800 / 5 + 100000 = 72000 + 100000 = 172000
# Set to 175000 for safety (matching backtest script)
LOOKBACK=175000
TIMEFRAME="5 secs"
INITIAL_CAPITAL=50000

# Number of CPU cores to use for parallel optimization
# Set to number of available cores, or override with WORKERS env var
WORKERS=${WORKERS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}

# Optimization settings
ALGORITHM="random_search"  
OBJECTIVE="total_return"  # sharpe_ratio, sortino_ratio, total_return, profit_factor
MAX_ITERATIONS=500  # With early stopping, will converge when no improvement
BATCH_SIZE=${BATCH_SIZE:-$WORKERS}  # Match worker count for full CPU utilization (default: auto-detect)

# Optional overrides (export before running)
START_DATE=${START_DATE:-""}  # e.g. 2024-08-01
END_DATE=${END_DATE:-""}      # e.g. 2024-12-31
# WORKERS can be set to control CPU cores used (default: auto-detect)


# Selected pairs from potential pairs analysis (24 pairs - NO duplicate tickers)
SYMBOLS=(PG MUSA AMN CORT ACA KTB FITB TMHC D SPXC RTX AEP ANDE LYEL ABR GRPN AHCO MGY AIN ESE ADPT IBEX ACT DUK ADC AFL AMZN PFBC ADEA PTCT DE ENPH ALRM NNOX WELL PBI ACCO WERN ALHC ATKR ANIP UAA O CTRA ETSY SHLS MSCI PECO)
SYMBOLS_CSV="PG,MUSA,AMN,CORT,ACA,KTB,FITB,TMHC,D,SPXC,RTX,AEP,ANDE,LYEL,ABR,GRPN,AHCO,MGY,AIN,ESE,ADPT,IBEX,ACT,DUK,ADC,AFL,AMZN,PFBC,ADEA,PTCT,DE,ENPH,ALRM,NNOX,WELL,PBI,ACCO,WERN,ALHC,ATKR,ANIP,UAA,O,CTRA,ETSY,SHLS,MSCI,PECO"
SYMBOLS_JSON='["PG","MUSA","AMN","CORT","ACA","KTB","FITB","TMHC","D","SPXC","RTX","AEP","ANDE","LYEL","ABR","GRPN","AHCO","MGY","AIN","ESE","ADPT","IBEX","ACT","DUK","ADC","AFL","AMZN","PFBC","ADEA","PTCT","DE","ENPH","ALRM","NNOX","WELL","PBI","ACCO","WERN","ALHC","ATKR","ANIP","UAA","O","CTRA","ETSY","SHLS","MSCI","PECO"]'
PAIRS_JSON='[
  ["PG","MUSA"],
  ["AMN","CORT"],
  ["ACA","KTB"],
  ["FITB","TMHC"],
  ["D","SPXC"],
  ["RTX","AEP"],
  ["ANDE","LYEL"],
  ["ABR","GRPN"],
  ["AHCO","MGY"],
  ["AIN","ESE"],
  ["ADPT","IBEX"],
  ["ACT","DUK"],
  ["ADC","AFL"],
  ["AMZN","PFBC"],
  ["ADEA","PTCT"],
  ["DE","ENPH"],
  ["ALRM","NNOX"],
  ["WELL","PBI"],
  ["ACCO","WERN"],
  ["ALHC","ATKR"],
  ["ANIP","UAA"],
  ["O","CTRA"],
  ["ETSY","SHLS"],
  ["MSCI","PECO"]
]'

# Parameter ranges for optimization
# Format: Lists of values to test for each parameter
# Centered around current backtest values that produced no trades
PARAM_RANGES=$(cat <<JSON
{
  "lookback_window": [100, 150, 200, 250, 300],
  "entry_threshold": [1.5, 1.8, 2.0, 2.2, 2.5, 3.0],
  "exit_threshold": [0.3, 0.5, 0.7, 1.0],
  "max_hold_bars": [50000, 100000, 200000, 300000],
  "stop_loss_zscore": [3.0, 3.5, 4.0, 4.5],
  "cooldown_bars": [360, 540, 720, 900],
  "hedge_refresh_bars": [500000, 750000, 1000000, 1500000],
  "volatility_window": [10, 15, 20, 25],
  "stats_aggregation_seconds": [900, 1200, 1500, 1800],
  "kalman_delta": [0.001, 0.005, 0.01, 0.05, 0.1],
  "kalman_R": [0.01, 0.05, 0.1, 0.2, 0.5]
}
JSON
)

# Additional configuration for optimizer
# Include fixed strategy parameters that don't change during optimization
# Matching backtest script configuration
CONFIG_JSON=$(cat <<JSON
{
  "initial_capital": $INITIAL_CAPITAL,
  "commission_per_share": 0.005,
  "min_commission": 1.0,
  "slippage_ticks": 1,
  "fixed_params": {
    "symbols": $SYMBOLS_JSON,
    "pairs": $PAIRS_JSON,
    "bar_timeframe": "$TIMEFRAME",
    "lookback_periods": $LOOKBACK,
    "position_size": 48,
    "base_pair_notional": 25000,
    "min_pair_notional": 5000,
    "max_pair_notional": 100000,
    "max_portfolio_notional": 300000,
    "min_hedge_lookback": 100000,
    "spread_history_bars": 300,
    "volatility_adaptation_enabled": true,
    "volatility_ema_alpha": 0.2,
    "min_volatility_ratio": 0.75,
    "max_volatility_ratio": 1.5,
    "min_exit_volatility_ratio": 0.8,
    "max_exit_volatility_ratio": 1.3,
    "volatility_positioning_enabled": true,
    "volatility_position_power": 1.0,
    "volatility_stop_multiplier": 2.5,
    "market_close_hour": 16,
    "market_close_minute": 0,
    "close_before_eod_minutes": -60,
    "timezone": "US/Eastern",
    "use_kalman": true,
    "stationarity_checks_enabled": true,
    "adf_pvalue_threshold": 0.05,
    "cointegration_pvalue_threshold": 0.05,
    "stationarity_check_interval": 60,
    "execution_type": "MKT",
    "risk_budget_per_pair": 1000,
    "halflife_weight_bars": 240,
    "max_halflife_bars": 720,
    "require_half_life": true,
    "max_pair_loss_pct": 0.02,
    "cooldown_after_all_exits": true
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
echo "Max Iterations: $MAX_ITERATIONS (random search with early stopping)"
echo "Batch Size:     $BATCH_SIZE (progress updates every $BATCH_SIZE tests)"
echo "Initial Capital: \$$INITIAL_CAPITAL"
echo "CPU Workers:    $WORKERS cores (100% utilization per batch)"
echo ""
echo "Trading Pairs (24 total - 4x expansion, NO duplicate tickers):"
echo "  Original 6: PG/MUSA, AMN/CORT, ACA/KTB, FITB/TMHC, D/SPXC, RTX/AEP"
echo "  +18 new pairs: ANDE/LYEL, ABR/GRPN, AHCO/MGY, AIN/ESE, ADPT/IBEX,"
echo "                ACT/DUK, ADC/AFL, AMZN/PFBC, ADEA/PTCT, DE/ENPH,"
echo "                ALRM/NNOX, WELL/PBI, ACCO/WERN, ALHC/ATKR, ANIP/UAA,"
echo "                O/CTRA, ETSY/SHLS, MSCI/PECO"
echo ""
echo "Optimizing Parameters (centered around current backtest values):"
echo "  • lookback_window:    100-300 spread entries (current: 200)"
echo "  • entry_threshold:    1.5-3.0 z-score (current: 2.0)"
echo "  • exit_threshold:     0.3-1.0 z-score (current: 0.5)"
echo "  • max_hold_bars:      50k-300k raw bars (current: 200k)"
echo "  • stop_loss_zscore:   3.0-4.5 (current: 3.5)"
echo "  • cooldown_bars:      360-900 raw bars (current: 720)"
echo "  • hedge_refresh_bars: 500k-1.5M raw bars (current: 1M)"
echo "  • volatility_window:  10-25 spread entries (current: 15)"
echo "  • stats_aggregation:  900-1800 sec (15-30 min, current: 1800)"
echo "  • kalman_delta:       0.001-0.1 (current: 0.01)"
echo "  • kalman_R:           0.01-0.5 (current: 0.1)"
echo ""
echo "Position Sizing (Notional-Based):"
echo "  • base_pair_notional: \$25,000 (target capital per pair)"
echo "  • min_pair_notional:  \$5,000 (minimum to enter)"
echo "  • max_pair_notional:  \$100,000 (max per pair)"
echo "  • max_portfolio:      \$300,000 (total exposure cap)"
echo "  • spread_history:     300 entries (buffer for calculations)"
echo "  • lookback_periods:   175,000 raw bars (covers 30m aggregations)"
echo ""
echo "Commission Model (IBKR Fixed):"
echo "  • per share:          \$0.005"
echo "  • minimum:            \$1.00"
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
