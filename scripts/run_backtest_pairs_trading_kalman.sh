#!/bin/bash
# Run the backtester against the Pairs_Trading_Adaptive_Kalman strategy
#
# Multi-threading support:
# - Auto-detects available CPU cores and uses them for NumPy/Pandas operations
# - Override with: WORKERS=8 ./run_backtest_pairs_trading_kalman.sh
# - This speeds up statistical calculations (correlations, rolling windows, etc.)
#  START_DATE=2025-09-10 END_DATE=2025-12-12 WORKERS=8 ./scripts/run_backtest_pairs_trading_kalman.sh

set -euo pipefail

# Change to project root directory (where compose.yaml is located)
cd "$(dirname "$0")/.." || exit 1

STRATEGY_NAME="Pairs_Trading_Adaptive_Kalman"
# LOOKBACK = raw 5-sec bars to pass on initial warmup
# Need: max(lookback_window) * max(stats_aggregation_seconds) / 5 + min_hedge_lookback
# = 200 * 1800 / 5 + 100000 = 72000 + 100000 = 172000
# Set to 175000 for safety
LOOKBACK=175000
TIMEFRAME="5 secs"
INITIAL_CAPITAL=50000

# Number of CPU cores to use for parallel processing
# Set to number of available cores, or override with WORKERS env var
WORKERS=${WORKERS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}

# Optional overrides (export before running)
START_DATE=${START_DATE:-""}  # e.g. 2024-08-01
END_DATE=${END_DATE:-""}      # e.g. 2024-12-31
# WORKERS can be set to control CPU cores used (default: auto-detect)

# Selected pairs from potential pairs analysis (24 pairs - NO duplicate tickers)
SYMBOLS=(PG MUSA AMN CORT ACA KTB FITB TMHC D SPXC RTX AEP ANDE LYEL ABR GRPN AHCO MGY AIN ESE ADPT IBEX ACT DUK ADC AFL AMZN PFBC ADEA PTCT DE ENPH ALRM NNOX WELL PBI ACCO WERN ALHC ATKR ANIP UAA O CTRA ETSY SHLS MSCI PECO)
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

PARAMS_JSON=$(cat <<JSON
{
  "symbols": $SYMBOLS_JSON,
  "pairs": $PAIRS_JSON,
  "bar_timeframe": "$TIMEFRAME",
  "lookback_periods": $LOOKBACK,
  "lookback_window": 200,
  "entry_threshold": 1.9,
  "exit_threshold": 0.5,
  "position_size": 48,
  "base_pair_notional": 25000,
  "min_pair_notional": 5000,
  "max_pair_notional": 100000,
  "max_portfolio_notional": 300000,
  "max_hold_bars": 200000,
  "stop_loss_zscore": 3.5,
  "cooldown_bars": 720,
  "cooldown_after_all_exits": true,
  "hedge_refresh_bars": 1000000,
  "min_hedge_lookback": 100000,
  "spread_history_bars": 300,
  "volatility_adaptation_enabled": true,
  "volatility_window": 15,
  "volatility_ema_alpha": 0.2,
  "min_volatility_ratio": 0.75,
  "max_volatility_ratio": 1.5,
  "min_exit_volatility_ratio": 0.8,
  "max_exit_volatility_ratio": 1.3,
  "volatility_positioning_enabled": true,
  "volatility_position_power": 1.0,
  "volatility_stop_multiplier": 2.5,
  "stats_aggregation_seconds": 1800,
  "market_close_hour": 16,
  "market_close_minute": 0,
  "close_before_eod_minutes": -60,
  "timezone": "US/Eastern",
  "use_kalman": true,
  "kalman_delta": 0.01,
  "kalman_R": 0.1,
  "stationarity_checks_enabled": true,
  "adf_pvalue_threshold": 0.05,
  "cointegration_pvalue_threshold": 0.05,
  "stationarity_check_interval": 60,
  "execution_type": "MKT",
  "risk_budget_per_pair": 1000,
  "halflife_weight_bars": 240,
  "max_halflife_bars": 720,
  "require_half_life": true,
  "max_pair_loss_pct": 0.02
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
echo "Trading Pairs: 24 total (4x expansion, NO duplicate tickers)"
echo "  Original 6: PG/MUSA, AMN/CORT, ACA/KTB, FITB/TMHC, D/SPXC, RTX/AEP"
echo "  +18 new pairs: ANDE/LYEL, ABR/GRPN, AHCO/MGY, AIN/ESE, ADPT/IBEX,"
echo "                ACT/DUK, ADC/AFL, AMZN/PFBC, ADEA/PTCT, DE/ENPH,"
echo "                ALRM/NNOX, WELL/PBI, ACCO/WERN, ALHC/ATKR, ANIP/UAA,"
echo "                O/CTRA, ETSY/SHLS, MSCI/PECO"
echo ""
echo "Key Parameters:"
echo "  • lookback_window:        200 spread entries"
echo "  • entry_threshold:        2.0 z-score"
echo "  • exit_threshold:         0.5 z-score"
echo "  • max_hold_bars:          200,000 raw bars"
echo "  • stop_loss_zscore:       3.5"
echo "  • cooldown_bars:          720 raw bars"
echo "  • hedge_refresh_bars:     1,000,000 raw bars"
echo "  • volatility_window:      15 spread entries"
echo "  • stats_aggregation:      1800 sec (30 min z-score sampling)"
echo "  • kalman_delta:           0.01"
echo "  • kalman_R:               0.1"
echo ""
echo "Position Sizing:"
echo "  • base_pair_notional:     \$25,000"
echo "  • min_pair_notional:      \$5,000"
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

