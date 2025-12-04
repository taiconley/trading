#!/bin/bash
# Optimizer command for new pairs from notes.txt (lines 150-162)
# Date range: 10/22/25 - 11/10/25 (note: optimizer uses all available data, filter in DB if needed)

# Change to project root directory (where compose.yaml is located)
cd "$(dirname "$0")/.." || exit 1

# Unique symbols extracted from pairs (sorted alphabetically)
SYMBOLS="ABR,ACA,ACT,ADPT,AHCO,AIN,AMN,ANDE,ANET,AORT,CORT,DUK,ESE,GRPN,IBEX,KTB,LYEL,MGY"

# Pairs configuration
PAIRS='[["AMN","CORT"],["ACA","KTB"],["ANET","AORT"],["ANDE","LYEL"],["ABR","GRPN"],["AHCO","MGY"],["AIN","ESE"],["ADPT","IBEX"],["ACT","DUK"]]'

# Parameter ranges for Bayesian optimization (ranges, not discrete values)
PARAMS='{
  "lookback_window": [120, 180, 240, 300],
  "entry_threshold": [1.5, 2.0, 2.5, 3.0],
  "exit_threshold": [0.3, 0.5, 0.7, 1.0],
  "position_size": [10, 25, 50],
  "max_hold_bars": [360, 720, 1080],
  "stop_loss_zscore": [2.5, 3.0, 3.5, 4.0],
  "cooldown_bars": [30, 60, 90]
}'

# Config with pairs and other fixed settings
CONFIG="{
  \"pairs\": $PAIRS,
  \"population_size\": 20,
  \"elite_size\": 3,
  \"mutation_rate\": 0.15,
  \"crossover_rate\": 0.8,
  \"stationarity_checks_enabled\": true,
  \"adf_pvalue_threshold\": 0.05,
  \"cointegration_pvalue_threshold\": 0.05,
  \"stationarity_check_interval\": 60,
  \"volatility_adaptation_enabled\": true,
  \"min_volatility_ratio\": 0.75,
  \"max_volatility_ratio\": 1.5,
  \"min_exit_volatility_ratio\": 0.8,
  \"max_exit_volatility_ratio\": 1.3,
  \"market_close_hour\": 16,
  \"market_close_minute\": 0,
  \"close_before_eod_minutes\": 5,
  \"timezone\": \"US/Eastern\",
  \"spread_history_bars\": 1000
}"

echo "Running optimizer with:"
echo "  Strategy: Pairs_Trading"
echo "  Symbols: $SYMBOLS"
echo "  Timeframe: 5 secs"
echo "  Algorithm: bayesian"
echo "  Objective: sharpe_ratio"
echo "  Workers: 22"
echo "  Batch size: 1"
echo "  Max iterations: 100 (overnight run)"
echo ""
echo "Pairs: $PAIRS"
echo ""
echo "Parameter ranges:"
echo "  lookback_window: [120, 180, 240, 300]"
echo "  entry_threshold: [1.5, 2.0, 2.5, 3.0]"
echo "  exit_threshold: [0.3, 0.5, 0.7, 1.0]"
echo "  position_size: [10, 25, 50]"
echo "  max_hold_bars: [360, 720, 1080]"
echo "  stop_loss_zscore: [2.5, 3.0, 3.5, 4.0]"
echo "  cooldown_bars: [30, 60, 90]"
echo ""

docker compose exec backend-api python -m src.services.optimizer.main optimize \
  --strategy Pairs_Trading \
  --symbols "$SYMBOLS" \
  --timeframe "5 secs" \
  --lookback 350 \
  --algorithm bayesian \
  --objective sharpe_ratio \
  --max-iterations 3 \
  --workers 22 \
  --batch-size 1 \
  --params "$PARAMS" \
  --config "$CONFIG"

