#!/bin/bash
# Optimize Pairs_Trading_Adaptive strategy WITH pair selection optimization

# Change to project root directory (where compose.yaml is located)
cd "$(dirname "$0")/.." || exit 1

docker compose exec backend-api python -m src.services.optimizer.main optimize \
  --strategy Pairs_Trading_Adaptive \
  --symbols ABR,ACA,ACT,ADPT,AIN,AHCO,AMN,ANDE,ANET,AORT,CORT,DUK,ESE,GRPN,IBEX,KTB,LYEL,MGY \
  --timeframe "5 secs" \
  --lookback 1000 \
  --algorithm bayesian \
  --objective sharpe_ratio \
  --max-iterations 30 \
  --workers 24 \
  --params '{"lookback_window": [120, 180, 240, 300], "entry_threshold": [1.5, 2.0, 2.5, 3.0], "exit_threshold": [0.3, 0.5, 0.7, 1.0], "position_size": [10, 25, 50], "max_hold_bars": [360, 720, 1080], "stop_loss_zscore": [2.5, 3.0, 3.5, 4.0], "cooldown_bars": [30, 60, 90], "pair_selection": {"AMN/CORT": [true, false], "ACA/KTB": [true, false], "ANET/AORT": [true, false], "ANDE/LYEL": [true, false], "ABR/GRPN": [true, false], "AHCO/MGY": [true, false], "AIN/ESE": [true, false], "ADPT/IBEX": [true, false], "ACT/DUK": [true, false]}}' \
  --config '{"pairs": [["AMN","CORT"],["ACA","KTB"],["ANET","AORT"],["ANDE","LYEL"],["ABR","GRPN"],["AHCO","MGY"],["AIN","ESE"],["ADPT","IBEX"],["ACT","DUK"]], "population_size": 20, "elite_size": 3, "mutation_rate": 0.15, "crossover_rate": 0.8, "hedge_refresh_bars": 30, "min_hedge_lookback": 120, "stationarity_checks_enabled": false, "adf_pvalue_threshold": 0.05, "cointegration_pvalue_threshold": 0.05, "stationarity_check_interval": 60, "volatility_adaptation_enabled": true, "volatility_window": 240, "volatility_ema_alpha": 0.2, "min_volatility_ratio": 0.75, "max_volatility_ratio": 1.5, "min_exit_volatility_ratio": 0.8, "max_exit_volatility_ratio": 1.3, "market_close_hour": 16, "market_close_minute": 0, "close_before_eod_minutes": 5, "timezone": "US/Eastern", "spread_history_bars": 1000}'
