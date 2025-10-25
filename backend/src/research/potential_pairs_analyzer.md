# Potential Pairs Analyzer

Offline research tool that screens equity pairs using the historical candles already stored in Postgres. It ranks pairs for the `Pairs_Trading` strategy by combining statistical tests with a lightweight spread-level simulation, then saves the results to the `potential_pairs` table.

## Running the Analyzer
- Enter the backend container and run the module directly:
```
docker compose exec backend-api python -m src.research.potential_pairs_analyzer \
  --timeframe "5 secs" \
  --lookback-days 5 \
  --min-bars 3000 \
  --min-dollar-volume 200000 \
  --parallelism 8 \
  --replace-existing
```
- Use `--help` to see all options without running an analysis:
```
docker compose exec backend-api python -m src.research.potential_pairs_analyzer --help
```
- By default the tool:
  - Reads *active* symbols from the `symbols` table.
  - Filters to candles that match the requested timeframe (default `"5 secs"`).
  - Limits the window to the last `--lookback-days` (unless `--start-date`/`--end-date` override it).
  - Inserts results into `potential_pairs` with a `status` value of `"candidate"` unless you change it.

## Key Flags
- `--timeframe`: Candles timeframe to evaluate (must match `candles.tf`). Default `"5 secs"`.
- `--lookback-days`: Rolling window length when explicit start/end dates are not supplied. Default `5`.
- `--start-date` / `--end-date`: UTC ISO timestamps (e.g. `2024-01-01T13:30:00Z`) limiting the analysis window.
- `--min-bars`: Minimum overlapping bars for a pair to be scored. Default `2500`.
- `--min-dollar-volume`: Filters symbols whose average `(close * volume)` per bar is below the threshold. Helps enforce tradeability.
- `--max-symbols`: Upper bound on how many symbols survive the liquidity screen (take the most liquid first).
- `--max-pairs`: Optional hard cap on the combinations examined (after liquidity filtering).
- `--entry-z` / `--exit-z`: Thresholds used in the spread simulation. Defaults mirror the live strategy (2.0/0.5) but you can experiment.
- `--min-trades`: Minimum closed trades the simulation must produce for a pair to qualify. Default `5`.
- `--parallelism`: Number of worker threads for pair evaluation (capped at 26). Default scales with CPU count.
- `--replace-existing`: Delete existing `potential_pairs` rows for the timeframe/window before inserting new ones.
- `--status`: Value to insert in `potential_pairs.status` (`candidate`, `validated`, or `rejected`). Default `candidate`.
- `--verbose`: Enable debug-level logging.

## What the Analyzer Measures
1. **Symbol Liquidity Screen**
   - Counts bars and average dollar volume for each active symbol in the requested window.
   - Keeps the highest-liquidity symbols that satisfy `--min-bars` and `--min-dollar-volume` (up to `--max-symbols`).
2. **Pair Construction**
   - Loads close/volume series for each surviving symbol.
   - Aligns timestamps and drops non-overlapping periods; pairs with fewer than `--min-bars` shared observations are skipped.
3. **Statistical Tests**
   - Ordinary Least Squares regression on log prices to estimate hedge ratio/intercept.
   - Spread mean/standard deviation and z-score series.
   - Augmented Dickey–Fuller (ADF) test on the spread (`adf_pvalue`).
   - Engle–Granger cointegration test (`coint_pvalue`).
   - Half-life estimate (mean-reversion speed) using spread autocorrelation.
4. **Simulation Metrics**
   - Runs a simple z-score-driven backtest using the supplied entry/exit thresholds.
   - Records trade count, win rate, total P&L (relative), Sharpe approximation, profit factor, drawdown, and holding time.
   - Stores the raw spread trade P&Ls and equity curve in `potential_pairs.meta` for deeper inspection.

Pairs that pass every filter are inserted into `potential_pairs` with all of these statistics, giving you a short list to promote into full backtests or optimization cycles.

## Interpreting the Results
- **adf_pvalue / coint_pvalue**: Lower is better; values < 0.05 indicate stronger evidence of stationarity/cointegration.
- **half_life_minutes**: Shorter half-life means quicker mean reversion. Compare against your max holding window.
- **spread_std**: Displays the spread volatility; extremely small values may indicate thin movement, while very large values can lead to wild z-scores.
- **pair_sharpe / pair_profit_factor / pair_win_rate**: Derived from the z-score simulation to give a sanity check before deeper testing.
- **pair_max_drawdown**: Worst simulated drawdown in spread units; use it alongside half-life to gauge risk.
- **meta.total_pnl / trade_pnls**: Helpful for manual audits—plot them if a pair looks too good or too bad.

Use these metrics to shortlist candidates, then hand the winners to the backtester/optimizer for full lifecycle evaluation.

