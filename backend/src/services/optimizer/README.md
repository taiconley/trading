# Strategy Parameter Optimizer

Automated parameter optimization for trading strategies using various search algorithms.

## Features

### Phase 1 (Current Implementation) ✅
- **Grid Search**: Exhaustive search over parameter grid
- **Random Search**: Random sampling of parameter space
- **Parallel Execution**: Multi-core backtest execution
- **Multiple Objectives**: Sharpe ratio, total return, profit factor, win rate
- **Constraint Handling**: Define parameter constraints (e.g., short_period < long_period)
- **Database Storage**: Full optimization history tracking
- **CLI Interface**: Command-line optimization tool
- **REST API**: Web-triggered optimizations

### Coming Soon
- **Phase 2**: Bayesian optimization, walk-forward analysis, out-of-sample validation
- **Phase 3**: Parameter sensitivity analysis, Pareto frontiers, advanced analytics

## Quick Start

### CLI Usage

Optimize SMA Crossover strategy parameters:

```bash
cd /app/src/services/optimizer

# Grid Search - Test all combinations
python main.py optimize \
  --strategy SMA_Crossover \
  --symbols AAPL \
  --timeframe "1 day" \
  --lookback 365 \
  --params '{"short_period": [5,10,15,20], "long_period": [30,40,50,60]}' \
  --algorithm grid_search \
  --objective sharpe_ratio \
  --constraints "short_period < long_period"

# Random Search - Test 50 random combinations
python main.py optimize \
  --strategy SMA_Crossover \
  --symbols AAPL \
  --timeframe "1 day" \
  --lookback 365 \
  --params '{"short_period": [5,10,15,20,25,30], "long_period": [30,40,50,60,70,80]}' \
  --algorithm random_search \
  --objective sharpe_ratio \
  --max-iterations 50 \
  --constraints "short_period < long_period"
```

### REST API Usage

Start the API server:

```bash
python main.py api --host 0.0.0.0 --port 8006
```

Or use uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8006
```

API endpoints:

```bash
# Start optimization
curl -X POST http://localhost:8006/optimizations \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_name": "SMA_Crossover",
    "symbols": ["AAPL"],
    "timeframe": "1 day",
    "lookback": 365,
    "param_ranges": {
      "short_period": [5, 10, 15, 20],
      "long_period": [30, 40, 50, 60]
    },
    "algorithm": "grid_search",
    "objective": "sharpe_ratio",
    "constraints": ["short_period < long_period"]
  }'

# Check optimization status
curl http://localhost:8006/optimizations/1

# Get top results
curl http://localhost:8006/optimizations/1/results?top_n=10

# List all optimizations
curl http://localhost:8006/optimizations?limit=20

# Health check
curl http://localhost:8006/healthz
```

## Parameter Ranges

### List Format
Specify exact values to test:

```json
{
  "short_period": [5, 10, 15, 20],
  "long_period": [30, 40, 50, 60]
}
```

This creates a 4×4 grid = 16 combinations.

### Range Tuple Format
Specify (start, stop, step) for numeric ranges:

```json
{
  "short_period": [5, 20, 5],    // [5, 10, 15, 20]
  "long_period": [30, 60, 10],   // [30, 40, 50, 60]
  "threshold": [1.5, 2.5, 0.25]  // [1.5, 1.75, 2.0, 2.25, 2.5]
}
```

## Optimization Algorithms

### Grid Search
Exhaustively tests all parameter combinations.

**Pros:**
- Guaranteed to find optimal parameters in the search space
- Comprehensive coverage
- Deterministic results

**Cons:**
- Exponentially expensive with more parameters
- Not practical for large parameter spaces

**Best For:**
- 2-3 parameters with limited ranges
- When you need guaranteed optimal results
- Initial exploration

**Example:** 4 parameters × 5 values each = 625 combinations

### Random Search
Randomly samples parameter combinations.

**Pros:**
- Much faster than grid search
- Often finds good solutions quickly
- Works well in high-dimensional spaces
- Can set a time/iteration budget

**Cons:**
- No guarantee of finding optimal parameters
- Results vary between runs (use --seed for reproducibility)

**Best For:**
- Many parameters (4+)
- Large parameter ranges
- Quick exploration
- When grid search is too expensive

**Example:** Test 100 random combinations instead of 625 grid combinations

## Objective Functions

Choose what to optimize for:

- **`sharpe_ratio`**: Risk-adjusted returns (default)
- **`total_return`**: Maximize absolute returns
- **`profit_factor`**: Ratio of gross profit to gross loss
- **`win_rate`**: Percentage of winning trades

Example:
```bash
--objective sharpe_ratio  # Best for risk-adjusted performance
--objective total_return  # Best for maximizing returns (ignores risk)
--objective profit_factor # Best for consistent profitability
```

## Constraints

Define parameter relationships to skip invalid combinations:

```bash
# Short period must be less than long period
--constraints "short_period < long_period"

# Multiple constraints (comma-separated)
--constraints "short_period < long_period,entry_threshold > exit_threshold,stop_loss < entry_threshold"
```

This reduces the search space by skipping invalid parameter combinations.

## Parallel Execution

The optimizer automatically uses all available CPU cores:

```bash
# Use all CPU cores (default)
python main.py optimize --strategy SMA_Crossover ...

# Limit to specific number of workers
python main.py optimize --strategy SMA_Crossover ... --workers 4

# Sequential execution (debugging)
python main.py optimize --strategy SMA_Crossover ... --workers 1
```

Performance example on 16-core machine:
- Grid search with 100 combinations: ~50 seconds parallel vs ~800 seconds sequential
- Each backtest takes ~8 seconds on average

## Database Schema

### optimization_runs
Tracks optimization jobs:
- `id`: Run identifier
- `strategy_name`: Strategy being optimized
- `algorithm`: grid_search or random_search
- `symbols`: Symbols tested
- `param_ranges`: Parameter space definition
- `objective`: Objective function
- `status`: pending, running, completed, failed, stopped
- `best_params`: Best parameter combination found
- `best_score`: Best objective score
- `completed_combinations`: Progress tracking

### optimization_results
Individual parameter combinations tested:
- `run_id`: Links to optimization_runs
- `params_json`: Specific parameters tested
- `backtest_run_id`: Links to backtest_runs table
- `score`: Objective function value
- `sharpe_ratio`, `total_return`, `max_drawdown`, etc.: Quick-access metrics

## Examples

### Example 1: Optimize SMA Crossover

```bash
python main.py optimize \
  --strategy SMA_Crossover \
  --symbols AAPL \
  --params '{"short_period": [5,10,15,20], "long_period": [30,40,50,60]}' \
  --algorithm grid_search \
  --objective sharpe_ratio \
  --constraints "short_period < long_period"
```

Expected output:
```
================================================================================
Strategy Parameter Optimization
================================================================================
Strategy: SMA_Crossover
Symbols: ['AAPL']
Algorithm: grid_search
Objective: sharpe_ratio
Parameter ranges: {'short_period': [5, 10, 15, 20], 'long_period': [30, 40, 50, 60]}
Constraints: ['short_period < long_period']
================================================================================
...
Progress: 16/16 (100.0%)
Current best: 1.2534 with params {'short_period': 10, 'long_period': 50}
================================================================================
Optimization Complete!
================================================================================
Run ID: 1
Status: completed
Combinations tested: 16
Duration: 142.3 seconds
Best sharpe_ratio: 1.2534
Best parameters: {
  "short_period": 10,
  "long_period": 50
}
================================================================================
```

### Example 2: Optimize Mean Reversion

```bash
python main.py optimize \
  --strategy Mean_Reversion \
  --symbols SPY \
  --params '{"lookback_period": [10,15,20,25], "entry_threshold": [1.5,2.0,2.5], "exit_threshold": [0.0,0.5,1.0]}' \
  --algorithm random_search \
  --objective profit_factor \
  --max-iterations 50 \
  --constraints "entry_threshold > exit_threshold"
```

### Example 3: Optimize Pairs Trading

```bash
python main.py optimize \
  --strategy Pairs_Trading \
  --symbols "AAPL,MSFT" \
  --params '{"lookback_window": [15,20,25,30], "entry_threshold": [1.5,2.0,2.5,3.0], "exit_threshold": [0.25,0.5,0.75]}' \
  --algorithm grid_search \
  --objective sharpe_ratio \
  --constraints "entry_threshold > exit_threshold"
```

### Example 4: Quick Random Exploration

```bash
# Test 30 random combinations in large parameter space
python main.py optimize \
  --strategy SMA_Crossover \
  --symbols AAPL \
  --params '{"short_period": [3,5,7,10,12,15,17,20], "long_period": [25,30,35,40,45,50,55,60]}' \
  --algorithm random_search \
  --max-iterations 30 \
  --objective sharpe_ratio \
  --seed 42  # Reproducible results
```

## Configuration Options

### CLI Arguments

- `--strategy`: Strategy name (required)
- `--symbols`: Comma-separated symbols (required)
- `--timeframe`: Bar timeframe (default: "1 day")
- `--lookback`: Lookback period in days (default: 365)
- `--params`: Parameter ranges as JSON (required)
- `--algorithm`: grid_search or random_search (default: grid_search)
- `--objective`: Objective function (default: sharpe_ratio)
- `--constraints`: Comma-separated constraints (optional)
- `--workers`: Number of parallel workers (default: all cores)
- `--max-iterations`: Max iterations for random search (default: 100)
- `--seed`: Random seed for reproducibility (optional)
- `--batch-size`: Batch size for processing (default: 50)
- `--config`: Additional config as JSON (optional)

### Additional Config JSON

Pass custom commission/slippage settings:

```bash
--config '{
  "commission_per_share": 0.005,
  "min_commission": 1.0,
  "slippage_ticks": 1
}'
```

## Performance Tips

1. **Start Small**: Test with a small parameter space first
2. **Use Constraints**: Reduce invalid combinations
3. **Random Search First**: Quick exploration before expensive grid search
4. **Batch Size**: Increase `--batch-size` for more efficient parallel execution
5. **Monitor Resources**: Check CPU/memory usage during optimization

## Troubleshooting

### Optimization takes too long
- Use random search instead of grid search
- Reduce parameter ranges
- Decrease lookback period
- Add more constraints to skip invalid combinations

### Out of memory errors
- Reduce `--workers` to limit parallel processes
- Decrease `--batch-size`
- Use fewer symbols or shorter lookback period

### No results or low scores
- Check if strategy has any signals in the lookback period
- Verify parameter ranges are reasonable
- Try different objective functions
- Check constraint definitions

## Future Enhancements (Phase 2 & 3)

Coming in Phase 2:
- Bayesian optimization with Optuna
- Walk-forward analysis
- Out-of-sample validation
- Cross-validation with time series splits

Coming in Phase 3:
- Parameter sensitivity analysis
- Pareto frontier visualization
- Real-time progress WebSocket updates
- Result caching to avoid redundant backtests
- Export to CSV/JSON
- Performance reports and charts

## Database Queries

Query optimization results directly:

```sql
-- Get all optimization runs
SELECT id, strategy_name, algorithm, status, completed_combinations, 
       best_score, created_at
FROM optimization_runs
ORDER BY created_at DESC;

-- Get top 10 parameter combinations from a run
SELECT params_json, score, sharpe_ratio, total_return, max_drawdown
FROM optimization_results
WHERE run_id = 1
ORDER BY score DESC
LIMIT 10;

-- Find best parameters across all runs for a strategy
SELECT r.params_json, r.score, run.created_at
FROM optimization_results r
JOIN optimization_runs run ON r.run_id = run.id
WHERE run.strategy_name = 'SMA_Crossover'
ORDER BY r.score DESC
LIMIT 10;
```

