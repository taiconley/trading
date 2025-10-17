# Strategy Parameter Optimizer

Automated parameter optimization for trading strategies using various search algorithms.

## Features

### Phase 1 ✅
- **Grid Search**: Exhaustive search over parameter grid
- **Random Search**: Random sampling of parameter space
- **Parallel Execution**: Multi-core backtest execution
- **Multiple Objectives**: Sharpe ratio, total return, profit factor, win rate
- **Constraint Handling**: Define parameter constraints (e.g., short_period < long_period)
- **Database Storage**: Full optimization history tracking
- **CLI Interface**: Command-line optimization tool
- **REST API**: Web-triggered optimizations

### Phase 2 ✅
- **Bayesian Optimization**: Intelligent parameter search using Optuna
- **Genetic Algorithm**: Evolutionary optimization with selection, crossover, and mutation
- **Walk-Forward Analysis**: Rolling window optimization with out-of-sample validation
- **Out-of-Sample Testing**: Train/test split to detect overfitting
- **Cross-Validation**: K-fold time series validation with purging and embargoes

### Phase 3 ✅
- **Parameter Sensitivity Analysis**: Identify which parameters have the most impact
- **Pareto Frontier Analysis**: Multi-objective optimization trade-offs
- **Result Caching**: Avoid redundant backtests with smart caching
- **Export Functionality**: Export results to CSV, JSON, or text reports
- **Resource Monitoring**: Track CPU and memory usage during optimization
- **Graceful Cancellation**: Stop running optimizations cleanly

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

# Bayesian Optimization - Intelligent parameter search  
python main.py optimize \
  --strategy SMA_Crossover \
  --symbols AAPL \
  --timeframe "1 day" \
  --lookback 180 \
  --params '{"short_period": [3,5,7,10,12,15,20], "long_period": [20,25,30,35,40,45,50,60]}' \
  --algorithm bayesian \
  --objective sharpe_ratio \
  --constraints "short_period < long_period" \
  --max-iterations 20 \
  --seed 42

# Walk-Forward Analysis - Rolling window validation
python main.py walk-forward \
  --strategy SMA_Crossover \
  --symbols AAPL \
  --timeframe "1 day" \
  --params '{"short_period": [5,10,15], "long_period": [20,30,40]}' \
  --window-size 180 \
  --step-size 30 \
  --algorithm grid_search \
  --objective sharpe_ratio \
  --constraints "short_period < long_period"

# Out-of-Sample Testing - Train/test split
python main.py out-of-sample \
  --strategy SMA_Crossover \
  --symbols AAPL \
  --timeframe "1 day" \
  --params '{"short_period": [5,10], "long_period": [20,30]}' \
  --train-ratio 0.7 \
  --lookback 100 \
  --algorithm grid_search \
  --objective sharpe_ratio \
  --constraints "short_period < long_period"

# Cross-Validation - K-fold validation
python main.py cross-validate \
  --strategy SMA_Crossover \
  --symbols AAPL \
  --timeframe "1 day" \
  --params '{"short_period": [5,10], "long_period": [20,30]}' \
  --n-splits 5 \
  --lookback 50 \
  --algorithm grid_search \
  --objective sharpe_ratio
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

### Bayesian Optimization (Optuna)
Uses past results to intelligently suggest promising parameters.

**Pros:**
- Much more efficient than random/grid search
- Learns from each iteration to focus on promising regions
- Excellent for expensive objective functions (like backtests)
- Handles constraints intelligently
- Often finds near-optimal solutions in far fewer iterations

**Cons:**
- Results vary between runs (use --seed for reproducibility)
- Slightly more overhead per iteration
- No guarantee of finding global optimum

**Best For:**
- Expensive objective functions (backtesting)
- Limited iteration budget (10-50 iterations)
- Many parameters with large ranges
- When you need good results quickly

**Example:** Find near-optimal parameters in 20-30 iterations instead of 625 grid combinations

**How It Works:**
- Builds a probabilistic model of the objective function
- Uses acquisition functions to balance exploration vs exploitation
- Automatically prunes unpromising trials early
- Handles discrete and continuous parameters

### Genetic Algorithm (NEW!)
Uses evolutionary principles to search parameter space.

**Pros:**
- Excellent for complex parameter interactions
- Balances exploration and exploitation naturally
- Provides multiple good solutions
- Transparent and interpretable
- Works well with constraints
- Handles large discrete spaces efficiently

**Cons:**
- Requires tuning hyperparameters (population size, mutation rate)
- Results vary between runs (use --seed for reproducibility)
- Slower per iteration than random search

**Best For:**
- Complex parameter interactions
- Large discrete parameter spaces (1000+ combinations)
- When you want multiple good solutions
- Problems where parameters interact in non-linear ways
- Exploring trade-offs between objectives

**Example:** Evolve optimal parameters through 100 generations

**How It Works:**
- Initialize random population of parameter combinations
- Evaluate fitness (run backtests)
- Select best individuals (tournament or roulette)
- Create offspring via crossover (combine parameters)
- Apply mutations (random changes)
- Preserve elite (best solutions)
- Repeat until convergence

**Configuration:**
```bash
--config '{
  "population_size": 50,
  "elite_size": 5,
  "mutation_rate": 0.1,
  "crossover_rate": 0.8,
  "selection_method": "tournament",
  "crossover_method": "uniform"
}'
```

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

### Example 5: Genetic Algorithm Evolution

```bash
# Evolve optimal parameters using genetic algorithm
python main.py optimize \
  --strategy Mean_Reversion \
  --symbols SPY \
  --params '{
    "lookback": [10,15,20,25,30],
    "entry_threshold": [1.5,2.0,2.5,3.0],
    "exit_threshold": [0.0,0.5,1.0],
    "stop_loss": [2.0,3.0,4.0]
  }' \
  --algorithm genetic \
  --max-iterations 200 \
  --objective sharpe_ratio \
  --seed 42 \
  --config '{
    "population_size": 50,
    "elite_size": 5,
    "mutation_rate": 0.1,
    "crossover_rate": 0.8,
    "selection_method": "tournament",
    "crossover_method": "uniform"
  }'
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

## Validation Methods (Phase 2)

### Walk-Forward Analysis

Tests strategy robustness by optimizing on rolling time windows and validating on future out-of-sample periods.

**How It Works:**
1. Split data into overlapping windows (e.g., 6 months each)
2. For each window:
   - Optimize on first 70% (in-sample)
   - Validate on last 30% (out-of-sample)
3. Step forward by a fixed period (e.g., 1 month)
4. Repeat until end of data

**Benefits:**
- Simulates real trading where you periodically re-optimize
- Detects overfitting by comparing in-sample vs out-of-sample performance
- Shows stability of parameters over time
- More realistic than single train/test split

**Usage:**
```bash
python main.py walk-forward \
  --strategy SMA_Crossover \
  --symbols AAPL \
  --timeframe "1 day" \
  --params '{"short_period": [5,10,15], "long_period": [20,30,40]}' \
  --window-size 180 \
  --step-size 30 \
  --train-ratio 0.7 \
  --algorithm grid_search
```

**Output:**
- Per-window best parameters and scores
- In-sample vs out-of-sample performance
- Stability score (consistency of parameters across windows)
- Average degradation (performance drop on out-of-sample data)

### Out-of-Sample Testing

Simple train/test split to validate if optimized parameters generalize to unseen data.

**How It Works:**
1. Split data into training (e.g., 70%) and testing (30%) sets
2. Optimize parameters on training data
3. Test best parameters on held-out testing data
4. Compare training vs testing performance

**Benefits:**
- Quick validation of overfitting
- Standard machine learning approach
- Easy to understand and interpret

**Usage:**
```bash
python main.py out-of-sample \
  --strategy SMA_Crossover \
  --symbols AAPL \
  --timeframe "1 day" \
  --params '{"short_period": [5,10], "long_period": [20,30]}' \
  --train-ratio 0.7 \
  --algorithm grid_search
```

**Output:**
- Best parameters from training data
- Training score vs testing score
- Score degradation (percentage drop)
- Overfitting detected flag (if degradation > threshold)

### Cross-Validation

K-fold validation adapted for time series data with purging and embargoes to prevent lookahead bias.

**How It Works:**
1. Split data into K consecutive folds
2. For each fold:
   - Train on fold data
   - Test on next fold
   - Apply purging (remove overlapping data)
   - Apply embargo (skip period after training)
3. Aggregate results across folds

**Benefits:**
- More robust than single train/test split
- Uses all data for both training and testing
- Purging prevents information leakage
- Embargo simulates real-world trading lag

**Usage:**
```bash
python main.py cross-validate \
  --strategy SMA_Crossover \
  --symbols AAPL \
  --timeframe "1 day" \
  --params '{"short_period": [5,10], "long_period": [20,30]}' \
  --n-splits 5 \
  --purge-pct 0.01 \
  --embargo-pct 0.01 \
  --algorithm grid_search
```

**Output:**
- Per-fold best parameters and scores
- Mean and standard deviation of scores
- Consistency of parameters across folds
- Overall validation score

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

## Phase 3 Features (Analytics & Production)

### Parameter Sensitivity Analysis

Identify which parameters have the greatest impact on strategy performance.

**API Usage:**
```bash
# Run sensitivity analysis
curl http://localhost:8006/optimizations/1/analysis
```

**Response includes:**
- Sensitivity scores (how much variance each parameter explains)
- Correlation with objective function
- Parameter importance ranking
- Interaction effects between parameters
- Statistical measures (mean, std, min, max)

**Example:**
```json
{
  "run_id": 1,
  "parameters": {
    "short_period": {
      "sensitivity_score": 0.45,
      "correlation": -0.32,
      "importance_rank": 1,
      "interactions": {"long_period": 0.65}
    },
    "long_period": {
      "sensitivity_score": 0.28,
      "correlation": 0.18,
      "importance_rank": 2
    }
  },
  "top_parameters": [
    {"name": "short_period", "sensitivity_score": 0.45, "rank": 1},
    {"name": "long_period", "sensitivity_score": 0.28, "rank": 2}
  ]
}
```

### Pareto Frontier Analysis

Find optimal trade-offs between competing objectives (e.g., maximizing returns while minimizing drawdown).

**API Usage:**
```bash
# Analyze Pareto frontier for two objectives
curl "http://localhost:8006/optimizations/1/pareto?objectives=sharpe_ratio,max_drawdown&maximize=true,false"
```

**Use Cases:**
- Maximize Sharpe ratio while minimizing max drawdown
- Maximize win rate while minimizing number of trades
- Balance return vs risk across multiple dimensions

**Example Response:**
```json
{
  "run_id": 1,
  "objectives": ["sharpe_ratio", "max_drawdown"],
  "n_solutions": 5,
  "pareto_solutions": [
    {
      "id": 42,
      "params": {"short_period": 10, "long_period": 40},
      "objectives": {"sharpe_ratio": 1.8, "max_drawdown": 0.12}
    }
  ]
}
```

### Export Functionality

Export optimization results in various formats.

**CSV Export:**
```bash
# Export top 20 results as CSV
curl http://localhost:8006/optimizations/1/export/csv?top_n=20 -o results.csv
```

**JSON Export:**
```bash
# Export full results as JSON
curl http://localhost:8006/optimizations/1/export/json > results.json
```

**Summary Report:**
```bash
# Get human-readable text report
curl http://localhost:8006/optimizations/1/report -o report.txt
```

The summary report includes:
- Run metadata (strategy, algorithm, symbols, timing)
- Best parameters found
- Top 10 results
- Parameter sensitivity analysis (if available)
- Execution statistics

### Result Caching

Automatically caches backtest results to avoid redundant computation.

**How It Works:**
- Results are cached by parameter combination hash
- Cache checks include strategy, symbols, timeframe, lookback, and config
- Reuses results from previous optimizations with identical conditions
- Configurable cache age limit

**Benefits:**
- Significantly faster for repeated optimizations
- Useful when tweaking optimization settings (algorithm, objective)
- Saves computational resources

**Performance:** Can achieve 80%+ cache hit rates when refining optimizations.

### Resource Monitoring

Track CPU and memory usage during optimization runs.

**System Info:**
```bash
# Get current system resource info
curl http://localhost:8006/healthz
```

**Features:**
- Real-time CPU and memory tracking
- Per-process resource usage
- Thread and subprocess counting
- Performance statistics (min, max, avg)
- Resource limit checking

### Graceful Cancellation

Stop running optimizations cleanly without losing partial results.

**Usage:**
```bash
# Stop a running optimization
curl -X POST http://localhost:8006/optimizations/1/stop
```

**Behavior:**
- Marks optimization as "stopped" in database
- Currently running backtests complete normally
- Partial results are saved
- No data loss or corruption

## Future Enhancements (Beyond Phase 3)

Planned features:
- Real-time progress WebSocket updates
- Performance reports with charts and visualizations
- Strategy comparison across multiple optimizations
- Hyperparameter importance plots with visualizations
- Automated parameter range suggestions
- GPU acceleration for objective function calculation

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

