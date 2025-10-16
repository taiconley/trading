# Critical Bugfix: Lookback Period Issue

## Problem
Both CLI backtester and optimizer were returning 0 trades, even with parameters that previously generated trades (e.g., run 17).

## Root Cause
The backtester engine only calls `strategy.on_bar()` when it has accumulated at least `lookback_periods` bars. When `lookback_periods` was set too high:

- **With lookback=250**: Engine waits until bar 250, then calls `on_bar()` only ONCE
- **With lookback=100**: Engine calls `on_bar()` starting from bar 100, giving 150 calls total
- **With lookback=8**: Engine calls `on_bar()` starting from bar 8, giving 242 calls total

### Why This Matters
Strategies like SMA Crossover need `on_bar()` to be called **multiple times** to:
1. Build up historical SMA values
2. Detect crossovers by comparing current vs previous SMA values
3. The strategy checks `if len(self._sma_data[symbol]["short"]) < 2`, which requires at least 2 calls

With only 1 call to `on_bar()`, the strategy cannot detect any crossovers → 0 trades.

## Investigation Steps
1. Added debug print statements to `on_bar()` in `sma_cross.py`
2. Discovered `on_bar()` was only called once with lookback=250
3. Tested with lookback=8 → `on_bar()` called 242 times → 1 trade generated ✅
4. Checked database - run 17 likely used default lookback=100 → trades generated

## Fix
Changed optimizer's default `--lookback` from **365** to **100** to match backtester CLI default.

**File**: `backend/src/services/optimizer/main.py`
```python
# Before:
optimize_parser.add_argument('--lookback', type=int, default=365, help='Lookback days')

# After:
optimize_parser.add_argument('--lookback', type=int, default=100, help='Lookback days')
```

## Test Results

### Before Fix (lookback=250 or 365)
- CLI backtest: 0 trades
- Grid search optimizer: 0 trades across all combinations
- Bayesian optimizer: 0 trades across all combinations

### After Fix (lookback=100)
- CLI backtest: 1 trade (100% win rate, $1,939 profit)
- Grid search optimizer run 22:
  - short=5, long=20: 1 trade, Sharpe=0.8365 ✅
  - short=5, long=30: 1 trade, Sharpe=0.1973 ✅
- Bayesian optimizer run 24:
  - short=4, long=24: 1 trade, Sharpe=0.8365 ✅
  - short=4, long=25: 1 trade, Sharpe=0.1984 ✅

## Lessons Learned
1. **Lookback should match strategy needs**: A strategy with SMA(10) only needs 10+ bars, not 365
2. **Balance warmup vs evaluation**: Longer lookback = more warmup, fewer evaluation periods
3. **Default values matter**: Inconsistent defaults between services caused confusion
4. **Test with minimal examples**: Starting with simple parameters (short=3, long=7) helped isolate the issue

## Recommendations
1. Consider adding a warning if lookback is > 2x the longest parameter
2. Document the tradeoff between lookback period and number of evaluation bars
3. For walk-forward analysis, ensure in-sample and out-of-sample windows use appropriate lookback values

