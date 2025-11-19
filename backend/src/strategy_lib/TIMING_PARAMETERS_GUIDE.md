# Pairs Trading Strategy: Timing Parameters Guide

This document explains all timing-related parameters in the pairs trading strategies and how they interact to prevent hidden bugs related to data availability and warmup periods.

## Critical Understanding: Bar Types

The strategy operates on **two types of bars**:

1. **Raw Bars**: The actual 5-second market data bars
2. **Spread Bars**: Aggregated statistical samples created from raw bars

### Example:
- `stats_aggregation_seconds = 60` means we aggregate 12 raw 5-second bars into 1 spread bar
- `lookback_window = 20` means we need 20 spread bars for z-score calculation
- Therefore, we need `20 × 12 = 240` raw bars minimum before trading

---

## Core Timing Parameters

### 1. `stats_aggregation_seconds` (Aggregation Strategy Only)
**What it does**: Groups multiple raw 5-second bars into one statistical sample for spread calculation.

**Why it exists**: Reduces noise and creates more stable mean-reversion signals.

**How it works**:
- Accumulates `stats_aggregation_seconds / 5` raw bars
- When the buffer is full, creates one spread entry
- Resets the buffer and starts accumulating again

**Example**:
```
stats_aggregation_seconds = 60
→ stats_aggregation_bars = 60 / 5 = 12 raw bars
→ Every 12th raw bar creates 1 spread bar
```

**Valid values**: Must be a multiple of 5 (the base bar timeframe)
- Recommended: 60, 180, 300, 600, 900, 1800, 3600
- Lower = more responsive but noisier
- Higher = smoother but slower to react

---

### 2. `lookback_window`
**What it does**: Number of **spread bars** used to calculate mean and standard deviation for z-score.

**Critical distinction**: This is in **spread bars**, not raw bars!

**Formula for raw bars needed**:
```
raw_bars_for_lookback = lookback_window × stats_aggregation_bars
```

**Example**:
```
lookback_window = 20 spread bars
stats_aggregation_seconds = 60 (12 raw bars per spread)
→ Need 20 × 12 = 240 raw bars for z-score calculation
```

**Valid values**: 
- Recommended: 10-40 spread bars
- Lower = more sensitive to short-term moves
- Higher = more stable, catches longer cycles

---

### 3. `min_hedge_lookback`
**What it does**: Minimum number of **raw bar prices** needed to calculate the hedge ratio (beta).

**Why it exists**: Linear regression needs sufficient data points for reliable hedge ratio estimation.

**Critical bug fix**: The `price_history` deque must be at least this long, or it will cap at a smaller size and prevent hedge ratio calculation!

**Formula**:
```python
maxlen = max(
    spread_history_bars,
    lookback_window,
    min_hedge_lookback  # ← MUST include this!
)
```

**Valid values**: 
- Recommended: 120 raw bars minimum
- Must be ≥ 100 for statistical significance
- Smaller values risk unstable hedge ratios

---

### 4. `spread_history_bars`
**What it does**: Maximum size of the spread history deque (rolling window).

**Why it exists**: Limits memory usage and keeps the spread history relevant to recent market conditions.

**Typical usage**:
- Used as a buffer size for spread calculations
- Should be larger than `lookback_window` to maintain context

**Valid values**:
- Recommended: 100-200 spread bars
- Must be ≥ `lookback_window`

---

### 5. `hedge_refresh_bars`
**What it does**: How many **raw bars** to wait before recalculating the hedge ratio.

**Why it exists**: Hedge ratio changes slowly; recalculating every bar is wasteful and can introduce noise.

**Valid values**:
- Recommended: 120-360 raw bars (10-30 minutes)
- Lower = more responsive to changing correlations
- Higher = more stable, less computation

---

### 6. `lookback_periods`
**What it does**: Number of **raw bars** the backtest engine passes to the strategy on initial warmup.

**Critical importance**: Must be large enough for:
1. Price history warmup (`min_hedge_lookback`)
2. Spread creation (`lookback_window × stats_aggregation_bars`)
3. Buffer for safety

**Formula for minimum**:
```
minimum_lookback_periods = min_hedge_lookback + (lookback_window × stats_aggregation_bars) + buffer
```

**Example**:
```
min_hedge_lookback = 120
lookback_window = 40
stats_aggregation_seconds = 3600 (720 raw bars)

minimum_lookback_periods = 120 + (40 × 720) + 100
                        = 120 + 28,800 + 100
                        = 29,020 raw bars
```

**Recommendation**: Always set to the maximum available data for your date range.

---

## Parameter Interdependencies

### Critical Constraint #1: Price History Size
```python
# BUG: This was missing min_hedge_lookback!
# maxlen = max(spread_history_bars, lookback_window)  # ❌ WRONG

# FIX: Include min_hedge_lookback
maxlen = max(
    spread_history_bars,
    lookback_window,
    min_hedge_lookback  # ✅ CORRECT
)
```

**Why this matters**: If `min_hedge_lookback > maxlen`, the price history will cap at `maxlen` and never have enough data to calculate the hedge ratio, preventing all spread calculations!

### Critical Constraint #2: Warmup Requirements
```
Total warmup time (raw bars) = max(
    min_hedge_lookback,
    lookback_window × stats_aggregation_bars
)
```

**Example failure case**:
```
min_hedge_lookback = 120
lookback_window = 20
stats_aggregation_seconds = 60 (12 bars)
lookback_periods = 200  # Too small!

Need: 120 + (20 × 12) = 360 raw bars
Have: 200 raw bars
Result: Strategy never finishes warmup, 0 trades
```

---

## Warmup Phase Explained

The strategy goes through these stages:

### Stage 1: Raw Bar Collection (0 to `min_hedge_lookback` bars)
- Collecting raw bar prices
- Cannot calculate hedge ratio yet
- Cannot create spreads
- **Status**: Warming up, no trading possible

### Stage 2: Spread Accumulation (`min_hedge_lookback` to `min_hedge_lookback + (lookback_window × stats_aggregation_bars)`)
- Hedge ratio calculated
- Creating spread bars from aggregated prices
- **Status**: Creating spread history, no trading yet

### Stage 3: Fully Warmed Up
- Have enough spread bars for z-score calculation
- Can enter/exit positions
- **Status**: Trading active

---

## Parameter Recommendations by Timeframe

### For 5-Second Bars (Recommended Settings)

#### Short-term (5-30 min cycles):
```yaml
stats_aggregation_seconds: 60-300   # 1-5 minutes
lookback_window: 10-20              # Recent context
min_hedge_lookback: 120             # 10 minutes of prices
hedge_refresh_bars: 120-240         # Refresh every 10-20 min
lookback_periods: 1000+             # At least 1.5 hours of data
```

#### Medium-term (30 min - 2 hour cycles):
```yaml
stats_aggregation_seconds: 300-900  # 5-15 minutes
lookback_window: 20-30              # Balance responsiveness/stability
min_hedge_lookback: 180             # 15 minutes of prices
hedge_refresh_bars: 240-480         # Refresh every 20-40 min
lookback_periods: 5000+             # At least 7 hours of data
```

#### Long-term (2+ hour cycles):
```yaml
stats_aggregation_seconds: 1800-3600  # 30-60 minutes
lookback_window: 30-40                # Longer context for stability
min_hedge_lookback: 240               # 20 minutes of prices
hedge_refresh_bars: 360-720           # Refresh every 30-60 min
lookback_periods: 20000+              # At least 1 day of data
```

---

## Time-Based Parameters (Not Bars)

### 1. `max_hold_bars`
**What it does**: Maximum number of **raw bars** to hold a position.

**Units**: Raw 5-second bars

**Example**:
```
max_hold_bars = 3600 = 3600 × 5 seconds = 5 hours
```

**Valid values**:
- Short-term: 720-1800 (1-2.5 hours)
- Medium-term: 1800-7200 (2.5-10 hours)
- Long-term: 7200-21600 (10-30 hours)

### 2. `cooldown_bars`
**What it does**: Number of **raw bars** to wait after closing a position before entering the same pair again.

**Units**: Raw 5-second bars

**Example**:
```
cooldown_bars = 600 = 600 × 5 seconds = 50 minutes
```

**Valid values**:
- Aggressive: 120-300 (10-25 minutes)
- Conservative: 300-1200 (25 minutes - 1.7 hours)

### 3. `stop_loss_zscore`
**What it does**: Z-score threshold to force exit (not time-based).

**Valid values**: 3.0-5.0 (must be larger than entry threshold)

---

## Common Pitfalls

### ❌ Pitfall #1: "0 Trades" Despite Correct Settings
**Symptom**: Backtest completes but generates 0 trades

**Causes**:
1. `lookback_periods` too small for warmup
2. `min_hedge_lookback` > `price_history maxlen` (now fixed!)
3. `spread_history_bars` < `lookback_window`
4. `entry_threshold` too high for the pair's volatility

**Fix**: Check logs for warmup progress: `"X/Y spread bars collected"`

### ❌ Pitfall #2: "Strategy Stuck at X/Y Bars"
**Symptom**: Warmup message shows `"18/20 spread bars collected"` forever

**Cause**: Aggregation buffer not filling (usually due to insufficient raw bars)

**Fix**: Ensure `lookback_periods` > required minimum

### ❌ Pitfall #3: Inconsistent Results Between Backtest and Optimizer
**Symptom**: Backtest trades, optimizer doesn't (or vice versa)

**Cause**: Different `lookback_periods` settings

**Fix**: Both must use the same warmup length

---

## Validation Checklist

Before running a backtest or optimizer, verify:

- [ ] `lookback_periods` ≥ `min_hedge_lookback` + (`lookback_window` × `stats_aggregation_bars`) + 100
- [ ] `price_history maxlen` ≥ `min_hedge_lookback`
- [ ] `spread_history_bars` ≥ `lookback_window`
- [ ] `stats_aggregation_seconds` is a multiple of 5
- [ ] `entry_threshold` > `exit_threshold`
- [ ] `stop_loss_zscore` > `entry_threshold`
- [ ] Available data covers the date range with sufficient bars

---

## Example: Valid Configuration

```yaml
# Strategy: Pairs_Trading_Adaptive_Aggregated
# Timeframe: 5 secs
# Target: 15-minute mean reversion cycles

stats_aggregation_seconds: 300      # 60 raw bars → 1 spread bar
lookback_window: 20                 # 20 spread bars for z-score
min_hedge_lookback: 120             # 120 raw bars for hedge ratio
spread_history_bars: 100            # Keep 100 spread bars in memory
hedge_refresh_bars: 240             # Refresh hedge every 20 minutes
max_hold_bars: 3600                 # Hold up to 5 hours
cooldown_bars: 300                  # Wait 25 minutes between trades
lookback_periods: 2000              # Pass 2.8 hours on warmup

# Validation:
# maxlen = max(100, 20, 120) = 120 ✅
# warmup needed = 120 + (20 × 60) = 1320 raw bars
# warmup provided = 2000 raw bars ✅
# All constraints satisfied!
```

---

## Live Trading Considerations

### Fast Warmup from Database
When the strategy starts in live trading, it can:
1. Query recent bars from the database
2. Populate `price_history` and `spread_history` quickly
3. Start trading without waiting for real-time bars

**Requirements**:
- Database must have `lookback_periods` bars
- Use the `_fast_warmup_pair()` method
- Set `lookback_periods` to 3600+ for instant warmup

### Dynamic Hedge Ratio Updates
In live trading, the hedge ratio updates every `hedge_refresh_bars`:
- Too frequent = noisy, over-reactive
- Too infrequent = missed correlation shifts

**Recommended**: 120-360 bars (10-30 minutes) for 5-second data

---

## Summary: Key Takeaways

1. **Two bar types exist**: Raw bars (5-sec data) and Spread bars (aggregated)
2. **Price history must be large enough** to hold `min_hedge_lookback` bars
3. **Warmup requires time**: `min_hedge_lookback + (lookback_window × aggregation_bars)`
4. **Always specify units**: Raw bars or Spread bars
5. **Validate before running**: Use the checklist above

---

*Last Updated: 2025-11-18*
*After fixing critical bugs in price_history sizing and bar processing*

