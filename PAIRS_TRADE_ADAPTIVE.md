# Pairs Trading Adaptive Strategy - Complete Variable Reference

This document provides a comprehensive explanation of every variable used in the `Pairs_Trading_Adaptive` strategy.

## Important Note: `lookback_periods` vs `lookback_window`

**`lookback_periods`** (e.g., `--lookback 350` in optimizer scripts) is **NOT** a strategy variable. It's a backtester/optimizer configuration parameter that controls how many bars of historical data are fetched and provided to the strategy during backtesting/optimization. This parameter is set in the optimizer/backtester, not in the strategy configuration.

**`lookback_window`** (e.g., `222` in strategy config) **IS** a strategy variable documented below. It controls how many bars are used to calculate the mean and standard deviation of the spread for z-score computation.

- `lookback_periods` (350): "How many bars to fetch for backtesting"
- `lookback_window` (222): "How many bars to use for spread statistics"

## Table of Contents

1. [Configuration Variables](#configuration-variables)
2. [Pair State Variables](#pair-state-variables)
3. [Internal Helper Variables](#internal-helper-variables)
4. [Variable Relationships](#variable-relationships)

---

## Configuration Variables

These variables are set in the strategy configuration and control the behavior of the strategy.

### Pairs Definition

#### `pairs`
- **Type**: `List[List[str]]`
- **Default**: `[]`
- **Description**: List of stock pairs to trade. Each pair is a list of two stock symbols `[stock_a, stock_b]`.
- **Example**: `[["AMN", "CORT"], ["ACA", "KTB"]]`
- **Usage**: Defines which pairs the strategy will monitor and trade. The strategy processes each pair independently.

---

### Core Trading Parameters

#### `lookback_window`
- **Type**: `int`
- **Default**: `240`
- **Description**: Number of recent bars used to calculate the mean and standard deviation of the spread for z-score computation.
- **Time Context**: At 5-second bars, 240 bars = 20 minutes of history
- **Usage**: 
  - Used to compute rolling mean and std of spread: `mean_spread = mean(spread_history[-lookback_window:])`
  - Must be ≤ `spread_history_bars` (automatically enforced)
  - Larger values = more stable but slower-responding statistics
- **Typical Range**: 100-500 bars

#### `entry_threshold`
- **Type**: `float`
- **Default**: `2.0`
- **Description**: Z-score threshold to trigger entry signals. When `|zscore| >= entry_threshold`, a trade is initiated.
- **Usage**:
  - If `zscore > entry_threshold`: Short stock A, Long stock B (spread is too high)
  - If `zscore < -entry_threshold`: Long stock A, Short stock B (spread is too low)
- **Note**: This threshold is dynamically adjusted by volatility adaptation when enabled
- **Typical Range**: 1.0-3.0

#### `exit_threshold`
- **Type**: `float`
- **Default**: `0.5`
- **Description**: Z-score threshold to exit positions. When `|zscore| < exit_threshold`, the position is closed.
- **Usage**: Exits when spread returns to mean (mean reversion complete)
- **Note**: This threshold is dynamically adjusted by volatility adaptation when enabled
- **Typical Range**: 0.2-1.0

---

### Position Sizing Variables

#### `position_size`
- **Type**: `int`
- **Default**: `100`
- **Description**: Legacy fallback position size in shares per leg. Used only if advanced sizing fails.
- **Usage**: Fallback when `_plan_position()` cannot determine size
- **Note**: Modern strategy uses `base_pair_notional` and risk-based sizing instead

#### `base_pair_notional`
- **Type**: `float`
- **Default**: `25000.0`
- **Description**: Baseline gross notional (dollar amount) allocated to each pair on entry.
- **Usage**: Starting point for position sizing calculation. Actual size may be adjusted by:
  - Volatility ratio
  - Half-life weighting
  - Signal strength
  - Risk budget constraints
- **Example**: $25,000 means $25k long + $25k short = $50k gross notional

#### `min_pair_notional`
- **Type**: `float`
- **Default**: `5000.0`
- **Description**: Minimum gross notional required to enter a position. If calculated size is below this, entry is skipped.
- **Usage**: Prevents entering positions that are too small to be meaningful
- **Note**: Must be less than `max_pair_notional`

#### `max_pair_notional`
- **Type**: `float`
- **Default**: `100000.0`
- **Description**: Maximum gross notional allowed for a single pair.
- **Usage**: Caps position size to prevent over-concentration in one pair
- **Note**: Actual size may be further limited by `max_portfolio_notional`

#### `max_portfolio_notional`
- **Type**: `float`
- **Default**: `300000.0`
- **Description**: Portfolio-wide cap on total gross notional across all pairs simultaneously.
- **Usage**: Limits total exposure across all pairs to manage portfolio risk
- **Calculation**: Sum of `entry_notional` for all pairs with `current_position != "flat"`

#### `volatility_positioning_enabled`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Enable volatility-aware position sizing. When enabled, position size is inversely scaled by current volatility ratio.
- **Usage**: 
  - When volatility is high (vol_ratio > 1.0): Reduce position size
  - When volatility is low (vol_ratio < 1.0): Increase position size
- **Formula**: `target_notional /= (vol_ratio ** volatility_position_power)`

#### `volatility_position_power`
- **Type**: `float`
- **Default**: `1.0`
- **Description**: Exponent applied when scaling position size by volatility ratio.
- **Usage**: 
  - `1.0` = linear scaling (2x volatility = 0.5x size)
  - `0.5` = square root scaling (less aggressive)
  - `2.0` = quadratic scaling (more aggressive)
- **Formula**: `size_multiplier = 1 / (vol_ratio ** power)`

#### `risk_budget_per_pair`
- **Type**: `float`
- **Default**: `1000.0`
- **Description**: Dollar risk budget per pair, divided by spread volatility to determine position size.
- **Usage**: Risk-based sizing: `risk_scaled = risk_budget_per_pair / spread_std`
- **Note**: If both `base_pair_notional` and `risk_budget_per_pair` are set, the smaller value is used

#### `halflife_weight_bars`
- **Type**: `int`
- **Default**: `240`
- **Description**: Reference window (in bars) used to weight position size by half-life.
- **Usage**: 
  - If half-life < `halflife_weight_bars`: Increase position size
  - If half-life > `halflife_weight_bars`: Decrease position size
- **Formula**: `hl_weight = min(1.0, halflife_weight_bars / half_life)`
- **Rationale**: Faster mean reversion (lower half-life) = better opportunity = larger size

#### `max_halflife_bars`
- **Type**: `int`
- **Default**: `720`
- **Description**: Maximum acceptable mean-reversion half-life (in bars). Positions with half-life above this are exited.
- **Usage**: 
  - Exit condition: `half_life > max_halflife_bars`
  - Also blocks entry if `require_half_life=True` and half-life exceeds this
- **Time Context**: At 5-second bars, 720 bars = 1 hour

#### `require_half_life`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Block entries until half-life is estimable and valid.
- **Usage**: 
  - If `True`: Entry requires `half_life` to be not None and `half_life <= max_halflife_bars`
  - If `False`: Entries allowed even without half-life estimate
- **Rationale**: Ensures mean reversion is measurable before risking capital

---

### Order Execution Variables

#### `execution_type`
- **Type**: `str`
- **Default**: `"ADAPTIVE"`
- **Description**: Order execution type for placing trades.
- **Options**:
  - `"MKT"`: Market order (immediate execution at current price)
  - `"LMT"`: Limit order (executes at specified limit price)
  - `"ADAPTIVE"`: IBKR Adaptive order (adjusts limit price dynamically)
  - `"PEG BEST"`: Pegged to best bid/ask
  - `"PEG MID"`: Pegged to midpoint
- **Usage**: Determines how orders are submitted to Interactive Brokers

#### `use_adaptive`
- **Type**: `bool`
- **Default**: `False`
- **Description**: Override flag to force Adaptive order type (overrides `execution_type`).
- **Usage**: If `True`, always uses Adaptive orders regardless of `execution_type`

#### `use_pegged`
- **Type**: `bool`
- **Default**: `False`
- **Description**: Override flag to force Pegged order type (overrides `execution_type`).
- **Usage**: If `True`, always uses Pegged orders regardless of `execution_type`

#### `pegged_type`
- **Type**: `Optional[str]`
- **Default**: `None`
- **Description**: Type of pegged order when `use_pegged=True`.
- **Options**: `"BEST"` or `"MID"`
- **Usage**: Determines whether to peg to best bid/ask or midpoint

#### `adaptive_priority`
- **Type**: `str`
- **Default**: `"Normal"`
- **Description**: Priority level for Adaptive orders.
- **Options**: 
  - `"Patient"`: Slower execution, better price
  - `"Normal"`: Balanced execution
  - `"Urgent"`: Faster execution, may pay more
- **Usage**: Controls aggressiveness of Adaptive order algorithm

#### `pegged_offset`
- **Type**: `float`
- **Default**: `0.01`
- **Description**: Offset from midpoint/best price for pegged orders (in dollars).
- **Usage**: 
  - For `PEG MID`: Price = midpoint + offset
  - For `PEG BEST`: Price = best + offset
- **Example**: `0.01` = 1 cent offset

---

### Risk Management Variables

#### `max_hold_bars`
- **Type**: `int`
- **Default**: `720`
- **Description**: Maximum number of bars to hold a position before forced exit.
- **Time Context**: At 5-second bars, 720 bars = 1 hour
- **Usage**: Exit condition: `bars_in_trade >= max_hold_bars`
- **Rationale**: Prevents holding positions too long if mean reversion doesn't occur

#### `stop_loss_zscore`
- **Type**: `float`
- **Default**: `3.0`
- **Description**: Z-score threshold for stop-loss exit. If z-score moves further from mean (worse), exit immediately.
- **Usage**:
  - If position is `short_a_long_b` and `zscore > stop_loss_zscore`: Exit
  - If position is `long_a_short_b` and `zscore < -stop_loss_zscore`: Exit
- **Rationale**: Cuts losses when spread continues to diverge instead of reverting

#### `cooldown_bars`
- **Type**: `int`
- **Default**: `60`
- **Description**: Cooling-off period (in bars) after stop-loss or loss-based exits before re-entering the same pair.
- **Time Context**: At 5-second bars, 60 bars = 5 minutes
- **Usage**: Prevents immediate re-entry after a loss, allowing market to stabilize
- **Note**: Only applies to exits with reasons: `"stop_loss"`, `"pnl_stop"`, `"volatility_stop"`

#### `max_pair_loss_pct`
- **Type**: `float`
- **Default**: `0.02`
- **Description**: Maximum fractional loss of allocated notional before exiting (2% = 0.02).
- **Usage**: Exit condition: `unrealized_pnl <= -max_pair_loss_pct * entry_notional`
- **Example**: If `entry_notional = $25,000` and `max_pair_loss_pct = 0.02`, exit at -$500 loss

#### `volatility_stop_multiplier`
- **Type**: `float`
- **Default**: `2.5`
- **Description**: Exit if spread deviates this many standard deviations from entry spread.
- **Usage**: Exit condition: `|entry_spread - current_spread| >= volatility_stop_multiplier * std_spread`
- **Rationale**: Exits when spread moves too far from entry, indicating potential breakdown of relationship

---

### Trading Session Variables

#### `market_close_hour`
- **Type**: `int`
- **Default**: `16`
- **Description**: Market close hour (24-hour format).
- **Usage**: Used to determine when to force-close all positions before market close
- **Example**: `16` = 4 PM

#### `market_close_minute`
- **Type**: `int`
- **Default**: `0`
- **Description**: Market close minute.
- **Usage**: Used with `market_close_hour` to determine exact market close time

#### `close_before_eod_minutes`
- **Type**: `int`
- **Default**: `5`
- **Description**: Close all positions N minutes before market close.
- **Usage**: Ensures all positions are closed before market close to avoid overnight risk
- **Example**: `5` = close all positions at 3:55 PM if market closes at 4:00 PM

#### `timezone`
- **Type**: `str`
- **Default**: `"US/Eastern"`
- **Description**: Trading session timezone for market hours calculations.
- **Usage**: Used to convert timestamps and determine market close time
- **Options**: Any valid timezone string (e.g., `"US/Eastern"`, `"US/Pacific"`)

---

### Statistical Modeling Variables

#### `spread_history_bars`
- **Type**: `int`
- **Default**: `1000`
- **Description**: Maximum number of spread values to keep in rolling history buffer.
- **Time Context**: At 5-second bars, 1000 bars ≈ 83 minutes of history
- **Usage**: 
  - Stores historical spread values in `spread_history` deque
  - Used for z-score calculations, stationarity tests, volatility adaptation
  - Must be ≥ `lookback_window` (automatically enforced)
- **Note**: Older values are automatically dropped when limit is reached

#### `hedge_refresh_bars`
- **Type**: `int`
- **Default**: `30`
- **Description**: Frequency (in bars) to recalculate hedge ratio using OLS regression.
- **Time Context**: At 5-second bars, 30 bars = 2.5 minutes
- **Usage**: 
  - Triggers `_refresh_hedge_ratio()` when `bars_since_hedge >= hedge_refresh_bars`
  - Hedge ratio determines how many shares of stock B to hedge 1 share of stock A
- **Rationale**: Regular updates ensure hedge ratio reflects current correlation

#### `min_hedge_lookback`
- **Type**: `int`
- **Default**: `120`
- **Description**: Minimum number of price bars required before computing spreads and hedge ratio.
- **Time Context**: At 5-second bars, 120 bars = 10 minutes
- **Usage**: 
  - Blocks spread calculation until `len(price_history_a) >= min_hedge_lookback`
  - Ensures sufficient data for reliable hedge ratio regression
- **Note**: Must be less than `spread_history_bars`

#### `stationarity_checks_enabled`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Enable statistical stationarity tests (ADF and cointegration) to gate entries.
- **Usage**: 
  - If `True`: Entries blocked if ADF or cointegration p-values exceed thresholds
  - If `False`: Stationarity tests are skipped (allows trading regardless of test results)
- **Rationale**: Stationary spreads are required for mean reversion to work

#### `adf_pvalue_threshold`
- **Type**: `float`
- **Default**: `0.05`
- **Description**: Maximum acceptable p-value for Augmented Dickey-Fuller (ADF) test.
- **Usage**: 
  - If `adf_pvalue > adf_pvalue_threshold`: Entry blocked (spread is not stationary)
  - Lower values = stricter requirement (0.01 = 1% significance level)
- **Note**: ADF test checks if spread is mean-reverting (stationary)

#### `cointegration_pvalue_threshold`
- **Type**: `float`
- **Default**: `0.05`
- **Description**: Maximum acceptable p-value for cointegration test.
- **Usage**: 
  - If `cointegration_pvalue > cointegration_pvalue_threshold`: Entry blocked (stocks are not cointegrated)
  - Lower values = stricter requirement
- **Note**: Cointegration test checks if the two stocks have a long-term equilibrium relationship

#### `stationarity_check_interval`
- **Type**: `int`
- **Default**: `60`
- **Description**: Number of bars between stationarity test recalculations.
- **Time Context**: At 5-second bars, 60 bars = 5 minutes
- **Usage**: 
  - Triggers `_update_stationarity_metrics()` when `bars_since_stationarity >= stationarity_check_interval`
  - Tests are computationally expensive, so not run every bar
- **Note**: Only applies when `stationarity_checks_enabled=True`

---

### Volatility Adaptation Variables

#### `volatility_adaptation_enabled`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Enable dynamic adjustment of entry/exit thresholds based on current volatility.
- **Usage**: 
  - If `True`: Entry/exit thresholds scaled by `volatility_ratio`
  - If `False`: Uses base thresholds (`entry_threshold`, `exit_threshold`) unchanged
- **Rationale**: In high volatility, require larger z-scores to enter (more conservative)

#### `volatility_window`
- **Type**: `int`
- **Default**: `240`
- **Description**: Number of bars used to calculate current volatility for adaptation.
- **Time Context**: At 5-second bars, 240 bars = 20 minutes
- **Usage**: 
  - Computes `window_std = std(spread_history[-volatility_window:])`
  - Compared to `baseline_spread_std` to compute `volatility_ratio`
- **Note**: Should be similar to or larger than `lookback_window`

#### `volatility_ema_alpha`
- **Type**: `float`
- **Default**: `0.2`
- **Description**: Exponential moving average smoothing factor for baseline volatility.
- **Usage**: 
  - Updates baseline: `baseline_spread_std = alpha * window_std + (1-alpha) * baseline_spread_std`
  - Lower values (0.1) = slower adaptation, more stable baseline
  - Higher values (0.3) = faster adaptation, more responsive baseline
- **Range**: 0.0-1.0

#### `min_volatility_ratio`
- **Type**: `float`
- **Default**: `0.75`
- **Description**: Lower clamp for volatility ratio when adjusting entry thresholds.
- **Usage**: 
  - Prevents entry threshold from being reduced too much in low volatility
  - `volatility_ratio = clip(actual_ratio, min_volatility_ratio, max_volatility_ratio)`
- **Example**: If actual ratio is 0.5, clamped to 0.75 (entry threshold reduced by 25% max)

#### `max_volatility_ratio`
- **Type**: `float`
- **Default**: `1.5`
- **Description**: Upper clamp for volatility ratio when adjusting entry thresholds.
- **Usage**: 
  - Prevents entry threshold from being increased too much in high volatility
  - `volatility_ratio = clip(actual_ratio, min_volatility_ratio, max_volatility_ratio)`
- **Example**: If actual ratio is 2.0, clamped to 1.5 (entry threshold increased by 50% max)

#### `min_exit_volatility_ratio`
- **Type**: `float`
- **Default**: `0.8`
- **Description**: Lower clamp for volatility ratio when adjusting exit thresholds.
- **Usage**: 
  - Separate clamp for exit threshold (can be different from entry clamp)
  - `exit_ratio = clip(volatility_ratio, min_exit_volatility_ratio, max_exit_volatility_ratio)`
- **Rationale**: Exit thresholds may need different bounds than entry thresholds

#### `max_exit_volatility_ratio`
- **Type**: `float`
- **Default**: `1.3`
- **Description**: Upper clamp for volatility ratio when adjusting exit thresholds.
- **Usage**: 
  - Separate clamp for exit threshold
  - `exit_ratio = clip(volatility_ratio, min_exit_volatility_ratio, max_exit_volatility_ratio)`
- **Rationale**: Exit thresholds may need different bounds than entry thresholds

---

## Pair State Variables

These variables are stored per pair in `_pair_states[pair_key]` and track the current state of each pair.

### Price and Spread History

#### `price_history_a`
- **Type**: `deque` (maxlen = max(spread_history_bars, lookback_window))
- **Description**: Rolling history of closing prices for stock A.
- **Usage**: 
  - Used for hedge ratio regression
  - Must have at least `min_hedge_lookback` values before spread calculation
- **Note**: Automatically drops oldest values when maxlen is reached

#### `price_history_b`
- **Type**: `deque` (maxlen = max(spread_history_bars, lookback_window))
- **Description**: Rolling history of closing prices for stock B.
- **Usage**: 
  - Used for hedge ratio regression
  - Must have at least `min_hedge_lookback` values before spread calculation
- **Note**: Automatically drops oldest values when maxlen is reached

#### `spread_history`
- **Type**: `deque` (maxlen = spread_history_bars)
- **Description**: Rolling history of computed spread values.
- **Usage**: 
  - Used for z-score calculation: `zscore = (current_spread - mean(spread_history[-lookback_window:])) / std(spread_history[-lookback_window:])`
  - Used for stationarity tests (ADF, cointegration)
  - Used for volatility adaptation
- **Formula**: `spread = log(price_a) - (hedge_intercept + hedge_ratio * log(price_b))`
- **Note**: Automatically drops oldest values when maxlen is reached

---

### Hedge Ratio Variables

#### `hedge_ratio`
- **Type**: `float`
- **Default**: `1.0`
- **Description**: OLS regression coefficient from `log(price_a) = hedge_ratio * log(price_b) + hedge_intercept`.
- **Usage**: 
  - Determines how many shares of stock B to hedge 1 share of stock A
  - Used in spread calculation: `spread = log(price_a) - (hedge_intercept + hedge_ratio * log(price_b))`
  - Used in position sizing to maintain dollar neutrality
- **Update Frequency**: Refreshed every `hedge_refresh_bars` bars
- **Example**: `hedge_ratio = 0.5` means 2 shares of B hedge 1 share of A

#### `hedge_intercept`
- **Type**: `float`
- **Default**: `0.0`
- **Description**: OLS regression intercept from hedge ratio calculation.
- **Usage**: 
  - Used in spread calculation: `spread = log(price_a) - (hedge_intercept + hedge_ratio * log(price_b))`
  - Captures any constant offset in the price relationship
- **Update Frequency**: Refreshed every `hedge_refresh_bars` bars

#### `bars_since_hedge`
- **Type**: `int`
- **Default**: `0`
- **Description**: Counter tracking bars since last hedge ratio refresh.
- **Usage**: 
  - Incremented each bar
  - When `bars_since_hedge >= hedge_refresh_bars`, triggers `_refresh_hedge_ratio()`
  - Reset to 0 after hedge ratio update

---

### Stationarity Test Variables

#### `adf_pvalue`
- **Type**: `Optional[float]`
- **Default**: `None`
- **Description**: P-value from Augmented Dickey-Fuller test on spread history.
- **Usage**: 
  - If `adf_pvalue > adf_pvalue_threshold`: Entry blocked (spread not stationary)
  - Lower p-value = more stationary (better for mean reversion)
  - `None` if test hasn't been run yet or failed
- **Update Frequency**: Every `stationarity_check_interval` bars (if `stationarity_checks_enabled=True`)

#### `cointegration_pvalue`
- **Type**: `Optional[float]`
- **Default**: `None`
- **Description**: P-value from cointegration test between log prices of stock A and stock B.
- **Usage**: 
  - If `cointegration_pvalue > cointegration_pvalue_threshold`: Entry blocked (stocks not cointegrated)
  - Lower p-value = stronger cointegration (better for pairs trading)
  - `None` if test hasn't been run yet or failed
- **Update Frequency**: Every `stationarity_check_interval` bars (if `stationarity_checks_enabled=True`)

#### `bars_since_stationarity`
- **Type**: `int`
- **Default**: `0`
- **Description**: Counter tracking bars since last stationarity test.
- **Usage**: 
  - Incremented each bar
  - When `bars_since_stationarity >= stationarity_check_interval`, triggers `_update_stationarity_metrics()`
  - Reset to 0 after stationarity tests

---

### Position State Variables

#### `current_position`
- **Type**: `str`
- **Default**: `"flat"`
- **Description**: Current position state for the pair.
- **Values**:
  - `"flat"`: No position
  - `"long_a_short_b"`: Long stock A, short stock B
  - `"short_a_long_b"`: Short stock A, long stock B
- **Usage**: Determines entry/exit logic and signal generation

#### `entry_zscore`
- **Type**: `float`
- **Default**: `0.0`
- **Description**: Z-score at the time of entry (for tracking and exit logic).
- **Usage**: 
  - Stored when position is opened
  - Used in exit logic to determine if z-score got worse (stop-loss)
  - Used in logging and state reporting

#### `entry_timestamp`
- **Type**: `Optional[datetime]`
- **Default**: `None`
- **Description**: Timestamp when position was entered.
- **Usage**: 
  - Stored when position is opened
  - Used for position duration tracking
  - Used in logging and state reporting

#### `entry_prices`
- **Type**: `Dict[str, float]`
- **Default**: `{}`
- **Description**: Dictionary of entry prices for each stock in the pair.
- **Keys**: Stock symbols (e.g., `"AMN"`, `"CORT"`)
- **Values**: Entry price for that stock
- **Usage**: 
  - Used to calculate unrealized PnL: `pnl = (current_price - entry_price) * quantity * direction`
  - Used in exit signal generation

#### `entry_spread`
- **Type**: `Optional[float]`
- **Default**: `None`
- **Description**: Spread value at the time of entry.
- **Usage**: 
  - Used in volatility stop calculation: `|entry_spread - current_spread| >= volatility_stop_multiplier * std_spread`
  - Used in logging and state reporting

#### `entry_quantities`
- **Type**: `Dict[str, int]`
- **Default**: `{}`
- **Description**: Dictionary of signed quantities for each stock in the pair.
- **Keys**: Stock symbols
- **Values**: Signed quantity (positive = long, negative = short)
- **Usage**: 
  - Used to calculate unrealized PnL
  - Used in exit signal generation to determine how many shares to close
- **Example**: `{"AMN": 100, "CORT": -50}` means long 100 AMN, short 50 CORT

#### `entry_notional`
- **Type**: `float`
- **Default**: `0.0`
- **Description**: Gross notional (dollar amount) allocated to this pair position.
- **Usage**: 
  - Used in PnL stop calculation: `unrealized_pnl <= -max_pair_loss_pct * entry_notional`
  - Used in portfolio exposure tracking: `portfolio_notional = sum(entry_notional for all pairs)`
- **Formula**: `entry_notional = qty_a * price_a + qty_b * price_b`

#### `bars_in_trade`
- **Type**: `int`
- **Default**: `0`
- **Description**: Number of bars the position has been held.
- **Usage**: 
  - Incremented each bar when `current_position != "flat"`
  - Exit condition: `bars_in_trade >= max_hold_bars`
  - Reset to 0 when position is closed

#### `cooldown_remaining`
- **Type**: `int`
- **Default**: `0`
- **Description**: Number of bars remaining in cooldown period.
- **Usage**: 
  - Blocks entry when `cooldown_remaining > 0`
  - Decremented each bar when `current_position == "flat"`
  - Set to `cooldown_bars` when position exits with stop-loss or loss-based reasons
- **Note**: Only applies to exits with reasons: `"stop_loss"`, `"pnl_stop"`, `"volatility_stop"`

---

### PnL Tracking Variables

#### `unrealized_pnl`
- **Type**: `float`
- **Default**: `0.0`
- **Description**: Current unrealized profit/loss for the open position.
- **Usage**: 
  - Calculated each bar: `unrealized_pnl = sum((current_price - entry_price) * abs(qty) * direction for each stock)`
  - Exit condition: `unrealized_pnl <= -max_pair_loss_pct * entry_notional`
  - Used in logging and state reporting
- **Note**: Updated in `_calculate_unrealized_pnl()`

#### `realized_pnl`
- **Type**: `float`
- **Default**: `0.0`
- **Description**: Cumulative realized profit/loss from closed positions.
- **Usage**: 
  - Incremented when position closes: `realized_pnl += unrealized_pnl`
  - Tracks total PnL across multiple trades for this pair
  - Used in performance metrics

---

### Statistical Quality Variables

#### `half_life`
- **Type**: `Optional[float]`
- **Default**: `None`
- **Description**: Estimated mean-reversion half-life (in bars) from Ornstein-Uhlenbeck model.
- **Usage**: 
  - Entry gating: If `require_half_life=True`, entry blocked until `half_life` is estimable
  - Exit condition: `half_life > max_halflife_bars` triggers exit
  - Position sizing: `hl_weight = min(1.0, halflife_weight_bars / half_life)` scales position size
- **Interpretation**: 
  - Lower half-life = faster mean reversion = better opportunity
  - Higher half-life = slower mean reversion = less attractive
- **Update Frequency**: Updated each bar in `_update_half_life()`
- **Note**: `None` if cannot be estimated or if spread is not mean-reverting

#### `baseline_spread_std`
- **Type**: `Optional[float]`
- **Default**: `None`
- **Description**: Exponential moving average of spread standard deviation (baseline volatility).
- **Usage**: 
  - Used to compute `volatility_ratio = current_std / baseline_std`
  - Updated using EMA: `baseline_spread_std = alpha * window_std + (1-alpha) * baseline_spread_std`
  - Initialized to first `volatility_window` std when available
- **Note**: Used for volatility adaptation

#### `last_volatility_ratio`
- **Type**: `float`
- **Default**: `1.0`
- **Description**: Most recent volatility ratio (current_std / baseline_std).
- **Usage**: 
  - Used to adjust entry/exit thresholds: `adjusted_threshold = base_threshold * volatility_ratio`
  - Used in position sizing when `volatility_positioning_enabled=True`
  - Clamped between `min_volatility_ratio` and `max_volatility_ratio`
- **Update Frequency**: Updated each bar in `_get_adaptive_thresholds()`

#### `last_zscore`
- **Type**: `Optional[float]`
- **Default**: `None`
- **Description**: Most recently calculated z-score.
- **Usage**: 
  - Used in entry/exit logic
  - Used in state reporting and logging
  - Calculated each bar: `zscore = (current_spread - mean_spread) / std_spread`

---

### Internal Helper Variables

#### `bars_since_entry`
- **Type**: `int`
- **Default**: `0`
- **Description**: Counter tracking bars since last entry (for logging purposes).
- **Usage**: 
  - Incremented each bar
  - Used to control logging frequency (e.g., log z-score every 30 bars)
  - Not used in trading logic

#### `stock_a`
- **Type**: `str`
- **Description**: Symbol of stock A in the pair (first stock).
- **Usage**: Used in signal generation, logging, and state tracking

#### `stock_b`
- **Type**: `str`
- **Description**: Symbol of stock B in the pair (second stock).
- **Usage**: Used in signal generation, logging, and state tracking

---

## Variable Relationships

### Position Sizing Flow

1. **Start**: `base_pair_notional` (e.g., $25,000)
2. **Risk Scaling**: `min(base_pair_notional, risk_budget_per_pair / spread_std)`
3. **Volatility Scaling** (if enabled): `target_notional /= (vol_ratio ** volatility_position_power)`
4. **Half-life Scaling** (if available): `target_notional *= min(1.0, halflife_weight_bars / half_life)`
5. **Signal Strength Scaling**: `target_notional *= signal_strength`
6. **Clamping**: `max(min_pair_notional, min(target_notional, max_pair_notional))`
7. **Portfolio Cap**: `min(target_notional, max_portfolio_notional - current_portfolio_notional)`
8. **Hedge Ratio Split**: 
   - `notional_a = target_notional / (1.0 + hedge_ratio)`
   - `notional_b = target_notional - notional_a`
9. **Final Quantities**: `qty_a = round(notional_a / price_a)`, `qty_b = round(notional_b / price_b)`

### Entry Signal Flow

1. **Data Collection**: Build `price_history_a`, `price_history_b` (need `min_hedge_lookback` bars)
2. **Hedge Ratio**: Calculate/refresh `hedge_ratio` every `hedge_refresh_bars`
3. **Spread Calculation**: `spread = log(price_a) - (hedge_intercept + hedge_ratio * log(price_b))`
4. **Spread History**: Append to `spread_history` (max `spread_history_bars` values)
5. **Z-score**: `zscore = (current_spread - mean(spread_history[-lookback_window:])) / std(...)`
6. **Stationarity Check** (if enabled): Block if `adf_pvalue > adf_pvalue_threshold` or `cointegration_pvalue > cointegration_pvalue_threshold`
7. **Half-life Check** (if `require_half_life=True`): Block if `half_life is None` or `half_life > max_halflife_bars`
8. **Volatility Adaptation**: `adjusted_entry = entry_threshold * volatility_ratio`
9. **Entry Decision**: 
   - If `zscore > adjusted_entry`: Short A, Long B
   - If `zscore < -adjusted_entry`: Long A, Short B
10. **Position Sizing**: Calculate quantities using `_plan_position()`
11. **Portfolio Check**: Ensure `current_portfolio_notional + entry_notional <= max_portfolio_notional`
12. **Cooldown Check**: Block if `cooldown_remaining > 0`

### Exit Signal Flow

1. **Mean Reversion Exit**: If `|zscore| < adjusted_exit` → Exit
2. **Stop Loss Exit**: If `zscore` moves further from mean (worse) → Exit
   - `short_a_long_b` and `zscore > stop_loss_zscore`
   - `long_a_short_b` and `zscore < -stop_loss_zscore`
3. **Time Exit**: If `bars_in_trade >= max_hold_bars` → Exit
4. **PnL Exit**: If `unrealized_pnl <= -max_pair_loss_pct * entry_notional` → Exit
5. **Volatility Exit**: If `|entry_spread - current_spread| >= volatility_stop_multiplier * std_spread` → Exit
6. **Half-life Exit**: If `half_life > max_halflife_bars` → Exit
7. **EOD Exit**: If `timestamp >= market_close - close_before_eod_minutes` → Exit

### Volatility Adaptation Flow

1. **Current Volatility**: `window_std = std(spread_history[-volatility_window:])`
2. **Baseline Update**: `baseline_spread_std = volatility_ema_alpha * window_std + (1 - volatility_ema_alpha) * baseline_spread_std`
3. **Volatility Ratio**: `volatility_ratio = window_std / baseline_spread_std`
4. **Clamping**: 
   - Entry: `volatility_ratio = clip(ratio, min_volatility_ratio, max_volatility_ratio)`
   - Exit: `exit_ratio = clip(ratio, min_exit_volatility_ratio, max_exit_volatility_ratio)`
5. **Threshold Adjustment**:
   - `adjusted_entry = entry_threshold * volatility_ratio`
   - `adjusted_exit = exit_threshold * exit_ratio`

---

## Summary

This strategy uses a sophisticated combination of:
- **Statistical arbitrage**: Mean reversion trading on spread z-scores
- **Dynamic hedging**: OLS-based hedge ratio updated regularly
- **Risk management**: Multiple exit conditions (time, PnL, volatility, half-life)
- **Adaptive sizing**: Volatility-aware and half-life-weighted position sizing
- **Quality gating**: Stationarity tests and half-life requirements before entry
- **Portfolio limits**: Per-pair and portfolio-wide notional caps

All variables work together to create a robust, risk-aware pairs trading system optimized for intraday 5-second bar trading.

