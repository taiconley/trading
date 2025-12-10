# Pairs Trading Kalman Strategy Parameters

**Strategy ID**: `pairs_trading_kalman_v1`  
**Strategy Name**: `Pairs_Trading_Adaptive_Kalman`

Complete reference guide for all parameters in the Pairs Trading Kalman Filter strategy.

---

## Table of Contents

1. [Core Strategy Parameters](#core-strategy-parameters)
2. [Timing & Bar Parameters](#timing--bar-parameters)
3. [Entry/Exit Thresholds](#entryexit-thresholds)
4. [Position Sizing & Risk Management](#position-sizing--risk-management)
5. [Kalman Filter Parameters](#kalman-filter-parameters)
6. [Stationarity & Statistical Checks](#stationarity--statistical-checks)
7. [Volatility Adaptation](#volatility-adaptation)
8. [Order Execution](#order-execution)
9. [Market Hours & Cooldowns](#market-hours--cooldowns)
10. [Advanced Risk Controls](#advanced-risk-controls)

---

## Core Strategy Parameters

### `strategy_id`
**Type**: `string`  
**Current Value**: `"pairs_trading_kalman_v1"`  
**Description**: Unique identifier for this strategy configuration in the database.

**Example**: 
```json
"pairs_trading_kalman_v1"
```

---

### `name`
**Type**: `string`  
**Current Value**: `"Pairs_Trading_Adaptive_Kalman"`  
**Description**: Human-readable name for the strategy.

**Example**: 
```json
"Pairs_Trading_Adaptive_Kalman"
```

---

### `symbols`
**Type**: `array of strings`  
**Current Value**: `["ITW", "PKG", "WEC", "CMS"]`  
**Description**: List of all stock symbols traded by this strategy. Must include all symbols from all pairs.

**Example**: 
```json
["ITW", "PKG", "WEC", "CMS"]
```
These 4 symbols form 2 pairs: ITW/PKG and WEC/CMS

---

### `pairs`
**Type**: `array of [string, string]`  
**Current Value**: `[["ITW","PKG"], ["WEC","CMS"]]`  
**Description**: List of stock pairs to trade. Each pair is an array of two symbols. The strategy monitors the spread between each pair and trades mean-reversion opportunities.

**Example**: 
```json
[
  ["ITW", "PKG"],    // Pair 1: Illinois Tool Works / Packaging Corp
  ["WEC", "CMS"]     // Pair 2: WEC Energy / CMS Energy
]
```

**Trading Logic**: 
- When spread is **too low** → Long stock A, Short stock B (A is relatively cheap)
- When spread is **too high** → Short stock A, Long stock B (A is relatively expensive)
- When spread returns to **mean** → Close position and take profit

---

## Timing & Bar Parameters

### `bar_timeframe`
**Type**: `string`  
**Current Value**: `"5 secs"`  
**Description**: The base timeframe for raw market data bars. This is the frequency at which price data arrives from the market.

**Example**: 
```
"5 secs" = One bar every 5 seconds
```

**Daily Volume**: 6.5 market hours × 12 bars/minute × 60 minutes = **4,680 bars per day**

---

### `stats_aggregation_seconds`
**Type**: `integer` (seconds)  
**Current Value**: `180`  
**Description**: How many seconds of raw bars to aggregate into one "spread bar" for statistical calculations. Must be a multiple of `bar_timeframe` (5 seconds).

**Why it exists**: Reduces noise in spread calculations by smoothing over multiple price observations.

**Calculation**: 
```
stats_aggregation_seconds = 180
→ 180 ÷ 5 = 36 raw bars per spread bar
→ Each spread bar represents 3 minutes of data
```

**Visual**:
```
Raw bars: |||||||||||||||||||||||||||||||||||||  (180 seconds = 36 bars)
           ↓ aggregation ↓
Spread bar:              ▓▓▓▓▓▓▓▓▓              (1 aggregated bar)
```

**Tuning**:
- **Lower** (60-120s): More responsive, noisier signals
- **Higher** (300-600s): Smoother, more stable, slower to react

---

### `lookback_window`
**Type**: `integer` (spread bars)  
**Current Value**: `30`  
**Description**: Number of **spread bars** (NOT raw bars!) used to calculate the rolling mean and standard deviation for z-score calculations.

**Critical**: This is measured in spread bars (aggregated), not raw bars!

**Calculation**: 
```
lookback_window = 30 spread bars
stats_aggregation_seconds = 180
→ 30 × 180 seconds = 5,400 seconds = 90 minutes of price history
```

**Formula**:
```python
z_score = (current_spread - rolling_mean) / rolling_std

where:
  rolling_mean = mean of last 30 spread bars
  rolling_std = std dev of last 30 spread bars
```

**Tuning**:
- **Lower** (10-20): More sensitive, faster reactions, more trades, higher risk
- **Higher** (40-60): More stable, fewer trades, catches longer mean-reversion cycles

---

### `spread_history_bars`
**Type**: `integer` (spread bars)  
**Current Value**: `40`  
**Description**: Maximum size of the spread history buffer (rolling window). This is the deque size that stores historical spread values.

**Why it exists**: 
1. Limits memory usage
2. Keeps calculations focused on recent market conditions
3. Provides context beyond the immediate lookback window

**Requirement**: Must be ≥ `lookback_window` (currently 30)

**Example**: 
```
spread_history_bars = 40
→ Stores up to 40 spread bars
→ At 180s per spread bar = 7,200 seconds = 2 hours of spread history
```

**Tuning**: Generally set to 100-200 for flexibility. Your value of 40 is minimal but acceptable.

---

### `min_hedge_lookback`
**Type**: `integer` (raw bars)  
**Current Value**: `40`  
**Description**: Minimum number of **raw price bars** required before calculating the hedge ratio (beta) using linear regression.

**Critical**: This is measured in raw bars, not spread bars!

**Why it exists**: Linear regression needs sufficient data points to produce a reliable hedge ratio. Too few points = unstable, noisy hedge ratios.

**Calculation**: 
```
min_hedge_lookback = 40 raw bars
→ 40 × 5 seconds = 200 seconds = 3.3 minutes
```

**⚠️ WARNING**: Your current value of 40 is **very low**!

**Recommended**: 120-180 raw bars minimum
```
120 bars = 10 minutes of price data
180 bars = 15 minutes of price data
```

**Statistical Significance**: 
- **< 100 bars**: High risk of unstable hedge ratios
- **100-150 bars**: Acceptable for short-term trading
- **150+ bars**: Recommended for stable hedge ratios

---

### `hedge_refresh_bars`
**Type**: `integer` (raw bars)  
**Current Value**: `180`  
**Description**: How many **raw bars** to wait before recalculating the hedge ratio using linear regression on price history.

**Why it exists**: Hedge ratios change slowly. Recalculating every bar wastes computation and can introduce noise.

**Calculation**: 
```
hedge_refresh_bars = 180 raw bars
→ 180 × 5 seconds = 900 seconds = 15 minutes
```

**Relationship to Spread Bars**: 
```
180 raw bars ÷ 36 raw bars per spread = 5 spread bars
→ Hedge ratio updates every ~5 spread bars
```

**Tuning**:
- **Lower** (60-120 bars / 5-10 min): More responsive to correlation changes
- **Higher** (240-480 bars / 20-40 min): More stable, less computation

---

## Entry/Exit Thresholds

### `entry_threshold`
**Type**: `float` (z-score)  
**Current Value**: `2.0`  
**Description**: Z-score threshold to trigger a trade entry. When the absolute value of the spread's z-score exceeds this value, the strategy enters a position.

**Formula**:
```python
if abs(z_score) >= entry_threshold:
    if z_score > 0:
        # Spread too high → Short A, Long B
    else:
        # Spread too low → Long A, Short B
```

**Example**: 
```
entry_threshold = 2.0
→ Enter trade when spread is 2+ standard deviations from mean
```

**Interpretation**:
- **Z-score = +2.0**: Spread is 2 std devs above mean (stock A expensive relative to B)
- **Z-score = -2.0**: Spread is 2 std devs below mean (stock A cheap relative to B)

**Tuning**:
- **Lower** (1.0-1.5): More trades, more frequent entries, lower win rate
- **Higher** (2.5-3.5): Fewer trades, higher confidence, higher win rate

**Statistical Meaning**:
```
Z-score = 1.0 → ~68% probability within normal range
Z-score = 2.0 → ~95% probability within normal range (current setting)
Z-score = 3.0 → ~99.7% probability within normal range
```

---

### `exit_threshold`
**Type**: `float` (z-score)  
**Current Value**: `0.5`  
**Description**: Z-score threshold to exit a profitable position. When the absolute z-score falls below this value, the strategy closes the position.

**Formula**:
```python
if in_position and abs(z_score) <= exit_threshold:
    close_position()  # Take profit
```

**Example**: 
```
exit_threshold = 0.5
→ Exit when spread returns to within 0.5 std devs of mean
```

**Tuning**:
- **Lower** (0.0-0.3): Exit closer to mean, take smaller profits, higher win rate
- **Higher** (0.7-1.0): Wait for deeper reversion, larger profits, risk of reversals

**Trade Example**:
```
Time 0: z_score = -2.5 → Enter (long A, short B)
Time 1: z_score = -1.8 → Hold (still below entry threshold)
Time 2: z_score = -0.8 → Hold (above exit threshold)
Time 3: z_score = -0.4 → EXIT (below exit threshold) ✓
```

---

### `stop_loss_zscore`
**Type**: `float` (z-score)  
**Current Value**: `3.5`  
**Description**: Emergency exit threshold when the trade goes against you. If the absolute z-score exceeds this value while in a position, force exit to prevent further losses.

**Formula**:
```python
if in_position:
    if (long_a_short_b and z_score >= stop_loss_zscore) or \
       (short_a_long_b and z_score <= -stop_loss_zscore):
        close_position()  # Stop loss!
```

**Example**: 
```
stop_loss_zscore = 3.5
entry_threshold = 2.0

Scenario: Entered at z = -2.0 (long A, short B)
If z moves to -3.5 or worse → Force exit
```

**Requirement**: Must be > `entry_threshold`

**Tuning**:
- **Tighter** (2.5-3.0): Limit losses but may exit prematurely
- **Wider** (4.0-5.0): Allow more room but larger potential losses

---

## Position Sizing & Risk Management

### `position_size`
**Type**: `integer` (shares)  
**Current Value**: `48`  
**Description**: **Legacy parameter** - Number of shares per leg when using fixed position sizing. This is primarily used as a fallback if notional-based sizing fails.

**Current Usage**: The strategy now uses notional-based position sizing (see `base_pair_notional`), so this parameter is less important.

**Example**: 
```
position_size = 48
→ Buy 48 shares of stock A
→ Sell 48 shares of stock B
```

---

### `base_pair_notional`
**Type**: `float` (dollars)  
**Current Value**: `25000`  
**Description**: Baseline gross notional (dollar value) allocated to each pair trade. This is the starting point for position size calculations before adjustments.

**Formula**:
```python
target_notional = base_pair_notional
# Then apply adjustments:
target_notional *= volatility_adjustment
target_notional *= half_life_adjustment
target_notional *= signal_strength
```

**Example**: 
```
base_pair_notional = $25,000
Stock A price = $100
Stock B price = $50

Gross notional = $25,000
→ Buy ~$12,500 of stock A (125 shares × $100)
→ Sell ~$12,500 of stock B (250 shares × $50)
```

**Tuning**:
- **Lower** ($10k-$20k): Smaller positions, less capital at risk
- **Higher** ($30k-$50k): Larger positions, more profit potential

---

### `min_pair_notional`
**Type**: `float` (dollars)  
**Current Value**: `5000`  
**Description**: Minimum gross notional required to enter a trade. If calculated position size falls below this, the trade is skipped.

**Why it exists**: Prevents tiny positions that aren't worth the transaction costs and tracking overhead.

**Example**: 
```
min_pair_notional = $5,000
calculated_notional = $3,500
→ Skip trade (below minimum)
```

---

### `max_pair_notional`
**Type**: `float` (dollars)  
**Current Value**: `100000`  
**Description**: Maximum gross notional allowed for a single pair. Position size calculations are capped at this value.

**Why it exists**: Limits exposure to any single pair to control risk concentration.

**Example**: 
```
max_pair_notional = $100,000
calculated_notional = $150,000
→ Cap at $100,000 (safety limit)
```

---

### `max_portfolio_notional`
**Type**: `float` (dollars)  
**Current Value**: `300000`  
**Description**: Portfolio-wide cap on total simultaneous pair exposure. Sum of all active positions cannot exceed this value.

**Why it exists**: Controls overall portfolio leverage and risk.

**Example**: 
```
max_portfolio_notional = $300,000

Active positions:
- Pair 1 (ITW/PKG): $50,000
- Pair 2 (WEC/CMS): $40,000
Total: $90,000 (OK, below $300k)

New signal for Pair 3: $80,000
Can enter? Yes ($90k + $80k = $170k < $300k)
```

---

### `volatility_positioning_enabled`
**Type**: `boolean`  
**Current Value**: `true`  
**Description**: Enable volatility-aware position sizing. When true, position sizes are scaled inversely with volatility (lower size in high volatility environments).

**Formula**:
```python
if volatility_positioning_enabled:
    target_notional /= (volatility_ratio ** volatility_position_power)
```

**Example**: 
```
volatility_positioning_enabled = true
base_pair_notional = $25,000
current_volatility / baseline_volatility = 1.3 (30% higher than normal)
volatility_position_power = 1.0

Adjusted notional = $25,000 / 1.3 = $19,230
→ Reduce position size by 23% due to elevated volatility
```

**Risk Management**: This prevents oversizing in choppy markets.

---

### `volatility_position_power`
**Type**: `float` (exponent)  
**Current Value**: `1.0`  
**Description**: Exponent applied when scaling position size by volatility ratio. Controls how aggressively size adjusts to volatility changes.

**Formula**:
```python
adjustment = 1 / (volatility_ratio ** power)
```

**Example**: 
```
volatility_ratio = 1.5 (50% higher volatility)

power = 0.5: adjustment = 1 / 1.5^0.5 = 0.82  (18% reduction)
power = 1.0: adjustment = 1 / 1.5^1.0 = 0.67  (33% reduction) ← current
power = 1.5: adjustment = 1 / 1.5^1.5 = 0.54  (46% reduction)
```

**Tuning**:
- **Lower** (0.5-0.8): Gentler size adjustments
- **Higher** (1.2-2.0): More aggressive risk reduction in volatile markets

---

### `risk_budget_per_pair`
**Type**: `float` (dollars)  
**Current Value**: `1000`  
**Description**: Dollar risk budget per pair. Position size is calculated as risk_budget divided by spread volatility.

**Formula**:
```python
target_notional = risk_budget_per_pair / spread_std
```

**Example**: 
```
risk_budget_per_pair = $1,000
spread_std = 0.05 (5% typical spread volatility)

Position size = $1,000 / 0.05 = $20,000
```

**Interpretation**: This ensures each trade risks approximately $1,000 based on typical spread movements.

---

### `halflife_weight_bars`
**Type**: `integer` (bars)  
**Current Value**: `240`  
**Description**: Reference anchor for half-life weighting. Position sizes are adjusted based on the ratio of estimated half-life to this value.

**Formula**:
```python
halflife_weight = min(1.0, halflife_weight_bars / estimated_halflife)
target_notional *= halflife_weight
```

**Example**: 
```
halflife_weight_bars = 240 (reference: 20 minutes at 5-sec bars)
estimated_halflife = 120 bars (10 minutes)

Weight = min(1.0, 240/120) = 1.0
→ Full position size (fast mean reversion)

estimated_halflife = 480 bars (40 minutes)
Weight = min(1.0, 240/480) = 0.5
→ Reduce position size by 50% (slower mean reversion)
```

**Logic**: Faster mean-reversion (lower half-life) = larger positions

---

### `max_halflife_bars`
**Type**: `integer` (bars)  
**Current Value**: `720`  
**Description**: Maximum acceptable half-life for entering a trade. If estimated half-life exceeds this, skip the trade.

**Why it exists**: Very slow mean reversion = holding risk and capital tied up too long.

**Example**: 
```
max_halflife_bars = 720 (60 minutes at 5-sec bars)
estimated_halflife = 900 bars (75 minutes)

→ Skip trade (mean reversion too slow)
```

---

### `require_half_life`
**Type**: `boolean`  
**Current Value**: `true`  
**Description**: If true, require a valid half-life estimate before entering trades. Prevents trading pairs that don't show mean-reverting behavior.

**Example**: 
```
require_half_life = true
half_life = None (not enough data yet)

→ Skip trade (wait for half-life calculation)
```

---

### `max_hold_bars`
**Type**: `integer` (raw bars)  
**Current Value**: `28800`  
**Description**: Maximum number of raw bars to hold a position before forcing an exit. This is a time-based stop.

**Calculation**: 
```
max_hold_bars = 28,800
→ 28,800 × 5 seconds = 144,000 seconds = 40 hours
→ Across multiple trading days (6.5 hrs/day = ~6 days max hold)
```

**⚠️ Note**: Your current value of 28,800 is **extremely high** for an intraday strategy!

**Typical for intraday**:
```
720 bars = 1 hour
1,440 bars = 2 hours
3,600 bars = 5 hours (one trading day)
```

**Why it exists**: Prevents positions from becoming stale and tying up capital indefinitely.

---

## Kalman Filter Parameters

### `use_kalman`
**Type**: `boolean`  
**Current Value**: `true`  
**Description**: Enable Kalman Filter for adaptive hedge ratio estimation. When true, uses Kalman filtering instead of ordinary least squares (OLS) regression.

**Benefits**:
- Adapts to changing correlations in real-time
- Smooths noise while remaining responsive
- Continuously updates hedge ratio with each new price observation

**Example**: 
```
use_kalman = true
→ Hedge ratio updates with every aggregated bar using Kalman filter
→ More adaptive to market regime changes
```

---

### `kalman_delta`
**Type**: `float` (covariance)  
**Current Value**: `0.01`  
**Description**: System noise covariance (process noise) in the Kalman Filter. Controls how much the hedge ratio is expected to change over time.

**Tuning**:
- **Lower** (1e-5 to 1e-3): Assumes stable hedge ratio, slower adaptation
- **Higher** (0.01 to 0.1): Allows faster hedge ratio changes, more responsive

**Example**: 
```
kalman_delta = 0.01 (moderate adaptation speed)
```

**Interpretation**: Higher delta = Kalman filter trusts new measurements more, adjusts faster.

---

### `kalman_R`
**Type**: `float` (variance)  
**Current Value**: `0.1`  
**Description**: Measurement noise variance in the Kalman Filter. Represents the expected noise level in price observations.

**Tuning**:
- **Lower** (0.01-0.05): Trusts measurements more, responds faster to price changes
- **Higher** (0.1-1.0): Filters out noise more aggressively, smoother but slower

**Example**: 
```
kalman_R = 0.1 (moderate noise filtering)
```

**Relationship**:
```
delta/R ratio determines adaptation speed:
- High delta, low R = Fast adaptation
- Low delta, high R = Slow, stable adaptation
```

---

## Stationarity & Statistical Checks

### `stationarity_checks_enabled`
**Type**: `boolean`  
**Current Value**: `true`  
**Description**: Enable periodic statistical tests to verify the spread is stationary (mean-reverting). Uses Augmented Dickey-Fuller (ADF) test and cointegration tests.

**Why it exists**: Pairs trading only works if the spread is stationary. Non-stationary spreads = trending relationships = strategy failure.

**Example**: 
```
stationarity_checks_enabled = true
→ Run ADF test every 60 bars
→ Skip trades if spread fails stationarity tests
```

---

### `adf_pvalue_threshold`
**Type**: `float` (probability)  
**Current Value**: `0.05`  
**Description**: Maximum acceptable p-value for the Augmented Dickey-Fuller (ADF) stationarity test. Lower p-value = stronger evidence of stationarity.

**Statistical Meaning**:
```
p-value < 0.05 → Reject null hypothesis → Spread IS stationary ✓
p-value > 0.05 → Cannot reject null → Spread NOT stationary ✗
```

**Example**: 
```
adf_pvalue_threshold = 0.05 (95% confidence level)
measured_pvalue = 0.03 → Pass (spread is stationary)
measured_pvalue = 0.12 → Fail (spread not stationary, skip trades)
```

---

### `cointegration_pvalue_threshold`
**Type**: `float` (probability)  
**Current Value**: `0.05`  
**Description**: Maximum acceptable p-value for cointegration test between the two stocks in a pair.

**Why it exists**: Cointegration = long-term statistical relationship. Strong cointegration = reliable pairs trading.

**Example**: 
```
cointegration_pvalue_threshold = 0.05
measured_pvalue = 0.01 → Pass (strongly cointegrated)
measured_pvalue = 0.18 → Fail (no cointegration, skip trades)
```

---

### `stationarity_check_interval`
**Type**: `integer` (raw bars)  
**Current Value**: `60`  
**Description**: How many raw bars to wait between stationarity checks (ADF and cointegration tests).

**Why not every bar**: Statistical tests are computationally expensive and stationarity changes slowly.

**Calculation**: 
```
stationarity_check_interval = 60 bars
→ 60 × 5 seconds = 300 seconds = 5 minutes between checks
```

**Tuning**:
- **More frequent** (30-60 bars): Faster detection of regime changes
- **Less frequent** (120-300 bars): Lower computational overhead

---

## Volatility Adaptation

### `volatility_adaptation_enabled`
**Type**: `boolean`  
**Current Value**: `true`  
**Description**: Enable dynamic adjustment of entry/exit thresholds based on current volatility relative to baseline.

**Why it exists**: In high volatility, wider thresholds prevent premature entries. In low volatility, tighter thresholds capture smaller moves.

**Formula**:
```python
if volatility_adaptation_enabled:
    adjusted_entry = entry_threshold * volatility_ratio
    adjusted_exit = exit_threshold * volatility_ratio
```

**Example**: 
```
entry_threshold = 2.0 (base)
current_volatility / baseline = 1.4 (40% higher)

Adjusted entry = 2.0 × 1.4 = 2.8
→ Need larger deviation to enter in volatile markets
```

---

### `volatility_window`
**Type**: `integer` (raw bars)  
**Current Value**: `15`  
**Description**: Number of spread bars used to calculate current spread volatility for adaptation.

**⚠️ Note**: Despite the name, this is actually **spread bars** not raw bars in the code!

**Calculation**: 
```
volatility_window = 15 spread bars
stats_aggregation_seconds = 180

→ 15 × 180 = 2,700 seconds = 45 minutes of spread data
```

**Tuning**:
- **Shorter** (5-10 bars): Rapid adaptation to volatility spikes
- **Longer** (20-40 bars): Smoother, more stable volatility estimates

---

### `volatility_ema_alpha`
**Type**: `float` (smoothing factor)  
**Current Value**: `0.2`  
**Description**: Exponential moving average (EMA) smoothing parameter for baseline volatility calculation. Controls how quickly the baseline adapts.

**Formula**:
```python
baseline_vol = alpha * new_vol + (1 - alpha) * old_baseline
```

**Example**: 
```
volatility_ema_alpha = 0.2
new_volatility = 0.08
old_baseline = 0.05

New baseline = 0.2 × 0.08 + 0.8 × 0.05 = 0.056
```

**Tuning**:
- **Lower** (0.05-0.1): Slower baseline adjustment, more stable
- **Higher** (0.3-0.5): Faster baseline adjustment, more reactive

---

### `min_volatility_ratio`
**Type**: `float` (ratio)  
**Current Value**: `0.75`  
**Description**: Lower bound clamp for volatility ratio adjustments. Prevents thresholds from becoming too tight in extremely low volatility.

**Formula**:
```python
volatility_ratio = max(min_volatility_ratio, current_vol / baseline_vol)
```

**Example**: 
```
min_volatility_ratio = 0.75
current_vol / baseline = 0.5 (very low volatility)

Clamped ratio = max(0.75, 0.5) = 0.75
→ Don't reduce thresholds below 75% of base values
```

---

### `max_volatility_ratio`
**Type**: `float` (ratio)  
**Current Value**: `1.5`  
**Description**: Upper bound clamp for volatility ratio adjustments. Prevents thresholds from becoming unreasonably wide in high volatility.

**Formula**:
```python
volatility_ratio = min(max_volatility_ratio, current_vol / baseline_vol)
```

**Example**: 
```
max_volatility_ratio = 1.5
current_vol / baseline = 2.3 (very high volatility)

Clamped ratio = min(1.5, 2.3) = 1.5
→ Don't widen thresholds beyond 150% of base values
```

---

### `min_exit_volatility_ratio`
**Type**: `float` (ratio)  
**Current Value**: `0.8`  
**Description**: Lower bound for volatility adjustments specifically for exit thresholds.

**Why separate from entry**: You may want different adaptation behavior for exits vs entries.

**Example**: 
```
min_exit_volatility_ratio = 0.8
→ Exit thresholds won't go below 80% of base values
```

---

### `max_exit_volatility_ratio`
**Type**: `float` (ratio)  
**Current Value**: `1.3`  
**Description**: Upper bound for volatility adjustments specifically for exit thresholds.

**Example**: 
```
max_exit_volatility_ratio = 1.3
→ Exit thresholds won't exceed 130% of base values
```

---

## Order Execution

### `execution_type`
**Type**: `string`  
**Current Value**: `"MKT"`  
**Description**: Order type for trade execution. Determines how orders are submitted to the broker.

**Options**:
- **`"MKT"`**: Market order - immediate execution at current market price
- **`"LMT"`**: Limit order - only execute at specified price or better
- **`"ADAPTIVE"`**: Interactive Brokers Adaptive algo - balances speed vs price
- **`"PEG BEST"`**: Pegged to best bid/ask
- **`"PEG MID"`**: Pegged to midpoint

**Current Setting**: `"MKT"` = Fast execution, accept slippage

**Trade-offs**:
- **MKT**: Fast, guaranteed fill, but may have slippage
- **LMT**: Better price control, but risk of no fill
- **ADAPTIVE**: Good balance for liquid stocks

---

## Market Hours & Cooldowns

### `market_close_hour`
**Type**: `integer` (hour in 24-hour format)  
**Current Value**: `16`  
**Description**: Market close hour in the specified timezone (US/Eastern). Used to determine end of trading day.

**Example**: 
```
market_close_hour = 16
→ 4:00 PM Eastern Time (US market close)
```

---

### `market_close_minute`
**Type**: `integer` (minute)  
**Current Value**: `0`  
**Description**: Market close minute.

**Example**: 
```
market_close_hour = 16
market_close_minute = 0
→ Market closes at 4:00 PM exactly
```

---

### `close_before_eod_minutes`
**Type**: `integer` (minutes)  
**Current Value**: `5`  
**Description**: Close all positions this many minutes before market close. Prevents holding overnight positions.

**Example**: 
```
close_before_eod_minutes = 5
market_close = 4:00 PM
→ Force close all positions at 3:55 PM
```

**Why it exists**: Ensures clean slate at market close, avoids overnight risk and gap risk.

---

### `cooldown_bars`
**Type**: `integer` (raw bars)  
**Current Value**: `180`  
**Description**: Number of raw bars to wait after closing a position before allowing re-entry on the same pair.

**Calculation**: 
```
cooldown_bars = 180
→ 180 × 5 seconds = 900 seconds = 15 minutes
```

**Why it exists**: Prevents rapid-fire entries and exits on the same pair, which can rack up transaction costs and indicates unstable conditions.

**Example**: 
```
Time 0: Close position on ITW/PKG
Time 1-180: Cooldown period (ignore new signals)
Time 181+: Can enter ITW/PKG again if signal triggers
```

---

### `cooldown_after_all_exits`
**Type**: `boolean`  
**Current Value**: `true`  
**Description**: If true, apply cooldown after ANY exit (profitable or loss). If false, behavior depends on `cooldown_on_loss_only`.

**Example**: 
```
cooldown_after_all_exits = true
→ Always apply cooldown, regardless of profit/loss
```

**Risk Management**: Prevents overtrading and allows market conditions to stabilize.

---

### `timezone`
**Type**: `string`  
**Current Value**: `"US/Eastern"`  
**Description**: Timezone for market hours and timestamp calculations. Must be a valid IANA timezone identifier.

**Example**: 
```
timezone = "US/Eastern"
→ All timestamps and market hours in Eastern Time
→ Market open: 9:30 AM ET, Close: 4:00 PM ET
```

---

## Advanced Risk Controls

### `max_pair_loss_pct`
**Type**: `float` (percentage as decimal)  
**Current Value**: `0.02`  
**Description**: Maximum loss allowed as a percentage of allocated notional before forcing exit. Acts as a percentage-based stop loss.

**Formula**:
```python
if abs(unrealized_pnl) / entry_notional >= max_pair_loss_pct:
    force_exit()  # Loss limit exceeded
```

**Example**: 
```
max_pair_loss_pct = 0.02 (2%)
entry_notional = $25,000

Max loss = $25,000 × 0.02 = $500
If position loses $500 or more → Force exit
```

**Risk Management**: Hard dollar-based stop independent of z-scores.

---

### `volatility_stop_multiplier`
**Type**: `float` (multiple)  
**Current Value**: `2.5`  
**Description**: Exit if spread deviates more than this many standard deviations from the entry spread. Prevents holding positions in extreme moves.

**Formula**:
```python
if abs(current_spread - entry_spread) > volatility_stop_multiplier * spread_std:
    force_exit()  # Volatility stop triggered
```

**Example**: 
```
volatility_stop_multiplier = 2.5
entry_spread = 0.10
spread_std = 0.04

Exit if spread moves beyond:
0.10 ± (2.5 × 0.04) = 0.10 ± 0.10
→ Exit if spread < 0.00 or > 0.20
```

---

## Summary: Quick Reference

### Critical Ratios & Relationships

**Warmup Requirements**:
```
Total warmup (raw bars) = min_hedge_lookback + (lookback_window × stats_aggregation_bars)
Current: 40 + (30 × 36) = 1,120 raw bars = ~93 minutes
```

**Data Buffer Sizing**:
```
price_history maxlen = max(spread_history_bars, lookback_window, min_hedge_lookback)
Current: max(40, 30, 40) = 40 bars
```

**Time Scales**:
```
1 raw bar = 5 seconds
1 spread bar = 180 seconds = 36 raw bars = 3 minutes
lookback_window = 30 spread bars = 90 minutes
hedge_refresh = 180 raw bars = 15 minutes
```

---

## Recommended Parameter Adjustments

Based on analysis of your current configuration:

### ⚠️ Critical Issues

1. **`min_hedge_lookback: 40`** is too low
   - **Recommendation**: Increase to 120-180 bars
   - **Risk**: Unstable hedge ratios with only 3 minutes of data

2. **`max_hold_bars: 28800`** is too high for intraday
   - **Recommendation**: Reduce to 1,440-3,600 bars (2-5 hours)
   - **Risk**: Positions held for days (not truly intraday)

3. **`spread_history_bars: 40`** is minimal
   - **Recommendation**: Increase to 100-200 bars
   - **Risk**: Limited statistical context

### ✅ Well-Configured

- `entry_threshold: 2.0` - Good balance
- `exit_threshold: 0.5` - Reasonable profit taking
- `stats_aggregation_seconds: 180` - Appropriate for 5-sec bars
- `kalman_delta/R` - Moderate adaptation settings
- `base_pair_notional: 25000` - Reasonable position sizing

---

**Last Updated**: December 5, 2025  
**Strategy Version**: pairs_trading_kalman_v1

