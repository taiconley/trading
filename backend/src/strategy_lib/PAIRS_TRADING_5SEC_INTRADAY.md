# Pairs Trading Strategy - 5-Second Intraday Version

## Overview

The pairs trading strategy has been updated for **5-second bar intraday trading**. This version:
- ✅ Operates on 5-second bars instead of daily data
- ✅ Closes all positions before market close (no overnight holds)
- ✅ Monitors 9 pairs simultaneously
- ✅ Tracks position duration in bars, not days

---

## Key Changes from Previous Version

### 1. **Time-Based Changes**
- **Lookback Window**: 20 bars → **240 bars** (20 minutes at 5-sec intervals)
- **Max Hold Time**: 20 days → **720 bars** (1 hour at 5-sec intervals)
- **Position Tracking**: `days_in_trade` → `bars_in_trade`

### 2. **Intraday-Only Trading**
- **Market Close**: 4:00 PM Eastern (configurable)
- **Close Before EOD**: 5 minutes (configurable)
- **Force Exit**: All positions automatically closed at 3:55 PM
- **No Overnight Risk**: Ensures flat positions by market close

### 3. **Default Pairs Configuration**
The strategy now includes 9 pairs across multiple sectors:

```python
pairs = [
    ["AAPL", "MSFT"],   # Tech Large Cap
    ["JPM", "BAC"],     # Banks
    ["GS", "MS"],       # Investment Banks
    ["XOM", "CVX"],     # Energy
    ["V", "MA"],        # Payments
    ["KO", "PEP"],      # Beverages
    ["WMT", "TGT"],     # Retail
    ["PFE", "MRK"],     # Pharma
    ["DIS", "NFLX"]     # Media
]
```

---

## Configuration Parameters

### Time-Based Parameters (Adjusted for 5-Second Bars)

| Parameter | Default | Description | Notes |
|-----------|---------|-------------|-------|
| `lookback_window` | 240 | Bars for mean/std calculation | 240 bars = 20 minutes |
| `max_hold_bars` | 720 | Maximum bars to hold position | 720 bars = 1 hour |
| `entry_threshold` | 2.0 | Z-score to enter trade | Higher = fewer trades |
| `exit_threshold` | 0.5 | Z-score to exit trade | Lower = faster exits |
| `stop_loss_zscore` | 3.0 | Z-score for stop loss | Prevents runaway losses |

### Intraday Trading Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `market_close_hour` | 16 | Market close hour (4 PM Eastern) |
| `market_close_minute` | 0 | Market close minute |
| `close_before_eod_minutes` | 5 | Close positions N minutes before close |

### Position Sizing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `position_size` | 100 | Shares per leg of pair |

---

## Exit Conditions

The strategy has **4 exit conditions** (in order of priority):

### 1. **End of Day (Highest Priority)**
- **Trigger**: Current time >= (market_close - close_before_eod_minutes)
- **Default**: 3:55 PM Eastern
- **Reason**: "end_of_day"
- **Action**: Force close ALL positions regardless of P&L

### 2. **Mean Reversion (Normal Exit)**
- **Trigger**: `|z-score| < exit_threshold`
- **Default**: |z-score| < 0.5
- **Reason**: "mean_reversion"
- **Action**: Close position (take profit)

### 3. **Stop Loss (Risk Management)**
- **Trigger**: Z-score moves against position beyond `stop_loss_zscore`
- **Default**: |z-score| > 3.0 in wrong direction
- **Reason**: "stop_loss"
- **Action**: Close position (cut losses)

### 4. **Maximum Hold Time**
- **Trigger**: `bars_in_trade >= max_hold_bars`
- **Default**: 720 bars (1 hour)
- **Reason**: "max_hold_time"
- **Action**: Close position (prevent stale positions)

---

## Running the Strategy

### Step 1: Prepare Historical Data

Ensure you have 5-second historical data for all pairs:

```bash
# Add symbols to watchlist
for symbol in AAPL MSFT JPM BAC GS MS XOM CVX V MA KO PEP WMT TGT PFE MRK DIS NFLX; do
  curl -X POST http://localhost:8000/api/watchlist \
    -H "Content-Type: application/json" \
    -d "{\"symbol\": \"$symbol\", \"action\": \"add\"}"
done

# Download 5-second historical data (30 days)
docker exec trading-historical python -m src.services.historical.main bulk \
  --bar-size "5 secs" \
  --lookback "30 D"
```

### Step 2: Verify Data in Database

```bash
docker exec -it trading-postgres psql -U bot -d trading

# Check data availability
SELECT symbol, tf, COUNT(*) as bar_count, 
       MIN(ts) as earliest, MAX(ts) as latest
FROM candles 
WHERE tf = '5 secs'
GROUP BY symbol, tf
ORDER BY symbol;

# Should see ~23,400 bars per day per symbol (6.5 hour trading day)
```

---

## Optimization

### Recommended Parameter Ranges

Based on 5-second bar characteristics, here are good parameter ranges:

```python
param_ranges = {
    # Lookback: 10 mins to 1 hour
    "lookback_window": [120, 240, 360, 720],  # 10min, 20min, 30min, 1hr
    
    # Entry threshold: more conservative for 5-sec data
    "entry_threshold": [1.5, 2.0, 2.5, 3.0],
    
    # Exit threshold: tighter for quick mean reversion
    "exit_threshold": [0.3, 0.5, 0.7, 1.0],
    
    # Max hold: 15 mins to 2 hours
    "max_hold_bars": [180, 360, 720, 1440],  # 15min, 30min, 1hr, 2hr
    
    # Stop loss: wider range for volatility
    "stop_loss_zscore": [2.5, 3.0, 3.5, 4.0]
}
```

### Run Grid Search Optimization

```bash
cd /home/taiconley/Desktop/Projects/trading/backend

# Grid search (exhaustive)
python -m src.services.optimizer.main optimize \
  --strategy Pairs_Trading \
  --config '{
    "pairs": [
      ["AAPL", "MSFT"],
      ["JPM", "BAC"],
      ["GS", "MS"]
    ]
  }' \
  --start-date "2024-09-01" \
  --end-date "2024-10-22" \
  --param-ranges '{
    "lookback_window": [120, 240, 360],
    "entry_threshold": [1.5, 2.0, 2.5],
    "exit_threshold": [0.3, 0.5, 0.7],
    "max_hold_bars": [360, 720, 1440]
  }' \
  --algorithm grid \
  --objective sharpe_ratio \
  --max-workers 4
```

### Run Bayesian Optimization (Recommended)

```bash
# Bayesian optimization (intelligent search)
python -m src.services.optimizer.main optimize \
  --strategy Pairs_Trading \
  --config '{
    "pairs": [
      ["AAPL", "MSFT"],
      ["JPM", "BAC"],
      ["GS", "MS"],
      ["XOM", "CVX"],
      ["V", "MA"]
    ]
  }' \
  --start-date "2024-09-01" \
  --end-date "2024-10-22" \
  --param-ranges '{
    "lookback_window": [120, 240, 360, 720],
    "entry_threshold": [1.5, 2.0, 2.5, 3.0],
    "exit_threshold": [0.3, 0.5, 0.7, 1.0],
    "max_hold_bars": [180, 360, 720, 1440],
    "stop_loss_zscore": [2.5, 3.0, 3.5, 4.0]
  }' \
  --algorithm bayesian \
  --objective sharpe_ratio \
  --max-iterations 50
```

### Validation: Walk-Forward Analysis

```bash
# Walk-forward validation (prevents overfitting)
python -m src.services.optimizer.main walk-forward \
  --strategy Pairs_Trading \
  --config '{
    "pairs": [
      ["AAPL", "MSFT"],
      ["JPM", "BAC"]
    ]
  }' \
  --start-date "2024-06-01" \
  --end-date "2024-10-22" \
  --param-ranges '{
    "lookback_window": [120, 240, 360],
    "entry_threshold": [1.5, 2.0, 2.5],
    "exit_threshold": [0.3, 0.5, 0.7]
  }' \
  --algorithm bayesian \
  --objective sharpe_ratio \
  --train-days 30 \
  --test-days 7 \
  --step-days 7
```

---

## Expected Performance Characteristics

### Trade Frequency (5-Second Bars)
- **High Frequency**: Much more frequent signals than daily bars
- **Expected**: 5-20 round-trip trades per pair per day
- **Duration**: Most trades complete within 15-60 minutes

### Risk Profile
- **Intraday Only**: No overnight gap risk
- **Market Close**: Guaranteed flat by 4:00 PM
- **Position Duration**: Typically < 1 hour per trade

### Resource Requirements
- **Data Volume**: ~23,400 bars per symbol per day
- **Storage**: ~5 MB per symbol per day (compressed)
- **Memory**: ~50 MB per pair for ratio history
- **CPU**: Minimal (strategy is lightweight)

---

## Monitoring

### Check Strategy Status

```bash
# View strategy state
curl http://localhost:8005/strategies/1 | jq '.state'
```

### Expected Output

```json
{
  "config": {
    "pairs": [["AAPL", "MSFT"], ["JPM", "BAC"], ...],
    "lookback_window": 240,
    "entry_threshold": 2.0,
    "exit_threshold": 0.5,
    "max_hold_bars": 720
  },
  "num_pairs": 9,
  "pairs_state": {
    "AAPL/MSFT": {
      "position": "long_a_short_b",
      "current_ratio": 7.856,
      "bars_in_trade": 145,
      "entry_zscore": 2.15
    },
    "JPM/BAC": {
      "position": "flat",
      "bars_in_trade": 0,
      "entry_zscore": 0.0
    }
  }
}
```

---

## Troubleshooting

### Issue: No signals generated

**Possible causes:**
1. Not enough 5-second data (need > 240 bars = 20 minutes)
2. Z-scores haven't exceeded thresholds
3. Near market close (positions closed, no new entries)

**Solution:**
```bash
# Check data availability
docker exec -it trading-postgres psql -U bot -d trading
SELECT symbol, COUNT(*) FROM candles WHERE tf = '5 secs' GROUP BY symbol;

# Check strategy logs
docker logs trading-strategy --tail=100 | grep "pairs trading"
```

### Issue: Positions not closing before market close

**Check configuration:**
```bash
curl http://localhost:8005/strategies/1 | jq '.config'
```

Ensure:
- `market_close_hour`: 16 (4 PM)
- `close_before_eod_minutes`: 5
- Positions should force-close at 3:55 PM

### Issue: Too many trades (overtrading)

**Solution:** Increase entry threshold
```json
{
  "entry_threshold": 2.5,  // from 2.0
  "lookback_window": 360   // from 240 (longer lookback = more stable)
}
```

### Issue: Too few trades (undertrading)

**Solution:** Decrease entry threshold
```json
{
  "entry_threshold": 1.5,  // from 2.0
  "lookback_window": 240   // keep shorter for more sensitivity
}
```

---

## Next Steps

1. ✅ **Historical Data**: Ensure 30+ days of 5-second data
2. ✅ **Backtest**: Run single backtest to verify strategy works
3. ✅ **Optimize**: Run Bayesian optimization to find best parameters
4. ✅ **Validate**: Use walk-forward analysis to prevent overfitting
5. ✅ **Paper Trade**: Test with live 5-second data in paper account
6. ✅ **Go Live**: Enable live trading after validation

---

## Summary of Changes

| Aspect | Old (Daily) | New (5-Second) |
|--------|-------------|----------------|
| Lookback | 20 bars (20 days) | 240 bars (20 minutes) |
| Max Hold | 20 days | 720 bars (1 hour) |
| Position Tracking | days_in_trade | bars_in_trade |
| Overnight Holds | Yes | **No - Force close at 3:55 PM** |
| Default Pairs | 1 pair | 9 pairs |
| Exit Conditions | 3 | 4 (added end_of_day) |
| Trade Duration | Days | Minutes |

---

## File Modified

- `backend/src/strategy_lib/pairs_trade.py` (539 lines)

**Changes:**
- Added `bars_in_trade` instead of `days_in_trade`
- Added `max_hold_bars` instead of `max_hold_days`
- Added intraday trading parameters (market_close_hour, close_before_eod_minutes)
- Added end-of-day position closure logic
- Updated default configuration with 9 pairs
- Adjusted lookback_window for 5-second bars

---

**Status**: ✅ Ready for optimization and testing

