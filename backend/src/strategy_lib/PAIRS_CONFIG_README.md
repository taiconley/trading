# Pairs Trading Configuration - Single Source of Truth

## Overview

The pairs trading configuration has been refactored to use a **single source of truth** located at:
```
backend/src/strategy_lib/pairs_trading_config.py
```

## How to Change Pairs

### Step 1: Edit the Configuration File

Edit `backend/src/strategy_lib/pairs_trading_config.py` and modify the `PAIRS` list:

```python
PAIRS = [
    ["AAPL", "MSFT"],   # Tech Large Cap
    ["JPM", "BAC"],     # Banks
    # Add or remove pairs here
]
```

### Step 2: Update the Database

Run the update script to sync the configuration to the database:

```bash
cd /home/taiconley/Desktop/Projects/trading
docker compose exec backend-api python /app/update_pairs_trading_params.py
```

### Step 3: Reload Strategy Service

The strategy service will automatically reload from the database, or you can manually trigger a reload via the API:

```bash
curl -X POST http://localhost:8005/strategies/reload-all
```

## Files Using This Configuration

1. **`pairs_trade.py`** - Strategy implementation (uses as default_config)
2. **`update_pairs_trading_params.py`** - Database update script (reads from config)

Both files import from `pairs_trading_config.py`, ensuring consistency.

## Configuration Structure

The configuration includes:
- **`PAIRS`** - List of pairs to track (easy to modify)
- **`PAIRS_TRADING_CONFIG`** - Complete strategy parameters (includes pairs)

All parameters are defined in one place, eliminating duplication and confusion.

## Benefits

✅ **Single source of truth** - Edit pairs in one place  
✅ **No duplication** - Configuration defined once  
✅ **Easy to modify** - Just edit the config file and run the update script  
✅ **Consistent** - Strategy defaults and database always match  

