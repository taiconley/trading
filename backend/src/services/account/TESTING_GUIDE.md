# Account Service Connection Testing Guide

This guide helps you test the Account Service connection and data streams **before implementing database operations**.

## Overview

We've created simple test files to verify:
1. ✅ TWS connection works with client ID 11
2. ✅ Account summary data streams are received
3. ✅ Position data streams are received  
4. ✅ PnL data streams are received
5. ✅ Health monitoring functions correctly

## Prerequisites

### 1. TWS/IB Gateway Setup
- **TWS or IB Gateway must be running**
- **API connections must be enabled** in TWS settings
- **Paper trading account recommended** for testing

### 2. Environment Configuration
Ensure your `.env` file contains:
```ini
TWS_HOST=127.0.0.1          # or your TWS host IP
TWS_PORT=7497               # 7497 for paper, 7496 for live  
USE_PAPER=1                 # Use paper trading
ENABLE_LIVE=0               # Disable live trading for testing
DRY_RUN=1                   # Enable dry run mode
```

### 3. Docker Compose
Start at least the database:
```bash
docker compose up -d postgres
```

## Running Tests

### Option 1: Quick Test Script
Run the automated test script from the project root:
```bash
./test_account_connection.sh
```

### Option 2: Individual Tests
Navigate to the account service directory:
```bash
cd backend/src/services/account
```

Run individual tests:
```bash
# 1. Simple connection test (fastest)
python3 simple_test.py

# 2. Health check test
python3 health_test.py  

# 3. Full connection stream test (shows live data)
python3 test_connection.py

# 4. Run all tests together
python3 run_tests.py
```

### Option 3: Docker-Compatible Test
```bash
cd backend/src/services/account
python3 docker_test.py
```

## Expected Results

### ✅ Successful Simple Test
```
Connecting to TWS at 127.0.0.1:7497
Paper trading: True
✓ Connected to TWS successfully!
✓ Managed accounts: ['DU123456']
✓ Using account: DU123456
Account Summary:
  NetLiquidation: 1000000.00 USD
  TotalCashValue: 1000000.00 USD
Positions:
  AAPL: 100 @ 150.00
✓ Basic account test completed successfully!
```

### ✅ Successful Health Test
```
Health Status: {'status': 'healthy', 'tws_connected': True, 'accounts': ['DU123456']}
Health check #1:
  Status: healthy
  TWS Connected: True
  Accounts: ['DU123456']
```

### ✅ Successful Connection Stream Test
```
[ACCOUNT SUMMARY] 2025-09-25T10:30:00
  Account: DU123456
  Tag: NetLiquidation
  Value: 1000000.00
  Currency: USD
--------------------------------------------------
[POSITION] 2025-09-25T10:30:01
  Account: DU123456
  Symbol: AAPL
  ConID: 265598
  Position: 100
  Avg Cost: 150.00
--------------------------------------------------
[PNL] 2025-09-25T10:30:02
  Account: DU123456
  Daily PnL: 250.00
  Unrealized PnL: 500.00
  Realized PnL: 0.00
--------------------------------------------------
```

## Troubleshooting

### ❌ Connection Failures

**Error: "Failed to connect to TWS"**
- ✅ Check TWS/IB Gateway is running
- ✅ Verify TWS_HOST and TWS_PORT in environment
- ✅ Enable API connections in TWS: File → Global Configuration → API → Settings
- ✅ Check if client ID 11 is already in use

**Error: "Socket connection failed"**
- ✅ Verify TWS is accepting connections on the correct port
- ✅ Check firewall settings
- ✅ Try connecting from TWS machine first

### ❌ No Data Received

**Connected but no account data:**
- ✅ Verify account has positions or activity
- ✅ Check TWS account permissions
- ✅ Ensure market data subscriptions are active
- ✅ Look at TWS API logs for errors

### ❌ Permission Errors

**Error: "No managed accounts"**
- ✅ Check TWS login credentials
- ✅ Verify account permissions for API access
- ✅ Ensure paper trading account is properly configured

## Health Check Server

You can also run a health check HTTP server:
```bash
python3 health_test.py --server --port 8001
```

Then test endpoints:
```bash
curl http://localhost:8001/healthz
curl http://localhost:8001/status
```

## Test File Descriptions

| File | Purpose | What It Does |
|------|---------|--------------|
| `simple_test.py` | Basic connection | Connects, gets account info, disconnects |
| `health_test.py` | Health monitoring | Tests health check functionality |
| `test_connection.py` | Full data streams | Shows live account/position/PnL updates |
| `run_tests.py` | Test suite runner | Runs all tests with summary |
| `docker_test.py` | Docker-compatible | Simplified tests for container environment |

## Success Criteria

✅ **All tests should pass before implementing database operations**

When tests pass, you'll have confirmed:
1. ✅ TWS connection is stable
2. ✅ Account data streams work correctly
3. ✅ Client ID 11 is properly allocated
4. ✅ Health monitoring functions
5. ✅ Error handling works

## Next Steps

Once all connection tests pass:

1. **Implement Database Operations**: Create the full `main.py` with database integration
2. **Add Data Processing**: Implement account summary, positions, and PnL database storage
3. **Add WebSocket Support**: Implement real-time updates for the dashboard
4. **Integration Testing**: Test with the full system

## Notes

- These tests use **client ID 11** as designated for the account service
- Tests are designed to be **non-destructive** (read-only operations)
- All tests include **proper cleanup** and connection management
- Tests work with both **paper and live** accounts (paper recommended)
