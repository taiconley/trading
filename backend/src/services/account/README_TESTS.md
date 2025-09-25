# Account Service Connection Tests

This directory contains connection and functionality tests for the Account Service before implementing full database operations.

## Test Files

### 1. `simple_test.py`
- **Purpose**: Basic TWS connection test
- **What it does**: Connects to TWS, prints account info, and disconnects
- **Run**: `python simple_test.py` or `make test.account.simple`
- **Expected output**: Account summary and positions printed to console

### 2. `health_test.py`
- **Purpose**: Health check functionality test
- **What it does**: Tests health monitoring and HTTP endpoints
- **Run**: 
  - Test mode: `python health_test.py`
  - Server mode: `python health_test.py --server --port 8001`
- **Expected output**: Health status information

### 3. `test_connection.py`
- **Purpose**: Full connection and data stream test
- **What it does**: Connects and prints all account data streams (summary, positions, PnL)
- **Run**: `python test_connection.py` or `make test.account.connection`
- **Expected output**: Continuous stream of account data updates

### 4. `run_tests.py`
- **Purpose**: Test runner that executes all tests in sequence
- **What it does**: Runs all tests and provides summary
- **Run**: `python run_tests.py` or `make test.account`
- **Expected output**: Complete test suite results

## Prerequisites

1. **TWS/IB Gateway Running**: Make sure TWS or IB Gateway is running and configured
2. **Environment Variables**: Ensure `.env` file is configured with correct TWS settings
3. **Python Dependencies**: All required packages installed (`ib-insync`, `fastapi`, etc.)

## Environment Configuration

Make sure your `.env` file contains:

```ini
TWS_HOST=127.0.0.1          # or your TWS host
TWS_PORT=7497               # 7497 for paper, 7496 for live
USE_PAPER=1                 # Use paper trading
ENABLE_LIVE=0               # Disable live trading for testing
DRY_RUN=1                   # Enable dry run mode
```

## Expected Test Results

### Simple Test Success
```
✓ Connected to TWS successfully!
✓ Managed accounts: ['DU123456']
✓ Using account: DU123456
Account Summary:
  NetLiquidation: 1000000.00 USD
  TotalCashValue: 1000000.00 USD
Positions:
  (no positions or list of current positions)
✓ Basic account test completed successfully!
```

### Health Test Success
```
Health Status: {'status': 'healthy', 'tws_connected': True, 'accounts': ['DU123456'], ...}
Health check #1:
  Status: healthy
  TWS Connected: True
  Accounts: ['DU123456']
```

### Connection Test Success
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
```

## Troubleshooting

### Connection Failures
- Verify TWS/IB Gateway is running
- Check TWS host/port settings
- Ensure API connections are enabled in TWS
- Verify client ID 11 is not in use

### No Data Received
- Check if account has positions/activity
- Verify TWS account permissions
- Ensure market data subscriptions are active
- Check TWS logs for errors

### Permission Errors
- Verify TWS API settings allow connections
- Check if live trading permissions are needed
- Ensure paper trading account is properly configured

## Next Steps

Once all tests pass successfully:

1. ✅ Connection to TWS works
2. ✅ Account data streams are received
3. ✅ Health monitoring functions
4. ✅ Ready to implement database operations

You can then proceed with implementing the full `main.py` with database integration.
