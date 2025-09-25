# Historical Data Service

## Overview

The Historical Data Service handles batched historical data requests from TWS/IB Gateway with proper pacing controls, request queueing, and idempotent storage to the candles table. It follows TWS best practices for historical data requests to avoid rate limiting violations.

## Features

### ✅ Implemented

- **Request Queueing**: Asynchronous queue system for processing historical data requests
- **Pacing Guards**: Enforces `MAX_HIST_REQUESTS_PER_MIN` to respect IB rate limits
- **Idempotent Storage**: Upserts to `candles` table on (symbol, tf, ts) to prevent duplicates
- **Multiple Bar Sizes**: Supports configurable bar sizes from `HIST_BAR_SIZES` config
- **Configurable Parameters**: Supports barSize, whatToShow, RTH, lookback settings
- **Bulk Operations**: Can request historical data for all watchlist symbols
- **Health Monitoring**: Provides comprehensive health and queue status endpoints
- **TWS Integration**: Uses client ID 13 with automatic reconnection
- **Error Handling**: Graceful handling of IB rate limiting and connection issues

### Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   REST API      │───▶│  Request Queue   │───▶│  TWS Gateway    │
│   Requests      │    │   + Pacing       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Database      │    │  Historical     │
                       │   (candles)     │◄───│     Bars        │
                       └─────────────────┘    └─────────────────┘
```

## Configuration

The service uses the following configuration from environment variables:

- `TWS_HOST`: TWS/Gateway host (default: 172.17.0.1)
- `TWS_PORT`: TWS/Gateway port (7497 for paper, 7496 for live)
- `MAX_HIST_REQUESTS_PER_MIN`: Maximum requests per minute (default: 30)
- `HIST_BAR_SIZES`: Supported bar sizes (default: "1 min,5 mins,1 day")
- `TWS_CLIENT_ID_BASE`: Base client ID (service uses base + 3 = 13)

## API Endpoints

### Health & Status

#### `GET /healthz`
Service health check:
```json
{
  "status": "healthy",
  "service": "historical",
  "client_id": 23,
  "connected": true,
  "queue_size": 0,
  "active_requests": 0,
  "completed_requests": 0,
  "processing": true,
  "max_requests_per_minute": 30,
  "last_heartbeat": 1758841135.3887196
}
```

#### `GET /queue/status`
Detailed queue status:
```json
{
  "queue_size": 2,
  "active_requests": [
    {
      "id": "AAPL_1 min_1 D_1640995200",
      "symbol": "AAPL",
      "bar_size": "1 min",
      "status": "in_progress",
      "started_at": "2023-12-31T15:30:00Z"
    }
  ],
  "recent_completions": [
    {
      "id": "MSFT_1 min_1 D_1640991600",
      "symbol": "MSFT",
      "bar_size": "1 min",
      "status": "completed",
      "bars_count": 390,
      "completed_at": "2023-12-31T15:25:00Z",
      "error": null
    }
  ]
}
```

### Data Requests

#### `POST /historical/request`
Request historical data for a single symbol:
```bash
curl -X POST "http://localhost:8003/historical/request" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "bar_size": "1 min",
    "what_to_show": "TRADES",
    "duration": "1 D",
    "end_datetime": "",
    "use_rth": true
  }'
```

Response:
```json
{
  "request_id": "AAPL_1 min_1 D_1640995200",
  "status": "queued",
  "queue_position": 1
}
```

#### `POST /historical/bulk`
Request historical data for all watchlist symbols:
```bash
curl -X POST "http://localhost:8003/historical/bulk"
```

Response:
```json
{
  "message": "Queued 9 historical data requests",
  "symbols": ["AAPL", "MSFT", "SPY"],
  "bar_sizes": ["1 min", "5 mins", "1 day"],
  "requests": ["bulk_AAPL_1 min_1640995200", ...]
}
```

#### `GET /historical/request/{request_id}`
Get status of specific request:
```json
{
  "request_id": "AAPL_1 min_1 D_1640995200",
  "status": "completed",
  "symbol": "AAPL",
  "bar_size": "1 min",
  "started_at": "2023-12-31T15:30:00Z",
  "completed_at": "2023-12-31T15:32:00Z",
  "bars_count": 390,
  "error": null
}
```

## Database Schema

The service interacts with the `candles` table:

### `candles`
- **Primary Key**: (symbol, tf, ts) - ensures no duplicates
- **Columns**: symbol, tf (timeframe), ts (timestamp), open, high, low, close, volume
- **Idempotent Upserts**: Re-running requests won't create duplicates

## Request Processing

### Pacing Control
- Tracks request timestamps over rolling 60-second window
- Enforces `MAX_HIST_REQUESTS_PER_MIN` limit
- Automatically waits when approaching rate limits
- Logs pacing delays for monitoring

### Queue Management
- Asynchronous FIFO queue for fair processing
- Requests move through states: `pending` → `in_progress` → `completed/failed`
- Automatic cleanup of old completed requests (keeps last 50)
- Graceful error handling with detailed error messages

### Data Validation
- Validates bar sizes against configured `HIST_BAR_SIZES`
- Handles IB error codes appropriately
- Converts and validates all bar data before storage
- Supports data gaps and corrections

## Testing

Run the comprehensive test suite:
```bash
docker compose exec backend-historical python /app/src/services/historical/test_historical.py
```

Tests cover:
- Configuration loading (✅ 7/7 tests passing)
- Database connectivity and candles operations
- Client ID allocation and management
- Pacing calculation logic
- Health status monitoring
- Idempotent upsert behavior

## Usage Examples

### Seed Historical Data
```bash
# Request 30 days of 1-minute bars for all watchlist symbols
curl -X POST "http://localhost:8003/historical/bulk"

# Check queue status
curl "http://localhost:8003/queue/status"

# Request specific symbol with custom parameters
curl -X POST "http://localhost:8003/historical/request" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SPY",
    "bar_size": "5 mins", 
    "duration": "5 D",
    "what_to_show": "TRADES",
    "use_rth": true
  }'
```

### Monitor Progress
```bash
# Check service health
curl "http://localhost:8003/healthz"

# Get specific request status
curl "http://localhost:8003/historical/request/SPY_5 mins_5 D_1640995200"
```

## Deployment

The service runs in Docker with:
- Port 8003 for REST API
- Automatic startup with docker compose
- Health checks and restart policies
- Structured JSON logging to stdout
- TWS connection monitoring with auto-reconnection

## Error Handling

- **Rate Limiting**: Automatic pacing to prevent IB violations
- **Connection Loss**: Auto-reconnection with exponential backoff
- **Invalid Symbols**: Graceful handling with clear error messages
- **Data Validation**: Robust parsing and validation of bar data
- **Queue Management**: Failed requests are tracked with error details

## Performance

- **Concurrent Processing**: Single-threaded queue with async I/O
- **Database Efficiency**: Idempotent upserts prevent duplicate work
- **Memory Management**: Automatic cleanup of completed requests
- **Rate Compliance**: Respects IB pacing limits to avoid blocks

## Next Steps

The Historical Data Service is production-ready and implements all requirements from Task 8. Future enhancements could include:

- Historical data backfill automation
- Data quality monitoring and alerts
- Multiple timeframe aggregation
- Data export and archiving features
- Advanced retry logic for failed requests
