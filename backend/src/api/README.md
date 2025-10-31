# Trading Bot API Gateway

The unified API gateway aggregates all microservices into a single REST API for the frontend dashboard.

## Architecture

The API gateway acts as a reverse proxy, routing requests to the appropriate microservices:
- **Account Service** (8001): Account stats, positions
- **Market Data Service** (8002): Live market data subscriptions
- **Historical Data Service** (8003): Historical data requests
- **Trader Service** (8004): Order placement and management
- **Strategy Service** (8005): Strategy management
- **Optimizer Service** (8006): Parameter optimization
- **Backtester Service** (8007): Backtest execution

## Base URL

```
http://localhost:8000
```

## API Documentation

Interactive API documentation is available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Authentication

**None required** - This is a local-only system with no authentication middleware.

---

## Endpoints

### Health & Status

#### `GET /healthz`
Simple health check for the gateway itself.

**Response:**
```json
{
  "status": "healthy",
  "service": "api-gateway"
}
```

#### `GET /api/health`
Aggregate health status from all services.

**Response:**
```json
{
  "gateway": "healthy",
  "overall": "healthy",
  "services": {
    "account": {"status": "healthy", "details": {...}},
    "marketdata": {"status": "healthy", "details": {...}},
    "historical": {"status": "healthy", "details": {...}},
    "trader": {"status": "healthy", "details": {...}},
    "strategy": {"status": "healthy", "details": {...}},
    "optimizer": {"status": "healthy", "details": {...}},
    "backtester": {"status": "healthy", "details": {...}}
  }
}
```

---

### Account Management

#### `GET /api/account`
Get account summary statistics.

**Response:**
```json
{
  "account_id": "DU7084660",
  "net_liquidation": 1000000.00,
  "available_funds": 950000.00,
  "buying_power": 1900000.00,
  "positions": [...]
}
```

#### `GET /api/positions`
Get current positions (same as account stats).

---

### Order Management

#### `GET /api/orders`
Get order history with optional filters.

**Query Parameters:**
- `status` (optional): Filter by order status
- `symbol` (optional): Filter by symbol
- `limit` (default: 100): Maximum number of orders to return

**Example:**
```
GET /api/orders?status=Filled&symbol=AAPL&limit=50
```

**Response:**
```json
{
  "orders": [
    {
      "id": 1,
      "symbol": "AAPL",
      "side": "BUY",
      "qty": 100,
      "order_type": "LMT",
      "limit_price": 150.00,
      "status": "Filled",
      "placed_at": "2025-10-17T12:00:00Z"
    }
  ]
}
```

#### `GET /api/orders/{order_id}`
Get specific order details.

**Response:**
```json
{
  "id": 1,
  "symbol": "AAPL",
  "side": "BUY",
  "qty": 100,
  "status": "Filled",
  ...
}
```

#### `POST /api/orders`
Place a new order.

**Request Body:**
```json
{
  "symbol": "AAPL",
  "side": "BUY",
  "qty": 100,
  "order_type": "LMT",
  "limit_price": 150.00,
  "tif": "DAY"
}
```

**Response:**
```json
{
  "order_id": 5,
  "status": "Submitted",
  "message": "Order placed successfully"
}
```

#### `POST /api/orders/{order_id}/cancel`
Cancel an order.

**Response:**
```json
{
  "message": "Order cancelled",
  "order_id": 5
}
```

---

### Market Data

#### `GET /api/ticks`
Get recent tick data for a symbol.

**Query Parameters:**
- `symbol` (required): Stock symbol
- `limit` (default: 100): Maximum number of ticks

**Example:**
```
GET /api/ticks?symbol=AAPL&limit=50
```

**Response:**
```json
{
  "symbol": "AAPL",
  "count": 50,
  "ticks": [
    {
      "timestamp": "2025-10-17T12:00:00Z",
      "bid": 149.98,
      "ask": 150.02,
      "last": 150.00,
      "bid_size": 100,
      "ask_size": 200,
      "last_size": 50
    }
  ]
}
```

#### `GET /api/subscriptions`
Get current market data subscriptions.

**Response:**
```json
{
  "subscriptions": ["AAPL", "MSFT", "SPY"],
  "count": 3,
  "max_subscriptions": 50
}
```

#### `GET /api/watchlist`
Get all symbols in watchlist.

**Response:**
```json
{
  "symbols": ["AAPL", "MSFT", "SPY"],
  "count": 3
}
```

#### `POST /api/watchlist`
Add or remove symbols from watchlist.

**Request Body:**
```json
{
  "action": "add",
  "symbol": "TSLA"
}
```

**Response:**
```json
{
  "message": "Added TSLA to watchlist"
}
```

---

### Historical Data

#### `POST /api/historical/request`
Request historical data for a symbol.

**Request Body:**
```json
{
  "symbol": "AAPL",
  "bar_size": "5 secs",
  "duration": "1 D"
}
```

**Response:**
```json
{
  "request_id": "AAPL_5 secs_1 D_1708000000",
  "status": "queued",
  "chunks": 24,
  "job_id": 42,
  "message": "Queued 24 chunk(s) for AAPL",
  "queue_position": 3
}
```

#### `POST /api/historical/bulk`
Request historical data for all watchlist symbols.

**Response:**
```json
{
  "message": "Bulk historical data request queued",
  "symbols": ["AAPL", "MSFT", "SPY"]
}
```

#### `GET /api/historical/jobs`
List persisted historical jobs.

**Response:**
```json
{
  "jobs": [
    {
      "id": 42,
      "job_key": "AAPL_5 secs_1 D_1708000000",
      "symbol": "AAPL",
      "bar_size": "5 secs",
      "status": "running",
      "total_chunks": 24,
      "completed_chunks": 6,
      "failed_chunks": 0,
      "created_at": "2024-02-15T03:15:00Z",
      "updated_at": "2024-02-15T03:35:00Z",
      "started_at": "2024-02-15T03:15:10Z",
      "completed_at": null
    }
  ],
  "count": 1
}
```

#### `GET /api/historical/jobs/{job_id}`
Get job details including chunk breakdown.

**Response:**
```json
{
  "id": 42,
  "job_key": "AAPL_5 secs_1 D_1708000000",
  "symbol": "AAPL",
  "bar_size": "5 secs",
  "status": "running",
  "duration": "1 D",
  "total_chunks": 24,
  "completed_chunks": 6,
  "failed_chunks": 0,
  "chunks": [
    {
      "id": 210,
      "request_id": "AAPL_5 secs_1 D_1708000000_chunk1of24",
      "chunk_index": 1,
      "status": "completed",
      "duration": "7200 S",
      "start_datetime": "2024-02-14T03:15:00Z",
      "end_datetime": "2024-02-14T05:15:00Z",
      "attempts": 1,
      "bars_received": 1440,
      "completed_at": "2024-02-15T03:17:00Z"
    }
  ]
}
```

#### `GET /api/historical/queue`
Get historical data request queue status.

**Response:**
```json
{
  "queue_size": 3,
  "active_requests": [
    {
      "id": "AAPL_5 secs_1 D_1708000000_chunk7of24",
      "symbol": "AAPL",
      "bar_size": "5 secs",
      "status": "in_progress",
      "started_at": "2024-02-15T03:36:02Z"
    }
  ],
  "recent_completions": [
    {
      "id": "AAPL_5 secs_1 D_1708000000_chunk6of24",
      "symbol": "AAPL",
      "bar_size": "5 secs",
      "status": "completed",
      "bars_count": 1440,
      "completed_at": "2024-02-15T03:34:59Z",
      "error": null
    }
  ],
  "db_summary": {
    "pending_chunks": 48,
    "in_progress_chunks": 1,
    "failed_chunks": 0,
    "skipped_chunks": 0,
    "total_jobs": 4,
    "completed_jobs": 1,
    "failed_jobs": 0
  }
}
```

#### `GET /api/historical/request/{request_id}`
Get a persisted request (chunk) status by request identifier.

**Response:**
```json
{
  "request_id": "AAPL_5 secs_1 D_1708000000_chunk7of24",
  "status": "in_progress",
  "symbol": "AAPL",
  "bar_size": "5 secs",
  "started_at": "2024-02-15T03:36:02Z",
  "completed_at": null,
  "bars_count": 0,
  "error": null,
  "job_id": 42
}
```

---

### Strategy Management

#### `GET /api/strategies`
Get all strategies from database.

**Response:**
```json
{
  "strategies": [
    {
      "id": 1,
      "name": "SMA_Crossover",
      "enabled": true,
      "params": {"short_period": 10, "long_period": 50},
      "created_at": "2025-10-01T00:00:00Z"
    }
  ]
}
```

#### `POST /api/strategies/{strategy_id}/enable`
Enable or disable a strategy.

**Request Body:**
```json
{
  "enabled": true
}
```

**Response:**
```json
{
  "message": "Strategy enabled",
  "strategy_id": 1,
  "enabled": true
}
```

#### `PUT /api/strategies/{strategy_id}/params`
Update strategy parameters.

**Request Body:**
```json
{
  "short_period": 15,
  "long_period": 60
}
```

**Response:**
```json
{
  "message": "Strategy parameters updated",
  "strategy_id": 1,
  "params": {"short_period": 15, "long_period": 60}
}
```

---

### Backtesting

#### `POST /api/backtests`
Trigger a new backtest.

**Request Body:**
```json
{
  "strategy_name": "SMA_Crossover",
  "symbols": ["AAPL"],
  "params": {"short_period": 10, "long_period": 50},
  "start_date": "2024-01-01",
  "end_date": "2024-12-31"
}
```

**Response:**
```json
{
  "run_id": 25,
  "message": "Backtest started"
}
```

#### `GET /api/backtests`
List recent backtests.

**Query Parameters:**
- `limit` (default: 50): Maximum number of results

**Response:**
```json
{
  "backtests": [
    {
      "id": 25,
      "strategy_name": "SMA_Crossover",
      "sharpe_ratio": 0.85,
      "total_return": 15.5,
      "max_drawdown": -5.2,
      "created_at": "2025-10-17T10:00:00Z"
    }
  ]
}
```

#### `GET /api/backtests/{run_id}`
Get backtest results.

**Response:**
```json
{
  "id": 25,
  "strategy_name": "SMA_Crossover",
  "params": {"short_period": 10, "long_period": 50},
  "pnl": 15500.00,
  "sharpe_ratio": 0.85,
  "max_drawdown": -5.2,
  "total_trades": 45,
  "win_rate": 55.5
}
```

#### `GET /api/backtests/{run_id}/trades`
Get trades from a backtest.

**Response:**
```json
{
  "trades": [
    {
      "id": 1,
      "symbol": "AAPL",
      "side": "BUY",
      "entry_ts": "2024-01-15T09:30:00Z",
      "entry_px": 150.00,
      "exit_ts": "2024-01-20T16:00:00Z",
      "exit_px": 155.00,
      "pnl": 500.00
    }
  ]
}
```

---

### Parameter Optimization

#### `POST /api/optimizations`
Start a new parameter optimization.

**Request Body:**
```json
{
  "strategy_name": "SMA_Crossover",
  "symbols": ["AAPL"],
  "param_ranges": {
    "short_period": [5, 10, 15, 20],
    "long_period": [30, 40, 50, 60]
  },
  "algorithm": "genetic",
  "objective": "sharpe_ratio",
  "max_iterations": 100
}
```

**Response:**
```json
{
  "run_id": 30,
  "message": "Optimization started",
  "status": "running"
}
```

#### `GET /api/optimizations`
List recent optimizations.

**Query Parameters:**
- `limit` (default: 50): Maximum number of results

**Response:**
```json
{
  "optimizations": [
    {
      "id": 30,
      "strategy_name": "SMA_Crossover",
      "algorithm": "genetic",
      "status": "completed",
      "best_score": 1.006,
      "best_params": {"short_period": 5, "long_period": 45}
    }
  ]
}
```

#### `GET /api/optimizations/{run_id}`
Get optimization status and results.

**Response:**
```json
{
  "id": 30,
  "strategy_name": "SMA_Crossover",
  "algorithm": "genetic",
  "status": "completed",
  "total_combinations": 100,
  "completed_combinations": 99,
  "best_score": 1.006,
  "best_params": {"short_period": 5, "long_period": 45},
  "duration_seconds": 1.3
}
```

#### `GET /api/optimizations/{run_id}/results`
Get top parameter combinations.

**Query Parameters:**
- `top_n` (default: 20): Number of top results to return

**Response:**
```json
{
  "results": [
    {
      "params": {"short_period": 5, "long_period": 45},
      "score": 1.006,
      "sharpe_ratio": 1.006,
      "total_return": 40.04,
      "rank": 1
    }
  ]
}
```

#### `GET /api/optimizations/{run_id}/analysis`
Get parameter sensitivity analysis.

**Response:**
```json
{
  "sensitivity": [
    {
      "parameter": "short_period",
      "sensitivity_score": 0.85,
      "importance_rank": 1,
      "correlation": 0.92
    },
    {
      "parameter": "long_period",
      "sensitivity_score": 0.62,
      "importance_rank": 2,
      "correlation": 0.78
    }
  ]
}
```

#### `GET /api/optimizations/{run_id}/pareto`
Get Pareto frontier for multi-objective optimization.

**Query Parameters:**
- `objectives` (default: "sharpe_ratio,max_drawdown"): Comma-separated objectives

**Response:**
```json
{
  "pareto_front": [
    {
      "params": {"short_period": 5, "long_period": 45},
      "sharpe_ratio": 1.006,
      "max_drawdown": -5.2,
      "is_efficient": true
    }
  ]
}
```

#### `POST /api/optimizations/{run_id}/stop`
Stop a running optimization.

**Response:**
```json
{
  "message": "Optimization stopped",
  "run_id": 30,
  "partial_results": true
}
```

---

### Signals & Analytics

#### `GET /api/signals`
Get recent strategy signals.

**Query Parameters:**
- `strategy_id` (optional): Filter by strategy
- `symbol` (optional): Filter by symbol
- `limit` (default: 100): Maximum number of signals

**Example:**
```
GET /api/signals?strategy_id=1&symbol=AAPL&limit=50
```

**Response:**
```json
{
  "count": 50,
  "signals": [
    {
      "id": 123,
      "strategy_id": 1,
      "symbol": "AAPL",
      "signal_type": "BUY",
      "strength": 0.85,
      "timestamp": "2025-10-17T12:00:00Z",
      "meta": {"price": 150.00}
    }
  ]
}
```

#### `GET /api/executions`
Get recent trade executions.

**Query Parameters:**
- `limit` (default: 100): Maximum number of executions

**Response:**
```json
{
  "count": 100,
  "executions": [
    {
      "id": 45,
      "order_id": 12,
      "trade_id": "T12345",
      "symbol": "AAPL",
      "qty": 100,
      "price": 150.00,
      "timestamp": "2025-10-17T12:00:00Z"
    }
  ]
}
```

---

## WebSocket Endpoints

Real-time data streaming via WebSocket connections.

### `WS /ws/account`
Real-time account updates.

**Message Format:**
```json
{
  "account_id": "DU7084660",
  "net_liquidation": 1000000.00,
  "available_funds": 950000.00,
  "timestamp": "2025-10-17T12:00:00Z"
}
```

**Update Frequency**: Every 2 seconds

### `WS /ws/market`
Real-time market data for watchlist symbols.

**Message Format:**
```json
{
  "AAPL": {
    "bid": 149.98,
    "ask": 150.02,
    "last": 150.00,
    "timestamp": "2025-10-17T12:00:00Z"
  },
  "MSFT": {
    "bid": 299.50,
    "ask": 299.75,
    "last": 299.60,
    "timestamp": "2025-10-17T12:00:00Z"
  }
}
```

**Update Frequency**: Every 1 second (up to 10 symbols)

### `WS /ws/orders`
Real-time order status updates.

**Message Format:**
```json
{
  "orders": [
    {
      "id": 5,
      "symbol": "AAPL",
      "side": "BUY",
      "qty": 100,
      "order_type": "LMT",
      "status": "Filled",
      "updated_at": "2025-10-17T12:00:00Z"
    }
  ]
}
```

**Update Frequency**: Every 2 seconds (last 20 orders)

---

## CORS Configuration

The API gateway has CORS enabled for:
- `http://localhost:3000` (production frontend)
- `http://localhost:5173` (Vite dev server)
- `*` (wildcard for development)

All methods and headers are allowed.

---

## Error Handling

All endpoints return standard HTTP status codes:

- **200 OK**: Successful request
- **400 Bad Request**: Invalid request parameters
- **404 Not Found**: Resource not found
- **503 Service Unavailable**: Backend service unavailable
- **500 Internal Server Error**: Unexpected error

**Error Response Format:**
```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## Development

### Running the API Gateway

**Via Docker Compose:**
```bash
docker compose up backend-api
```

**Standalone:**
```bash
cd backend/src/api
python main.py
```

The gateway will start on `http://localhost:8000`.

### Testing Endpoints

```bash
# Health check
curl http://localhost:8000/healthz

# Aggregate health
curl http://localhost:8000/api/health

# Get orders
curl http://localhost:8000/api/orders

# Place order
curl -X POST http://localhost:8000/api/orders \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "side": "BUY", "qty": 100, "order_type": "MKT"}'
```

### WebSocket Testing

Use a WebSocket client or JavaScript:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/market');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Market data:', data);
};
```

---

## Service Dependencies

The API gateway depends on all backend services being available:
- Account Service (8001)
- Market Data Service (8002)
- Historical Data Service (8003)
- Trader Service (8004)
- Strategy Service (8005)
- Optimizer Service (8006)
- Backtester Service (8007)

If any service is unavailable, the gateway will return a `503 Service Unavailable` error for that endpoint.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────┐
│           Frontend (React/Vue)                  │
│              localhost:3000                     │
└──────────────────┬──────────────────────────────┘
                   │ HTTP/WebSocket
                   ▼
┌──────────────────────────────────────────────────┐
│          API Gateway (FastAPI)                   │
│              localhost:8000                      │
│  - CORS enabled                                  │
│  - Request routing                               │
│  - Health aggregation                            │
│  - WebSocket proxies                             │
└─────────────┬────────────────────────────────────┘
              │
    ┌─────────┴──────────┬──────────┬──────────┐
    ▼                    ▼          ▼          ▼
┌─────────┐         ┌─────────┐ ┌──────────┐ ┌────────┐
│Account  │         │Trader   │ │Strategy  │ │Optimizer
│  8001   │         │  8004   │ │  8005    │ │  8006  │
└─────────┘         └─────────┘ └──────────┘ └────────┘
    │                    │          │            │
    └────────────────────┴──────────┴────────────┘
                         │
                         ▼
                 ┌──────────────┐
                 │  PostgreSQL  │
                 │    5432      │
                 └──────────────┘
```

---

## Future Enhancements

- [ ] Authentication with JWT tokens
- [ ] Rate limiting per endpoint
- [ ] Request/response caching
- [ ] WebSocket connection pooling
- [ ] Metrics and logging aggregation
- [ ] API versioning (v1, v2)
