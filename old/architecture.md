# Trading Bot Architecture

## Overview

This is a microservices-based algorithmic trading system built with Docker Compose, designed to connect to Interactive Brokers TWS API for paper and live trading. The system is optimized for local deployment with PostgreSQL persistence and focuses on safety, observability, and multi-strategy support.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   TWS/IB API    │    │   PostgreSQL    │
│ (React/Vue/JS)  │    │ (172.25.0.100)  │    │   (Database)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌────────┼───────────────────────┘
         │              │        │
         │    ┌─────────────┐    │
         └────│ Backend API │    │
              │  Gateway    │    │
              └─────────────┘    │
                       │         │
┌──────────────────────┼─────────┼──────────────────────────────────┐
│              Docker Network    │                                  │
│                      │         │                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  │  Account    │  │ Market Data │  │ Historical  │  │   Trader    │
│  │  Service    │  │  Service    │  │  Service    │  │  Service    │
│  │ (Client 11) │  │ (Client 12) │  │ (Client 13) │  │ (Client 14) │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐                                │
│  │  Strategy   │  │ Backtester  │                                │
│  │  Service    │  │  Service    │                                │
│  │(Clients 15+)│  │ (On-demand) │                                │
│  └─────────────┘  └─────────────┘                                │
└──────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
trading/
├── backend/                    # Python microservices
│   ├── src/
│   │   ├── common/            # Shared libraries
│   │   ├── tws_bridge/        # Interactive Brokers integration
│   │   ├── services/          # Microservices
│   │   ├── strategy_lib/      # Strategy implementations
│   │   └── api/              # REST API gateway
│   ├── tests/
│   ├── migrations/           # Database migrations
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/                  # Web dashboard
│   ├── src/
│   ├── public/
│   ├── Dockerfile
│   └── package.json
└── compose.yaml              # Docker orchestration
```

## Core Services

### 0. Backend API Gateway (`api`)
- **Purpose**: Central API endpoint for frontend communication
- **Port**: 8000
- **Data Flow**: Frontend ↔ API Gateway ↔ Backend Services ↔ Database
- **Responsibilities**:
  - REST API endpoints for frontend
  - WebSocket connections for real-time updates
  - CORS configuration
  - Request routing to backend services
  - Response aggregation and formatting

### 1. Account Service (`account`)
- **Purpose**: Monitors account summary, positions, and P&L
- **TWS Client ID**: 11
- **Data Flow**: TWS → DB (account_summary, positions tables)
- **Responsibilities**:
  - Real-time account data streaming
  - Position tracking
  - Account health monitoring

### 2. Market Data Service (`marketdata`)
- **Purpose**: Live market data ingestion
- **TWS Client ID**: 12
- **Data Flow**: TWS → DB (ticks table)
- **Responsibilities**:
  - Subscribe to symbols from watchlist table
  - Respect MAX_SUBSCRIPTIONS limit
  - Handle reconnection and resubscription
  - React to watchlist changes via Postgres LISTEN/NOTIFY

### 3. Historical Data Service (`historical`)
- **Purpose**: Batch historical data collection
- **TWS Client ID**: 13
- **Data Flow**: TWS → DB (candles table)
- **Responsibilities**:
  - Configurable bar sizes, timeframes, lookback periods
  - Respect IB pacing limits (MAX_HIST_REQUESTS_PER_MIN)
  - Idempotent data storage (no duplicates)
  - Queue management for batch requests

### 4. Trader Service (`trader`)
- **Purpose**: Order management and execution
- **TWS Client ID**: 14
- **API**: REST endpoints for order placement/cancellation
- **Data Flow**: Strategy/Dashboard → Trader → TWS → DB (orders, executions)
- **Responsibilities**:
  - Risk limit enforcement
  - Order lifecycle management
  - Support for paper/live/dry-run modes
  - Multiple order types (MKT, LMT, STP, STP-LMT)

### 5. Strategy Service (`strategy`)
- **Purpose**: Live strategy execution
- **TWS Client IDs**: 15-29 (range for multiple strategies)
- **Execution Model**: Bar-driven event loop
- **Responsibilities**:
  - Load and run enabled strategies from database
  - Generate signals (stored in signals table)
  - Place orders via Trader service REST API
  - Hot-reload strategy parameters
  - Multi-strategy concurrent execution

### 6. Backtester Service (`backtester`)
- **Purpose**: Historical strategy testing
- **Mode**: On-demand (CLI/REST triggered)
- **Data Source**: Local candles table only
- **Responsibilities**:
  - Run strategies against historical data
  - Simulate commissions and slippage
  - Generate performance metrics (Sharpe, MaxDD)
  - Store results in backtest_runs/backtest_trades

### 7. Frontend Dashboard (`frontend`)
- **Purpose**: Web-based monitoring and control interface
- **Stack**: React/Vue/Vanilla JS + Modern Build Tools
- **Port**: 3000
- **Authentication**: None (local LAN deployment)
- **Responsibilities**:
  - Real-time system monitoring via WebSocket
  - Strategy management interface
  - Backtest execution and results visualization
  - Order and position monitoring
  - Market data visualization

## Data Architecture

### Database Schema (PostgreSQL)
```sql
-- Reference & Control
symbols(symbol, conid, primary_exchange, currency, active, updated_at)
watchlist(id, symbol, added_at)
strategies(strategy_id, name, enabled, params_json, created_at)
risk_limits(id, key, value_json, updated_at)
health(service, status, updated_at)
logs(id, service, level, msg, ts, meta_json)

-- Account & Portfolio
accounts(account_id, currency, created_at)
account_summary(id, account_id, tag, value, currency, ts)
positions(id, account_id, symbol, conid, qty, avg_price, ts)

-- Market Data
ticks(id, symbol, ts, bid, ask, last, bid_size, ask_size, last_size)
candles(id, symbol, tf, ts, open, high, low, close, volume)

-- Trading
orders(id, account_id, strategy_id, symbol, side, qty, order_type, limit_price, stop_price, tif, status, external_order_id, placed_at, updated_at)
executions(id, order_id, trade_id, symbol, qty, price, ts)

-- Signals & Research
signals(id, strategy_id, symbol, signal_type, strength, ts, meta_json)
backtest_runs(id, strategy_name, params_json, start_ts, end_ts, pnl, sharpe, maxdd, trades, created_at)
backtest_trades(id, run_id, symbol, side, qty, entry_ts, entry_px, exit_ts, exit_px, pnl)
```

### Communication Patterns

1. **Frontend-to-Backend**: HTTP REST API calls via API Gateway
2. **Frontend-to-Backend (Real-time)**: WebSocket connections via API Gateway
3. **API Gateway-to-Services**: Internal HTTP calls and database queries
4. **Service-to-TWS**: Direct ib-insync connections with dedicated client IDs
5. **Service-to-DB**: SQLAlchemy with connection pooling and retry logic
6. **Inter-service**: Postgres LISTEN/NOTIFY for event-driven updates
7. **Strategy-to-Trader**: HTTP REST API for order placement

## Client ID Management

```
Service          Client ID    Purpose
─────────────────────────────────────────
account_ws       11          Account monitoring
marketdata       12          Live quotes
historical       13          Historical data pulls
trader           14          Order management
strategy[0]      15          Strategy instance 1
strategy[1]      16          Strategy instance 2
...              ...         ...
strategy[14]     29          Strategy instance 15
```

- **Allocation**: Database-persisted with collision detection
- **Recovery**: Automatic reclaim on service restart
- **Scaling**: Support for up to 15 concurrent strategies

## Safety & Risk Management

### Trading Modes
1. **Dry Run** (`DRY_RUN=1`): Full simulation, no TWS orders
2. **Paper Trading** (`USE_PAPER=1`, `TWS_PORT=7497`): TWS paper account
3. **Live Trading**: Requires multiple safety switches

### Safety Switches for Live Trading
All conditions must be met:
- `ENABLE_LIVE=1`
- `USE_PAPER=0`
- `TWS_PORT=7496`
- `DRY_RUN=0`
- `risk_limits.block_live_trading_until` in the past

### Risk Limits (Examples)
- `max_notional_per_order`
- `max_notional_per_symbol`
- `max_daily_loss`
- `block_live_trading_until`

## Resilience & Observability

### Connection Management
- **TWS Reconnection**: Exponential backoff with jitter
- **Database Resilience**: Connection pooling with retry logic
- **Service Health**: 15-second heartbeats to health table
- **Graceful Shutdown**: SIGTERM handling with cleanup

### Monitoring
- **Health Checks**: Docker health checks + database health table
- **Logging**: Structured JSON logs to stdout + optional DB sampling
- **Dashboard Alerts**: Visual indicators for service staleness (>60s)

### Data Consistency
- **Idempotent Operations**: Historical data upserts prevent duplicates
- **Atomic Transactions**: Order placement with proper rollback
- **Event Ordering**: Timestamp-based processing where critical

## Deployment

### Docker Composition
```yaml
services:
  postgres:     # Data persistence
  account_ws:   # Account monitoring
  marketdata:   # Live quotes
  historical:   # Historical data
  trader:       # Order execution  
  strategy:     # Strategy runner
  dashboard:    # Web interface
  # backtester: # On-demand only
```

### Environment Configuration
- **Network**: Fixed TWS host (172.25.0.100)
- **Ports**: Configurable (7497 paper, 7496 live)
- **Scaling**: Service-specific resource limits
- **Storage**: Persistent PostgreSQL volume

## Strategy Development

### Strategy Interface
```python
class Strategy:
    def on_start(self, config, instruments): 
        """Initialize strategy with configuration"""
        
    def on_bar(self, symbol, tf, bar_df): 
        """Process new bar data (latest N bars)"""
        
    def on_stop(self): 
        """Cleanup on strategy stop"""
```

### Strategy Lifecycle
1. **Development**: Create strategy class in strategy_lib/
2. **Registration**: Add to database strategies table
3. **Configuration**: Set parameters via dashboard JSON editor
4. **Backtesting**: Test against historical data
5. **Paper Trading**: Enable with paper account
6. **Live Trading**: Enable with safety switches

## Asset Class Support

The system is designed to be asset-class agnostic:
- **Equities**: Primary focus with US markets
- **Options**: Framework supports (future enhancement)
- **Futures**: Framework supports (future enhancement)  
- **Forex**: Framework supports (future enhancement)
- **Crypto**: Not supported (different infrastructure needed)

## Performance Considerations

### Data Volume Management
- **Tick Data**: High volume, user manages retention
- **Candle Data**: Moderate volume, multiple timeframes
- **Index Strategy**: Optimized queries with proper indexing
- **Connection Pooling**: Efficient database resource usage

### Latency Optimization
- **Local Deployment**: Minimize network latency
- **In-Memory Caching**: Strategy data in memory where appropriate
- **Async Processing**: Non-blocking I/O for market data ingestion
- **Batch Operations**: Efficient database writes

This architecture prioritizes safety, observability, and maintainability while providing the flexibility to run multiple strategies concurrently against various asset classes.
