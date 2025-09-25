# Project Structure

```
trading/
├─ backend/
│  ├─ src/
│  │  ├─ common/
│  │  │  ├─ config.py
│  │  │  ├─ db.py
│  │  │  ├─ models.py
│  │  │  ├─ schemas.py
│  │  │  ├─ logging.py
│  │  │  └─ notify.py
│  │  ├─ tws_bridge/
│  │  │  ├─ ib_client.py
│  │  │  └─ client_ids.py
│  │  ├─ services/
│  │  │  ├─ account/        # account summary/positions into DB
│  │  │  ├─ marketdata/     # live quotes -> ticks table
│  │  │  ├─ historical/     # batched historical pulls -> candles
│  │  │  ├─ trader/         # order routing + risk + executions
│  │  │  ├─ strategy/       # live bar-driven runner
│  │  │  └─ backtester/     # offline runs over candles
│  │  ├─ strategy_lib/
│  │  │  ├─ base.py
│  │  │  ├─ registry.py
│  │  │  └─ examples/
│  │  │     ├─ sma_cross.py
│  │  │     └─ mean_revert.py
│  │  └─ api/              # REST API gateway (optional)
│  ├─ tests/
│  │  ├─ unit/
│  │  └─ integration/
│  ├─ migrations/
│  ├─ Dockerfile
│  ├─ pyproject.toml
│  └─ alembic.ini
├─ frontend/
│  ├─ src/
│  │  ├─ components/       # UI components
│  │  ├─ pages/           # Dashboard pages  
│  │  ├─ services/        # API client
│  │  └─ utils/
│  ├─ public/
│  ├─ Dockerfile
│  └─ package.json
├─ compose.yaml
├─ .env.example
├─ Makefile
├─ README.md
├─ architecture.md
└─ tasks.md
```
## 1) Docker & Compose Setup
### Tasks:
- [x] Create `backend/Dockerfile`
  - [x] Use Python 3.10+ base image
  - [x] Install pip for dependency management (using requirements.txt instead of poetry)
  - [x] Set system timezone data
  - [x] Create non-root user for security
  - [x] Copy backend source code and dependencies
  - [x] Set appropriate entrypoint for services
- [x] Create `frontend/Dockerfile`
  - [x] Use Node.js LTS base image
  - [x] Install npm dependencies
  - [x] Build frontend assets
  - [x] Serve with nginx
  - [x] Expose port 3000
- [x] Create `compose.yaml` at project root
  - [x] Define postgres service with persistent volume
  - [x] Define backend services (account, marketdata, historical, trader, strategy)
  - [x] Define frontend service
  - [x] Create shared network for all services
  - [x] Configure environment variables from .env
  - [x] Set up volume for postgres data persistence
  - [x] Add health checks for all services
- [x] Test: `docker compose config` validates successfully
- [x] Test: Services can communicate through shared network
- [x] Test: Unhealthy services retry with exponential backoff

## 2) Environment & Config
### Tasks:
- [x] Create `.env.example` with all required environment variables:
  - [x] Database settings (POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_HOST, POSTGRES_PORT)
  - [x] TWS connection settings (TWS_HOST=172.25.0.100, TWS_PORT, USE_PAPER, ENABLE_LIVE, DRY_RUN)
  - [x] Market data settings (DEFAULT_SYMBOLS, MAX_SUBSCRIPTIONS, BAR_SIZE, WHAT_TO_SHOW, RTH, LOOKBACK)
  - [x] Historical data settings (MAX_HIST_REQUESTS_PER_MIN, HIST_BAR_SIZES)
  - [x] Client ID management (TWS_CLIENT_ID_BASE, RECONNECT_BACKOFF_MIN, RECONNECT_BACKOFF_MAX)
  - [x] Backtest defaults (BT_COMM_PER_SHARE, BT_MIN_COMM_PER_ORDER, BT_DEFAULT_SLIPPAGE_TICKS, BT_TICK_SIZE_US_EQUITY)
- [x] Create `backend/src/common/config.py`
  - [x] Define typed Pydantic settings class
  - [x] Load from environment variables (Docker-native, no python-dotenv)
  - [x] Add validation for required fields
  - [x] Include feature flags and safety switches
- [x] Test: All services can load configuration without errors
- [x] Test: Invalid configuration values raise clear validation errors

### Example .env.example content:
```ini
POSTGRES_USER=bot
POSTGRES_PASSWORD=botpw
POSTGRES_DB=trading
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

TWS_HOST=172.25.0.100
TWS_PORT=7497            # 7497 paper, 7496 live
USE_PAPER=1
ENABLE_LIVE=0
DRY_RUN=1

DEFAULT_SYMBOLS=AAPL,MSFT,SPY
MAX_SUBSCRIPTIONS=50     # hard cap for live quotes
BAR_SIZE=1 min
WHAT_TO_SHOW=TRADES
RTH=1
LOOKBACK=30 D

MAX_HIST_REQUESTS_PER_MIN=30
HIST_BAR_SIZES=1 min,5 mins,1 day  # allow multiple

TWS_CLIENT_ID_BASE=10
RECONNECT_BACKOFF_MIN=1
RECONNECT_BACKOFF_MAX=30

# Backtest defaults
BT_COMM_PER_SHARE=0.005
BT_MIN_COMM_PER_ORDER=1.0
BT_DEFAULT_SLIPPAGE_TICKS=1
BT_TICK_SIZE_US_EQUITY=0.01
```

## 3) Database & Migrations
### Tasks:
- [x] Set up Alembic configuration
  - [x] Create `backend/alembic.ini` configuration file
  - [x] Create `backend/migrations/` directory structure
  - [x] Configure database connection string
- [x] Create `backend/src/common/models.py` with SQLAlchemy models:
  - [x] **Reference & Control tables:**
    - [x] `symbols(symbol, conid, primary_exchange, currency, active, updated_at)`
    - [x] `watchlist(id, symbol, added_at)` — live subs choose from here
    - [x] `strategies(strategy_id, name, enabled, params_json, created_at)`
    - [x] `risk_limits(id, key, value_json, updated_at)`
    - [x] `health(service, status, updated_at)`
    - [x] `logs(id, service, level, msg, ts, meta_json)`
  - [x] **Account & Portfolio tables:**
    - [x] `accounts(account_id, currency, created_at)`
    - [x] `account_summary(id, account_id, tag, value, currency, ts)`
    - [x] `positions(id, account_id, symbol, conid, qty, avg_price, ts)`
  - [x] **Market Data tables:**
    - [x] `ticks(id, symbol, ts, bid, ask, last, bid_size, ask_size, last_size)`
    - [x] `candles(id, symbol, tf, ts, open, high, low, close, volume)`
  - [x] **Trading tables:**
    - [x] `orders(id, account_id, strategy_id, symbol, side, qty, order_type, limit_price, stop_price, tif, status, external_order_id, placed_at, updated_at)`
    - [x] `executions(id, order_id, trade_id, symbol, qty, price, ts)`
  - [x] **Signals & Research tables:**
    - [x] `signals(id, strategy_id, symbol, signal_type, strength, ts, meta_json)` — all strategy signals for analysis
    - [x] `backtest_runs(id, strategy_name, params_json, start_ts, end_ts, pnl, sharpe, maxdd, trades, created_at)`
    - [x] `backtest_trades(id, run_id, symbol, side, qty, entry_ts, entry_px, exit_ts, exit_px, pnl)`
- [x] Create database indexes for performance:
  - [x] `ticks(symbol, ts DESC)`
  - [x] `candles(symbol, tf, ts DESC)`
  - [x] `orders(updated_at DESC)`
  - [x] `executions(order_id, ts)`
  - [x] `positions(symbol)`
  - [x] `signals(strategy_id, ts DESC)`
- [x] Create initial Alembic migration
  - [x] Generate migration: `alembic revision --autogenerate -m "Initial schema"`
  - [x] Review and edit migration file
  - [x] Test migration: `alembic upgrade head`
- [x] Create `backend/src/common/db.py` for database utilities:
  - [x] SQLAlchemy engine creation with connection pooling
  - [x] Session factory with proper cleanup
  - [x] Retry logic for transient database errors
  - [x] Health check function
- [x] Test: Migrations run cleanly without errors
- [x] Test: Round-trip data insertion and retrieval for all tables
- [x] Test: Database connection retry logic works

## 4) Common Libraries
### Tasks:
- [x] Create `backend/src/common/config.py` (completed in task 2)
  - [x] Typed Pydantic settings class for environment variables
  - [x] Feature flags and validation logic
  - [x] Safety switch validation functions
- [x] Create `backend/src/common/db.py` (completed in task 3)
  - [x] SQLAlchemy engine creation with connection pooling
  - [x] Session factory with proper cleanup and context management
  - [x] Retry logic for transient database errors
  - [x] Database health check utilities
- [x] Create `backend/src/common/logging.py`
  - [x] Structured JSON logging configuration
  - [x] Log to stdout for Docker container logs
  - [x] Optional database log sampling functionality
  - [x] Service-specific logger creation
  - [x] Log level configuration from environment
- [x] Create `backend/src/common/notify.py`
  - [x] Postgres LISTEN/NOTIFY wrapper functions
  - [x] Event types: `signals.new`, `orders.new`, `watchlist.update`
  - [x] Async notification handlers
  - [x] Connection management for notification channels
- [x] Create `backend/src/common/schemas.py`
  - [x] Pydantic models for API requests/responses
  - [x] Data validation schemas for orders, signals, etc.
  - [x] Type definitions for common data structures
- [x] Test: All services can import and use common libraries
- [x] Test: No code duplication across services

## 5) TWS Bridge & Client IDs
### Tasks:
- [x] Create `backend/src/tws_bridge/ib_client.py`
  - [x] Wrap ib-insync.IB connection with retry logic
  - [x] Implement exponential backoff with jitter for reconnections
  - [x] Auto-resubscribe to market data after reconnect
  - [x] Request throttling to avoid IB pacing violations
  - [x] Connection state management and health monitoring
  - [x] Graceful disconnect handling
- [x] Create `backend/src/tws_bridge/client_ids.py`
  - [x] Define client ID ranges for each service:
    - [x] account_ws: 11
    - [x] marketdata: 12  
    - [x] historical: 13
    - [x] trader: 14
    - [x] strategy runner instances: 15–29
  - [x] Database persistence for allocated client IDs
  - [x] Collision detection and next ID probing
  - [x] Automatic client ID reclaim on service restart
  - [x] Client ID allocation and deallocation functions
- [x] Create base TWS service class
  - [x] Common connection management
  - [x] Health check endpoints (`/healthz`)
  - [x] Graceful shutdown handling (SIGTERM)
  - [x] Error handling and logging
- [x] Test: Service restart automatically reclaims valid client ID
- [x] Test: Multiple services can connect simultaneously without conflicts
- [x] Test: Connection resilience during TWS restarts

## 6) Account Service (account_ws) ✅ COMPLETE
### Tasks:
- [x] Create `backend/src/services/account/main.py`
  - [x] Use TWS client ID 11
  - [x] Connect to TWS and request account summary stream
  - [x] Request positions stream
  - [x] Request PnL stream
  - [x] Handle account data updates with event-driven streaming
- [x] FastAPI REST API:
  - [x] `GET /healthz` endpoint with TWS connection status
  - [x] `GET /account/stats` endpoint for account statistics
  - [x] Production-ready error handling and logging
- [x] WebSocket endpoint for dashboard:
  - [x] FastAPI WebSocket endpoint for real-time updates (`/ws`)
  - [x] Event-driven broadcasting of account changes to connected clients
  - [x] Real-time account values, positions, and P&L streaming
- [x] Health monitoring:
  - [x] Implement `/healthz` endpoint
  - [x] Connection status monitoring
  - [x] Service statistics tracking
- [x] Docker integration:
  - [x] Static IP configuration (172.25.0.100)
  - [x] Proper container networking and health checks
  - [x] Service starts independently of TWS connection status
- [x] Test: TWS connection working with account DU7084660
- [x] Test: FastAPI server responding on port 8001
- [x] Test: Real-time streaming confirmed with live trading data
- [x] Test: Event handlers processing account updates correctly

### Status: ✅ PRODUCTION READY
- **Service**: Running and healthy
- **TWS Connection**: Connected to account DU7084660
- **API Endpoints**: All responding correctly
- **Real-time Streaming**: Event-driven architecture working
- **WebSocket Broadcasting**: Ready for dashboard integration

## 7) Market Data Service (marketdata) ✅ COMPLETE
### Tasks:
- [x] Create `backend/src/services/marketdata/main.py`
  - [x] Use TWS client ID 12 (allocated dynamically, actual ID: 22)
  - [x] Subscribe only to symbols in `watchlist` table
  - [x] Enforce MAX_SUBSCRIPTIONS limit from config
  - [x] Map IB tick types to bid/ask/last with sizes
- [x] Data processing:
  - [x] Write tick data to `ticks` table
  - [x] Handle tick data validation and filtering
  - [x] Asynchronous database writes for performance
- [x] Subscription management:
  - [x] Handle resubscribe on TWS reconnect
  - [x] Listen for `watchlist_update` notifications (fixed PostgreSQL channel naming)
  - [x] Add/remove subscriptions at runtime
  - [x] Gracefully handle subscription limits
- [x] Health monitoring:
  - [x] Implement `/healthz` endpoint
  - [x] Monitor subscription count and status
  - [x] Track data flow rates and connection status
- [x] FastAPI REST API:
  - [x] `GET /healthz` endpoint with TWS connection status
  - [x] `GET /subscriptions` endpoint for current market data subscriptions
  - [x] Production-ready error handling and logging
- [x] WebSocket endpoint for dashboard:
  - [x] FastAPI WebSocket endpoint for real-time updates (`/ws`)
  - [x] Event-driven broadcasting of market data changes to connected clients
  - [x] Real-time tick streaming with structured JSON messages
- [x] Event-driven architecture:
  - [x] Following TWS best practices from notes.md
  - [x] Ticker `updateEvent` handlers for real-time streaming
  - [x] Asynchronous tick data processing
  - [x] Non-blocking database operations
- [x] Docker integration:
  - [x] Fixed import issues for container deployment
  - [x] Proper Python path setup for module imports
  - [x] Service runs on port 8002
  - [x] Health checks and restart policies
- [x] Test: Comprehensive test suite with 6/6 tests passing
- [x] Test: MAX_SUBSCRIPTIONS limit is respected
- [x] Test: Database connectivity and operations working
- [x] Test: Client ID allocation and management working
- [x] Test: Health status monitoring functional

### Status: ✅ PRODUCTION READY
- **Service**: Running and healthy in Docker
- **TWS Connection**: Connected with proper event handling
- **API Endpoints**: All responding correctly (`/healthz`, `/subscriptions`)
- **WebSocket Streaming**: Real-time market data broadcasting ready
- **Database Integration**: Tick data storage and health monitoring working
- **Event Architecture**: Following notes.md best practices for reliable streaming
- **Testing**: Comprehensive test suite passing (6/6 tests)
- **Documentation**: Complete README.md with API documentation

## 8) Historical Data Service (historical) ✅ COMPLETE
### Tasks:
- [x] Create `backend/src/services/historical/main.py`
  - [x] Use TWS client ID 13 (allocated dynamically, actual ID: 23)
  - [x] Implement batched historical data pulls with async queue processing
  - [x] Support configurable barSize, whatToShow, RTH, lookback from settings
- [x] Request management:
  - [x] Implement asynchronous request queueing system with FIFO processing
  - [x] Enforce pacing guards (MAX_HIST_REQUESTS_PER_MIN = 30) with rolling window
  - [x] Handle IB rate limiting gracefully with automatic wait periods
  - [x] Support multiple bar sizes from HIST_BAR_SIZES config ("1 min,5 mins,1 day")
- [x] Data storage:
  - [x] Idempotent upsert to `candles` table on (symbol, tf, ts) - prevents duplicates
  - [x] Validate and clean historical data with proper type conversion
  - [x] Handle data gaps and corrections with update-or-insert logic
- [x] API endpoints:
  - [x] `POST /historical/request` - REST endpoint to trigger single symbol pulls
  - [x] `GET /queue/status` - Status endpoint for queue monitoring with active/completed requests
  - [x] `POST /historical/bulk` - Bulk data loading for all watchlist symbols
  - [x] `GET /historical/request/{id}` - Individual request status tracking
- [x] Health monitoring:
  - [x] Implement `/healthz` endpoint with comprehensive service status
  - [x] Track request queue status (queue size, active requests, completed count)
  - [x] Monitor pacing compliance with request rate tracking
- [x] FastAPI Integration:
  - [x] Production-ready REST API on port 8003
  - [x] Comprehensive error handling and validation
  - [x] Request/response models with proper HTTP status codes
  - [x] Async request processing with background task management
- [x] Database Integration:
  - [x] Idempotent storage to prevent duplicate historical data
  - [x] Health status updates to database every 30 seconds
  - [x] Proper transaction handling with rollback on errors
- [x] TWS Integration:
  - [x] Following notes.md best practices for historical data requests
  - [x] Automatic reconnection with exponential backoff
  - [x] Proper error handling for IB-specific error codes
  - [x] Request throttling to avoid pacing violations
- [x] Docker Integration:
  - [x] Fixed Python 3.11 compatibility issues (Queue import)
  - [x] Fixed ib-insync API compatibility (mktDataType parameter)
  - [x] Service runs healthy in container environment
  - [x] Proper logging and health check endpoints
- [x] Test: Comprehensive test suite with 7/7 tests passing
- [x] Test: Idempotent upsert behavior verified (rerun produces no duplicates)
- [x] Test: Pacing calculation logic validated
- [x] Test: Multiple timeframes supported through configuration

### Status: ✅ PRODUCTION READY
- **Service**: Running and healthy in Docker (client ID 23)
- **TWS Connection**: Connected with proper historical data request handling
- **API Endpoints**: All responding correctly (`/healthz`, `/queue/status`, `/historical/*`)
- **Request Processing**: Async queue with pacing controls active (0 queue, 0 active, 0 completed)
- **Database Integration**: Idempotent candles storage working with proper upsert logic
- **Rate Limiting**: Pacing guards enforcing 30 requests/minute limit
- **Testing**: Comprehensive test suite passing (7/7 tests)
- **Documentation**: Complete README.md with API documentation and usage examples

## 9) Trader Service (trader)
### Tasks:
- [ ] Create `backend/src/services/trader/main.py`
  - [ ] Use TWS client ID 14
  - [ ] FastAPI application with REST endpoints
- [ ] REST API endpoints:
  - [ ] `POST /orders` - Place orders (MKT/LMT/STP/STP-LMT)
  - [ ] `POST /cancel/{id}` - Cancel specific order
  - [ ] `GET /orders/{id}` - Get order status
  - [ ] `GET /orders` - List orders with filtering
- [ ] Order management:
  - [ ] Support bracket orders (future consideration)
  - [ ] Validate order parameters
  - [ ] Write to `orders` table on placement
  - [ ] Update order status on TWS callbacks
  - [ ] Insert executions to `executions` table
- [ ] Risk management:
  - [ ] Enforce `risk_limits` before sending orders
  - [ ] Validate against position limits
  - [ ] Check daily loss limits
  - [ ] Audit all risk decisions in logs
- [ ] Trading modes:
  - [ ] DRY_RUN: full lifecycle in DB without TWS
  - [ ] Paper trading support (USE_PAPER=1)
  - [ ] Live trading with safety switches
  - [ ] Hard block live unless ENABLE_LIVE=1 && DRY_RUN=0
- [ ] Health monitoring:
  - [ ] Implement `/healthz` endpoint
  - [ ] Monitor order flow and execution rates
- [ ] Test: Paper LMT order flows to filled state
- [ ] Test: Orders visible in both database and TWS
- [ ] Test: Risk limits properly block invalid orders

## 10) Strategy Interface & Live Runner (strategy)
### Tasks:
- [ ] Create strategy interface in `backend/src/strategy_lib/base.py`:
  - [ ] Define base `Strategy` class with required methods:
    - [ ] `on_start(self, config, instruments)` - Initialize strategy
    - [ ] `on_bar(self, symbol, tf, bar_df)` - Process latest N bars
    - [ ] `on_stop(self)` - Cleanup on strategy stop
  - [ ] Signal generation interface
  - [ ] Configuration and parameter management
- [ ] Create example strategies in `backend/src/strategy_lib/examples/`:
  - [ ] `sma_cross.py` - Simple moving average crossover
  - [ ] `mean_revert.py` - Mean reversion strategy
  - [ ] Parameter validation and defaults
- [ ] Create strategy registry in `backend/src/strategy_lib/registry.py`:
  - [ ] Dynamic strategy loading
  - [ ] Strategy validation and registration
  - [ ] Parameter schema definitions
- [ ] Create strategy runner in `backend/src/services/strategy/main.py`:
  - [ ] Use TWS client IDs 15-29 for strategy instances
  - [ ] Scheduled bar-driven event loop
  - [ ] Fetch latest bars from `candles` table
  - [ ] Call strategy `on_bar` methods
- [ ] Signal and order management:
  - [ ] Insert all signals to `signals` table for analysis
  - [ ] Check if strategy enabled and risk limits OK
  - [ ] Call Trader REST API to place orders
  - [ ] Tag orders/executions with strategy_id
- [ ] Multi-strategy support:
  - [ ] Run multiple strategies simultaneously
  - [ ] Assign distinct client IDs if TWS reads needed
  - [ ] Support purely DB-driven strategies
- [ ] Configuration management:
  - [ ] Load parameters from `strategies.params_json`
  - [ ] Hot-reload on parameter changes
  - [ ] Validate parameter schemas
- [ ] Health monitoring:
  - [ ] Implement `/healthz` endpoint
  - [ ] Monitor strategy execution status
- [ ] Test: Enabling sma_cross produces signals and paper orders
- [ ] Test: Multiple strategies can run concurrently
- [ ] Test: Parameter changes trigger strategy reload

## 11) Backtester Service (backtester)
### Tasks:
- [ ] Create `backend/src/services/backtester/main.py`
  - [ ] CLI interface for command-line backtesting
  - [ ] REST API for web-triggered backtests
  - [ ] On-demand service (not always running)
- [ ] Backtest execution engine:
  - [ ] Load strategy from strategy_lib
  - [ ] Use only existing `candles` data (no external fetching)
  - [ ] Simulate order execution with realistic fills
  - [ ] Apply commission and slippage models
- [ ] Configuration parameters:
  - [ ] Strategy name and parameters
  - [ ] Symbol list and timeframe
  - [ ] Date range for backtest
  - [ ] Commission settings (default $0.005/share, min $1)
  - [ ] Slippage settings (default 1 tick per side)
  - [ ] Partial fills simulation
- [ ] Results generation:
  - [ ] Calculate performance metrics (PnL, Sharpe, MaxDD)
  - [ ] Count total trades and win rate
  - [ ] Store results in `backtest_runs` table
  - [ ] Store individual trades in `backtest_trades` table
- [ ] Optional features:
  - [ ] Generate basic performance plots
  - [ ] Save plots to local folder
  - [ ] Export results to CSV
- [ ] API endpoints:
  - [ ] `POST /backtests` - Start new backtest
  - [ ] `GET /backtests/{id}` - Get backtest results
  - [ ] `GET /backtests` - List all backtests
- [ ] Test: Sample backtest runs create results with Sharpe, MaxDD, trade count
- [ ] Test: Backtest uses only local candles data
- [ ] Test: Results are properly stored in database

## 12) Backend API Gateway
### Tasks:
- [ ] Create `backend/src/api/main.py`
  - [ ] FastAPI application as API gateway
  - [ ] CORS configuration for frontend
  - [ ] Route aggregation from all services
  - [ ] Authentication middleware (if needed later)
- [ ] API endpoints for frontend:
  - [ ] `GET /api/account` - Account summary data
  - [ ] `GET /api/positions` - Current positions
  - [ ] `GET /api/orders` - Order history and status
  - [ ] `GET /api/ticks?symbol=&limit=` - Recent tick data
  - [ ] `POST /api/strategies/{id}/enable` - Enable/disable strategy
  - [ ] `PUT /api/strategies/{id}/params` - Update strategy parameters
  - [ ] `POST /api/backtests` - Trigger new backtest
  - [ ] `POST /api/watchlist` - Add/remove symbols from watchlist
  - [ ] `GET /api/health` - Service health status
- [ ] WebSocket endpoints:
  - [ ] `/ws/account` - Real-time account updates
  - [ ] `/ws/market` - Live market data
  - [ ] `/ws/orders` - Order status updates
- [ ] Test: All API endpoints return correct data
- [ ] Test: WebSocket connections work properly

## 13) Frontend Dashboard
### Tasks:
- [ ] Set up frontend structure in `frontend/`
  - [ ] Choose framework (React/Vue/Vanilla JS + HTMX)
  - [ ] Set up build system (Vite/Webpack)
  - [ ] Configure development server
- [ ] Create `frontend/src/services/api.js`
  - [ ] HTTP client for backend API calls
  - [ ] WebSocket client for real-time updates
  - [ ] Error handling and retry logic
- [ ] Web pages:
  - [ ] **Overview page**: equity/PnL, positions, recent orders/executions, service health
  - [ ] **Market page**: live quotes from watchlist
  - [ ] **Strategies page**: list strategies, enable/disable, edit params (JSON)
  - [ ] **Backtests page**: run new backtests, list results
  - [ ] **Logs page**: tail service logs from database
- [ ] REST API endpoints:
  - [ ] `GET /api/account` - Account summary data
  - [ ] `GET /api/positions` - Current positions
  - [ ] `GET /api/orders` - Order history and status
  - [ ] `GET /api/ticks?symbol=&limit=` - Recent tick data
  - [ ] `POST /api/strategies/{id}/enable` - Enable/disable strategy
  - [ ] `PUT /api/strategies/{id}/params` - Update strategy parameters
  - [ ] `POST /api/backtests` - Trigger new backtest
  - [ ] `POST /api/watchlist` - Add/remove symbols from watchlist
  - [ ] `GET /api/health` - Service health status
- [ ] Real-time updates:
  - [ ] WebSocket connections for live data
  - [ ] HTMX polling for periodic updates
  - [ ] Service health monitoring with alerts
- [ ] User interface:
  - [ ] Responsive design for desktop/mobile
  - [ ] Dark/light theme support
  - [ ] Data tables with sorting and filtering
  - [ ] Form validation and error handling
- [ ] Test: UI shows live database data
- [ ] Test: Toggling strategy affects runner within seconds
- [ ] Test: All API endpoints return correct data

## 14) Risk Management & Safety
### Tasks:
- [ ] Define risk limit types in database:
  - [ ] `max_notional_per_order` - Maximum dollar amount per single order
  - [ ] `max_notional_per_symbol` - Maximum position size per symbol
  - [ ] `max_daily_loss` - Maximum daily loss threshold
  - [ ] `block_live_trading_until` - UTC timestamp to block live trading
- [ ] Implement risk checking in Trader service:
  - [ ] Pre-order risk validation
  - [ ] Real-time position monitoring
  - [ ] Daily P&L tracking
  - [ ] Risk limit enforcement logic
- [ ] Risk management functions:
  - [ ] Load risk limits from `risk_limits` table
  - [ ] Calculate current exposure by symbol
  - [ ] Track daily P&L against limits
  - [ ] Generate risk alerts and notifications
- [ ] Audit and logging:
  - [ ] Log all risk decisions with reasoning
  - [ ] Store risk violations in database
  - [ ] Alert on limit breaches
- [ ] Safety switches:
  - [ ] Live trading safety checks
  - [ ] Emergency stop functionality
  - [ ] Risk limit override controls
- [ ] Test: Exceeding any limit yields clean rejection with reason
- [ ] Test: Risk decisions are properly audited in logs
- [ ] Test: Emergency stops work immediately

## 15) Multiple Concurrent Strategies
### Tasks:
- [ ] Strategy scheduling system:
  - [ ] Query `strategies` table for enabled strategies
  - [ ] Schedule each enabled strategy independently
  - [ ] Handle different timeframes per strategy
  - [ ] Manage strategy lifecycle (start/stop/restart)
- [ ] Strategy isolation:
  - [ ] Tag all signals with `strategy_id`
  - [ ] Tag all orders with `strategy_id`
  - [ ] Tag all executions with `strategy_id`
  - [ ] Separate strategy state management
- [ ] Client ID management:
  - [ ] Allocate dedicated client ID ranges (15-29)
  - [ ] Handle client ID conflicts gracefully
  - [ ] Support strategies that don't need TWS connections
  - [ ] Automatic client ID cleanup on strategy stop
- [ ] Resource management:
  - [ ] Memory isolation between strategies
  - [ ] CPU scheduling and limits
  - [ ] Database connection pooling
- [ ] Monitoring and control:
  - [ ] Individual strategy health monitoring
  - [ ] Performance metrics per strategy
  - [ ] Enable/disable individual strategies
- [ ] Test: Two strategies trade different symbols without client ID conflicts
- [ ] Test: Strategies can be enabled/disabled independently
- [ ] Test: Strategy failures don't affect other strategies

## 16) Observability & Resilience
### Tasks:
- [ ] Connection resilience:
  - [ ] Exponential backoff with jitter for TWS reconnections
  - [ ] Database connection retry with backoff
  - [ ] Circuit breaker pattern for failing connections
  - [ ] Connection health monitoring
- [ ] Health monitoring system:
  - [ ] Heartbeat updates to `health` table every 15 seconds
  - [ ] Dashboard warnings for services stale >60 seconds
  - [ ] Service dependency health checks
  - [ ] Automated health alerts
- [ ] Graceful shutdown handling:
  - [ ] SIGTERM signal handling in all services
  - [ ] Cancel in-flight TWS requests
  - [ ] Flush database transactions
  - [ ] Clean up resources and connections
- [ ] Comprehensive logging:
  - [ ] Structured JSON logs to stdout
  - [ ] Optional database log sampling
  - [ ] Log correlation IDs across services
  - [ ] Performance and error metrics
- [ ] Recovery mechanisms:
  - [ ] Auto-resubscribe to market data after reconnect
  - [ ] State recovery from database
  - [ ] Transaction rollback on failures
- [ ] Test: Kill TWS, services auto-recover and resubscribe
- [ ] Test: Database connection failures are handled gracefully
- [ ] Test: Graceful shutdown completes within timeout

## 17) Build System & Testing
### Tasks:
- [ ] Create `Makefile` with common targets:
  - [ ] `make up` - Start all services with docker compose
  - [ ] `make down` - Stop all services
  - [ ] `make db.up` - Start only database
  - [ ] `make db.migrate` - Run Alembic migrations
  - [ ] `make seed` - Seed database with test data
  - [ ] `make test` - Run all tests
  - [ ] `make test.unit` - Run unit tests only
  - [ ] `make test.integration` - Run integration tests
  - [ ] `make logs` - Follow service logs
  - [ ] `make clean` - Clean up containers and volumes
- [ ] Unit test suite in `backend/tests/unit/`:
  - [ ] Configuration loading and validation tests
  - [ ] Client ID allocation and collision tests
  - [ ] Risk checking logic tests
  - [ ] Database model and schema tests
  - [ ] Idempotent upsert operation tests
  - [ ] Strategy interface and example tests
- [ ] Integration test suite in `backend/tests/integration/`:
  - [ ] End-to-end order placement and fill lifecycle
  - [ ] TWS connection and reconnection tests
  - [ ] Market data subscription and processing
  - [ ] Multi-service communication tests
  - [ ] Database migration and rollback tests
- [ ] Test configuration:
  - [ ] Pytest configuration with fixtures
  - [ ] Test database setup and teardown
  - [ ] Mock TWS connections for unit tests
  - [ ] Skip integration tests if TWS not available
- [ ] Test: `pytest -q` runs green (integration skipped if TWS absent)
- [ ] Test: All Make targets work correctly

## 18) Live Trading Safety Switches
### Tasks:
- [ ] Implement safety switch validation:
  - [ ] Check `ENABLE_LIVE=1` environment variable
  - [ ] Check `USE_PAPER=0` for live mode
  - [ ] Check `TWS_PORT=7496` for live TWS connection
  - [ ] Check `DRY_RUN=0` to disable simulation mode
  - [ ] Check `risk_limits.block_live_trading_until` is in the past
- [ ] Safety validation in Trader service:
  - [ ] Validate all conditions before any live order
  - [ ] Return clear error messages for missing conditions
  - [ ] Log all safety check results
  - [ ] Block live orders if any condition fails
- [ ] Safety monitoring:
  - [ ] Dashboard indicators for live trading status
  - [ ] Alerts when safety switches change
  - [ ] Audit log of safety switch changes
- [ ] Emergency controls:
  - [ ] Ability to immediately disable live trading
  - [ ] Cancel all pending live orders
  - [ ] Switch back to paper mode quickly
- [ ] Test: Any missing safety condition blocks live orders with clear error
- [ ] Test: All safety switches work independently
- [ ] Test: Emergency stop functions work immediately

## 19) Future Enhancements (Post-MVP)
### Backlog Items:
- [ ] **Advanced Order Types**:
  - [ ] Bracket/OCO helpers in Trader payloads
  - [ ] TODO: Decide if bracket orders are single logical orders with multiple legs or separate orders
  - [ ] Trailing stop orders
  - [ ] Conditional orders
- [ ] **Data Management**:
  - [ ] Symbol loader job for periodic listing updates
  - [ ] Data quality monitoring and alerts
  - [ ] Historical data gap filling
  - [ ] Data retention policies and archiving
- [ ] **Strategy Development**:
  - [ ] Basic feature store (rolling factors) for strategy R&D
  - [ ] Strategy backtesting optimization
  - [ ] Portfolio-level strategies
  - [ ] Strategy performance attribution
- [ ] **Export and Integration**:
  - [ ] CSV export endpoints for all data
  - [ ] JSON/REST data feeds
  - [ ] Third-party integration webhooks
- [ ] **Bug Fixes and Improvements**:
  - [x] Fix timezone issue: Added `tzdata==2023.3` to requirements.txt to resolve "No time zone found with key US/Eastern" warnings from ib-insync execution detail parsing ✅
  - [ ] Add database persistence to Account Service streaming data (currently streaming to WebSocket only)
  - [ ] Implement proper error recovery for TWS disconnections in streaming services
- [ ] **Monitoring and Metrics**:
  - [ ] Optional Prometheus metrics integration
  - [ ] Grafana dashboard templates
  - [ ] Performance monitoring and alerting
- [ ] **Multi-Asset Support**:
  - [ ] Options trading support
  - [ ] Futures contract handling
  - [ ] Forex pair support
  - [ ] Crypto asset integration (if needed)
  - [ ] Framework should not hardcode US equities assumptions

### Notes:
- These items are not required for MVP but provide growth path
- Prioritize based on actual usage patterns
- Consider user feedback before implementing