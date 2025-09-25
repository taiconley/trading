# Market Data Service

## Overview

The Market Data Service is a real-time market data streaming service that follows TWS best practices for event-driven data processing. It subscribes to market data for symbols in the watchlist table, processes tick data, and provides both REST API and WebSocket endpoints for real-time updates.

## Features

### ✅ Implemented

- **Event-driven streaming**: Uses TWS event handlers for real-time tick updates (following notes.md best practices)
- **Watchlist-based subscriptions**: Only subscribes to symbols in the `watchlist` table
- **Subscription limits**: Enforces `MAX_SUBSCRIPTIONS` limit from configuration
- **Database integration**: Stores tick data in the `ticks` table
- **Dynamic updates**: Listens for `watchlist_update` notifications to add/remove subscriptions
- **Health monitoring**: Provides `/healthz` endpoint and updates database health status
- **WebSocket support**: Real-time streaming to connected clients
- **TWS reconnection**: Automatic reconnection and resubscription on connection loss
- **Client ID management**: Uses designated client ID 12 for marketdata service

### Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   TWS/Gateway   │───▶│  Market Data     │───▶│   Database      │
│                 │    │    Service       │    │   (ticks)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   WebSocket     │
                       │    Clients      │
                       └─────────────────┘
```

## Configuration

The service uses the following configuration from environment variables:

- `TWS_HOST`: TWS/Gateway host (default: 172.17.0.1)
- `TWS_PORT`: TWS/Gateway port (7497 for paper, 7496 for live)
- `MAX_SUBSCRIPTIONS`: Maximum number of concurrent subscriptions (default: 50)
- `TWS_CLIENT_ID_BASE`: Base client ID (service uses base + 2 = 12)

## API Endpoints

### REST Endpoints

#### `GET /healthz`
Health check endpoint returning service status:
```json
{
  "status": "healthy",
  "service": "marketdata", 
  "client_id": 22,
  "connected": true,
  "subscriptions": 0,
  "max_subscriptions": 50,
  "watchlist_size": 0,
  "last_heartbeat": 1758840371.5022361
}
```

#### `GET /subscriptions`
Current market data subscriptions:
```json
{
  "subscriptions": [
    {
      "symbol": "AAPL",
      "bid": 150.25,
      "ask": 150.27,
      "last": 150.26,
      "bid_size": 100,
      "ask_size": 200,
      "last_size": 50
    }
  ],
  "count": 1,
  "max_allowed": 50
}
```

### WebSocket Endpoint

#### `WS /ws`
Real-time market data streaming. Messages include:

**Subscription Status:**
```json
{
  "type": "subscription_status",
  "data": {
    "subscribed_symbols": ["AAPL", "MSFT"],
    "count": 2
  }
}
```

**Tick Updates:**
```json
{
  "type": "tick_update", 
  "data": {
    "symbol": "AAPL",
    "bid": 150.25,
    "ask": 150.27,
    "last": 150.26,
    "bid_size": 100,
    "ask_size": 200,
    "last_size": 50,
    "timestamp": "2025-09-25T22:45:57.808957+00:00"
  }
}
```

## Database Schema

The service interacts with these database tables:

### `watchlist`
- Symbols to subscribe to for market data
- Service only subscribes to symbols present in this table

### `ticks` 
- Real-time tick data storage
- Columns: symbol, ts, bid, ask, last, bid_size, ask_size, last_size

### `health`
- Service health status tracking
- Updated every 30 seconds with heartbeat

## Event-Driven Architecture

Following the best practices from `notes.md`, the service uses:

1. **Ticker event handlers**: Each subscribed ticker has an `updateEvent` handler
2. **Non-blocking processing**: Tick data processing happens asynchronously
3. **Database writes**: Executed in thread pool to avoid blocking event loop
4. **WebSocket broadcasting**: Real-time updates to all connected clients

## Notifications

The service listens for `watchlist_update` notifications to dynamically:
- Add new subscriptions when symbols are added to watchlist
- Remove subscriptions when symbols are removed from watchlist

## Testing

Run the comprehensive test suite:
```bash
docker compose exec backend-marketdata python /app/src/services/marketdata/test_marketdata.py
```

Tests cover:
- Configuration loading
- Database connectivity  
- Client ID allocation
- Watchlist operations
- Health status updates
- Logging functionality

## Deployment

The service runs in Docker with:
- Port 8002 for REST API and WebSocket
- Automatic startup with docker compose
- Health checks and restart policies
- Structured JSON logging to stdout

## Monitoring

- Health endpoint for load balancer checks
- Database health status updates
- Structured logging with market data events
- WebSocket connection tracking
- TWS connection monitoring with auto-reconnection

## Next Steps

The Market Data Service is production-ready and implements all requirements from Task 7. Future enhancements could include:

- Real-time bar aggregation
- Multiple timeframe support
- Advanced filtering options
- Performance metrics collection
- Rate limiting for API endpoints
