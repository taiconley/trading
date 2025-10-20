# Observability & Resilience

## Overview

This document describes the observability and resilience features implemented in the trading system (Task 17). These features ensure system reliability, fault tolerance, and comprehensive monitoring across all services.

## 📊 Features Implemented

### 1. **Circuit Breaker Pattern** ✅
Prevents cascading failures by temporarily blocking operations to consistently failing services.

**Location**: `backend/src/common/circuit_breaker.py`

**Key Features**:
- Three states: CLOSED (normal), OPEN (blocking), HALF_OPEN (testing recovery)
- Configurable failure thresholds and timeout periods
- Automatic recovery attempts
- Statistics tracking (failures, successes, rejections)
- Global circuit breakers for database and TWS connections

**Usage**:
```python
from common.circuit_breaker import get_database_circuit_breaker

# Get circuit breaker
cb = get_database_circuit_breaker()

# Execute operation through circuit breaker
async def database_operation():
    result = await cb.call(my_db_function, arg1, arg2)
    return result

# Check circuit breaker status
status = cb.get_state()
# Returns: {"state": "closed", "failure_count": 0, ...}
```

**Configuration**:
- Database CB: 5 failures threshold, 30s timeout
- TWS CB: 3 failures threshold, 60s timeout

### 2. **Correlation IDs** ✅
Enables distributed tracing of requests across multiple services.

**Location**: `backend/src/common/correlation.py`

**Key Features**:
- Automatic correlation ID generation
- Context propagation across async operations
- HTTP header injection/extraction
- Log filter integration
- Request duration tracking

**Usage**:
```python
from common.correlation import correlation_context, with_correlation

# Using context manager
with correlation_context(correlation_id="abc-123"):
    # All operations will have this correlation ID
    logger.info("Processing order")
    # Log will include: {"correlation_id": "abc-123", ...}

# Using decorator
@with_correlation
async def process_order(order_id: int):
    # Correlation ID automatically managed
    logger.info(f"Processing order {order_id}")
```

**FastAPI Integration**:
```python
from common.correlation import CorrelationMiddleware

app = FastAPI()
app.add_middleware(CorrelationMiddleware)
```

**Log Filter**:
```python
from common.correlation import CorrelationLogFilter

logger = logging.getLogger(__name__)
logger.addFilter(CorrelationLogFilter())
```

### 3. **Graceful Shutdown** ✅
Standardized signal handling for clean shutdown across all services.

**Location**: `backend/src/common/shutdown.py`

**Key Features**:
- SIGTERM and SIGINT handling
- Cleanup task registration and execution
- Timeout protection
- Shutdown status tracking
- Support for both sync and async cleanup tasks

**Usage**:
```python
from common.shutdown import GracefulShutdownHandler

# Initialize handler
shutdown_handler = GracefulShutdownHandler("my_service", timeout=30.0)

# Register cleanup tasks
async def cleanup_database():
    await db.close()

async def disconnect_tws():
    await tws_client.disconnect()

shutdown_handler.add_cleanup_task(cleanup_database)
shutdown_handler.add_cleanup_task(disconnect_tws)

# Install signal handlers
shutdown_handler.install()

# Main loop
while not shutdown_handler.should_shutdown():
    # Do work...
    await asyncio.sleep(1)

# Manual shutdown
await shutdown_handler.shutdown()
```

### 4. **Health Monitoring & Aggregation** ✅
Centralized health monitoring for all services with stale detection and alerting.

**Location**: `backend/src/common/health_monitor.py`

**Key Features**:
- System-wide health aggregation
- Stale service detection (>60s threshold)
- Critical service identification
- Health status levels: HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN
- Continuous monitoring loop

**Usage**:
```python
from common.health_monitor import get_health_monitor

# Get monitor instance
monitor = get_health_monitor()

# Get overall system health
health = await monitor.get_system_health()
# Returns:
# {
#   "status": "healthy",
#   "message": "All services healthy",
#   "services": [...],
#   "summary": {
#     "total": 6,
#     "healthy": 6,
#     "unhealthy": 0,
#     "stale": 0,
#     "critical_unhealthy": 0
#   }
# }

# Get stale services
stale = await monitor.get_stale_services(threshold_seconds=60)

# Get unhealthy services
unhealthy = await monitor.get_unhealthy_services()

# Start continuous monitoring (in background task)
asyncio.create_task(monitor.monitor_loop(interval=15.0))
```

**Critical Services** (system unhealthy if any are down):
- `postgres`
- `trader`
- `account`

### 5. **Database Connection Resilience** ✅
Already implemented with enhancements.

**Location**: `backend/src/common/db.py`

**Features**:
- Connection pooling (10 connections, 20 overflow)
- Pre-ping validation before use
- Automatic connection recycling (1 hour)
- Exponential backoff retry logic
- Circuit breaker integration

**Functions**:
```python
from common.db import (
    execute_with_retry,
    execute_with_circuit_breaker,
    check_database_health,
    get_circuit_breaker_status
)

# Retry with exponential backoff
def my_operation(session):
    return session.query(...).all()

result = execute_with_retry(my_operation, max_retries=3)

# Use circuit breaker
result = await execute_with_circuit_breaker(my_operation)

# Check health
health = check_database_health()
# Returns: {"status": "healthy", "response_time_ms": 2.5}

# Check circuit breaker status
cb_status = get_circuit_breaker_status()
```

### 6. **TWS Connection Resilience** ✅
Already implemented in EnhancedIBClient.

**Location**: `backend/src/tws_bridge/ib_client.py`

**Features**:
- Exponential backoff with jitter
- Configurable max reconnect attempts
- Request throttling (pacing protection)
- Auto-resubscribe after reconnect
- Connection state tracking
- Event handler management

**Configuration**:
```python
from tws_bridge.ib_client import EnhancedIBClient

client = EnhancedIBClient(
    client_id=24,
    max_reconnect_attempts=10,
    base_backoff=1.0,
    max_backoff=60.0,
    jitter=True
)

# Connection includes automatic retry
await client.connect()
```

### 7. **Enhanced Logging** ✅
Structured JSON logging with correlation ID support.

**Location**: `backend/src/common/logging.py`

**Features**:
- JSON-structured output
- Correlation ID tracking
- Request duration tracking
- Exception details
- Service name tagging
- Docker-friendly stdout logging

**Log Format**:
```json
{
  "timestamp": "2025-10-20T10:30:45.123Z",
  "service": "trader",
  "level": "INFO",
  "logger": "trader.orders",
  "message": "Order placed successfully",
  "correlation_id": "abc-123-def-456",
  "request_duration_ms": 45.2,
  "module": "main",
  "function": "place_order",
  "line": 234
}
```

## 🔌 API Endpoints

### Health Monitoring Endpoints (API Gateway)

#### `GET /api/health`
Get comprehensive system health status.

**Response**:
```json
{
  "status": "healthy",
  "message": "All services healthy",
  "services": [
    {
      "service": "trader",
      "status": "healthy",
      "last_update": "2025-10-20T10:30:45Z",
      "age_seconds": 5.2,
      "is_critical": true,
      "is_healthy": true,
      "is_stale": false
    }
  ],
  "summary": {
    "total": 6,
    "healthy": 6,
    "unhealthy": 0,
    "stale": 0,
    "critical_unhealthy": 0
  },
  "timestamp": "2025-10-20T10:30:50Z"
}
```

#### `GET /api/health/detailed`
Detailed health with live service pings.

**Response**:
```json
{
  "database_health": {...},
  "live_services": {
    "account": {
      "status": "healthy",
      "response_time_ms": 12.5,
      "details": {...}
    }
  },
  "gateway": "healthy"
}
```

#### `GET /api/health/circuit-breakers`
Get circuit breaker status.

**Response**:
```json
{
  "circuit_breakers": {
    "database": {
      "name": "database",
      "state": "closed",
      "failure_count": 0,
      "total_failures": 5,
      "total_successes": 1234,
      "total_rejected": 0
    },
    "tws": {
      "name": "tws_connection",
      "state": "closed",
      ...
    }
  }
}
```

## 🔍 Testing

### Manual Testing

#### 1. Test Circuit Breaker
```python
# Simulate database failures
from common.circuit_breaker import get_database_circuit_breaker

cb = get_database_circuit_breaker()

# Check initial state
print(cb.get_state())  # state: "closed"

# Trigger failures (manually kill DB or simulate)
# After 5 failures, circuit will OPEN
print(cb.get_state())  # state: "open"

# Wait 30 seconds for recovery attempt
# Circuit will transition to HALF_OPEN
```

#### 2. Test Correlation IDs
```bash
# Send request with correlation ID
curl -H "X-Correlation-ID: test-123" http://localhost:8000/api/orders

# Check logs - all log entries will have correlation_id: "test-123"
docker logs trading-trader | grep "test-123"
```

#### 3. Test Graceful Shutdown
```bash
# Send SIGTERM to service
docker kill -s SIGTERM trading-trader

# Watch logs for cleanup sequence
docker logs -f trading-trader
# Should see:
# "Received SIGTERM signal - initiating graceful shutdown"
# "Executing cleanup task 1/3: close_database"
# "Graceful shutdown completed in 2.34s"
```

#### 4. Test Health Monitoring
```bash
# Get system health
curl http://localhost:8000/api/health | jq

# Get detailed health
curl http://localhost:8000/api/health/detailed | jq

# Stop a service and check for stale detection
docker stop trading-marketdata
sleep 65
curl http://localhost:8000/api/health | jq
# Should show marketdata as stale
```

### Integration Testing

#### Test TWS Reconnection
1. Kill TWS while services are running
2. Services should log reconnection attempts with backoff
3. Restart TWS
4. Services should automatically reconnect and resubscribe

#### Test Database Failure
1. Stop postgres container
2. Circuit breaker should OPEN after 5 failures
3. Requests should be rejected immediately
4. Restart postgres
5. Circuit breaker should transition to HALF_OPEN
6. After 2 successes, return to CLOSED

## 📈 Monitoring Best Practices

### 1. Health Checks
- Monitor `/api/health` endpoint every 30 seconds
- Alert on `status: "unhealthy"` or `status: "degraded"`
- Watch for `critical_unhealthy > 0`

### 2. Circuit Breakers
- Monitor circuit breaker state changes
- Alert when circuit transitions to OPEN
- Track `total_rejected` for impact assessment

### 3. Stale Services
- Services not updating for >60s are considered stale
- May indicate deadlock, crash, or network issues
- Investigate immediately

### 4. Correlation IDs
- Use for debugging multi-service transactions
- Search logs by correlation ID to trace request flow
- Include in error reports

## 🔧 Configuration

### Environment Variables
```ini
# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Circuit Breaker
DB_CIRCUIT_FAILURE_THRESHOLD=5
DB_CIRCUIT_TIMEOUT=30
TWS_CIRCUIT_FAILURE_THRESHOLD=3
TWS_CIRCUIT_TIMEOUT=60

# Health Monitoring
HEALTH_STALE_THRESHOLD=60  # seconds

# Shutdown
SHUTDOWN_TIMEOUT=30  # seconds
```

### Database Circuit Breaker Settings
Edit in `backend/src/common/circuit_breaker.py`:
```python
_database_circuit_breaker = CircuitBreaker(
    name="database",
    failure_threshold=5,  # Adjust as needed
    success_threshold=2,
    timeout=30.0,
    expected_exception=(OperationalError, DisconnectionError)
)
```

## 📝 Summary

**Task 17 Implementation Status**: ✅ **COMPLETE**

### Implemented Features:
- ✅ Circuit breaker pattern for DB and TWS
- ✅ Correlation IDs for distributed tracing
- ✅ Graceful shutdown with cleanup tasks
- ✅ Health monitoring and aggregation
- ✅ Database connection resilience with retry logic
- ✅ TWS connection resilience with exponential backoff
- ✅ Enhanced structured logging with correlation support
- ✅ API endpoints for health and circuit breaker status

### Already Existing:
- ✅ TWS exponential backoff and jitter
- ✅ Database retry logic
- ✅ Service heartbeat updates (every 30s)
- ✅ Auto-resubscribe to market data
- ✅ Connection pooling
- ✅ Structured JSON logging

### Benefits:
- **Fault Tolerance**: Circuit breakers prevent cascading failures
- **Observability**: Correlation IDs enable end-to-end request tracing
- **Reliability**: Graceful shutdown ensures no data loss
- **Monitoring**: Centralized health monitoring detects issues early
- **Recovery**: Automatic recovery mechanisms minimize downtime

### Next Steps:
- Task 18: Build System & Testing
- Task 19: Live Trading Safety Switches (already mostly complete)

