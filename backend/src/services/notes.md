# TWS Data Streaming Guide

## Overview
This guide documents the proven methods for streaming real-time data from Interactive Brokers TWS/IB Gateway using the `ib-insync` library, based on extensive testing and troubleshooting.

## Key Learnings

### ✅ What Works: Event-Driven Streaming
The **correct approach** is to use **event handlers** with streaming subscriptions. This provides real-time updates as data changes.

### ❌ What Doesn't Work: Blocking Method Calls
Calling methods like `reqAccountUpdates()` and waiting for data to appear in `accountValues()` is **not reliable** for real-time streaming.

---

## Account Data Streaming

### 1. Account Values (Balance, Margin, etc.)

**Setup Event Handler:**
```python
def on_account_value(accountValue):
    if accountValue.account == target_account:
        print(f"💰 {accountValue.tag} = {accountValue.value} {accountValue.currency}")

# Connect event handler
ib.accountValueEvent += on_account_value
```

**Start Streaming:**
```python
await ib.reqAccountUpdatesAsync(account)
```

**Data Received:**
- Net Liquidation Value
- Cash Balance
- Buying Power
- Margin Requirements
- Available Funds
- Unrealized/Realized P&L
- And 50+ other account metrics

### 2. Position Updates

**Setup Event Handler:**
```python
def on_position(position):
    if position.account == target_account:
        print(f"📍 {position.contract.symbol} = {position.position} @ {position.avgCost}")

# Connect event handler
ib.positionEvent += on_position
```

**Start Streaming:**
```python
await ib.reqPositionsAsync()
```

**Data Received:**
- Symbol and quantity
- Average cost
- Market value
- Unrealized P&L per position

### 3. Portfolio P&L Updates

**Setup Event Handler:**
```python
def on_pnl(pnl):
    if hasattr(pnl, 'account') and pnl.account == target_account:
        print(f"📊 Daily P&L: {pnl.dailyPnL}, Unrealized: {pnl.unrealizedPnL}")

# Connect event handler
ib.pnlEvent += on_pnl
```

**Start Streaming:**
```python
pnl = await ib.reqPnLAsync(account, '')
```

### 4. Account Summary Updates

**Setup Event Handler:**
```python
def on_account_summary(accountSummary):
    if accountSummary.account == target_account:
        print(f"📈 {accountSummary.tag} = {accountSummary.value} {accountSummary.currency}")

# Connect event handler
ib.accountSummaryEvent += on_account_summary
```

**Start Streaming:**
```python
summary = await ib.reqAccountSummaryAsync()
```

---

## Market Data Streaming

### Real-Time Market Data

**Setup Event Handler:**
```python
def on_ticker_update(ticker):
    print(f"📈 {ticker.contract.symbol}: {ticker.marketPrice()}")

# Connect event handler
ib.tickerEvent += on_ticker_update
```

**Start Streaming:**
```python
contract = Stock('SPY', 'SMART', 'USD')
ticker = ib.reqMktData(contract, '', False, False)
```

### Historical Data

**One-time Request:**
```python
contract = Stock('SPY', 'SMART', 'USD')
bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='1 D',
    barSizeSetting='1 min',
    whatToShow='TRADES',
    useRTH=True
)
```

---

## Order and Execution Streaming

### Order Status Updates

**Setup Event Handler:**
```python
def on_order_status(trade):
    print(f"📋 Order {trade.order.orderId}: {trade.orderStatus.status}")

# Connect event handler
ib.orderStatusEvent += on_order_status
```

**Start Streaming:**
```python
await ib.reqAllOpenOrdersAsync()
```

### Execution Reports

**Setup Event Handler:**
```python
def on_execution(trade, fill):
    print(f"✅ Execution: {fill.contract.symbol} {fill.execution.shares} @ {fill.execution.price}")

# Connect event handler
ib.execDetailsEvent += on_execution
```

**Start Streaming:**
```python
await ib.reqExecutionsAsync(ExecutionFilter())
```

---

## Complete Working Example

```python
import asyncio
from ib_insync import IB, Stock

class TWSStreamer:
    def __init__(self):
        self.ib = IB()
        
    async def connect(self):
        await self.ib.connectAsync('172.17.0.1', 7497, clientId=11)
        
    def setup_event_handlers(self, account):
        # Account values
        def on_account_value(accountValue):
            if accountValue.account == account:
                print(f"💰 {accountValue.tag} = {accountValue.value} {accountValue.currency}")
        
        # Positions
        def on_position(position):
            if position.account == account:
                print(f"📍 {position.contract.symbol} = {position.position} @ {position.avgCost}")
        
        # Market data
        def on_ticker_update(ticker):
            print(f"📈 {ticker.contract.symbol}: {ticker.marketPrice()}")
        
        # Connect handlers
        self.ib.accountValueEvent += on_account_value
        self.ib.positionEvent += on_position
        self.ib.tickerEvent += on_ticker_update
        
    async def start_streaming(self, account):
        # Account data
        await self.ib.reqAccountUpdatesAsync(account)
        await self.ib.reqPositionsAsync()
        await self.ib.reqPnLAsync(account, '')
        
        # Market data example
        spy = Stock('SPY', 'SMART', 'USD')
        self.ib.reqMktData(spy, '', False, False)
        
    async def run(self):
        await self.connect()
        accounts = self.ib.managedAccounts()
        account = accounts[0]
        
        self.setup_event_handlers(account)
        await self.start_streaming(account)
        
        # Keep running to receive updates
        while True:
            await asyncio.sleep(1)

# Usage
streamer = TWSStreamer()
asyncio.run(streamer.run())
```

---

## Important Notes

### Client ID Management
- Each service needs a unique client ID
- Account service: 11
- Market data service: 12
- Historical service: 13
- Trading service: 14
- Strategy services: 15-29

### Connection Configuration
- **Host**: `172.17.0.1` (Docker bridge gateway)
- **Port**: `7497` (paper trading) or `7496` (live)
- **TWS Trusted IPs**: Must include container IP (`172.25.0.100`)

### Data Behavior
- **Streaming data**: Only updates when values change
- **Initial burst**: Upon connection, you receive current state
- **Paper accounts**: May have less frequent updates than live accounts
- **Market hours**: Some data only updates during market hours

### Error Handling
- Always wrap API calls in try-catch blocks
- Handle timezone issues (install `tzdata` package)
- Monitor connection status with `ib.isConnected()`
- Implement reconnection logic for production

### Performance Considerations
- Event handlers should be fast (avoid blocking operations)
- Use async/await for all TWS API calls
- Limit concurrent subscriptions to avoid rate limits
- Consider data filtering to reduce noise

---

## Database Integration Pattern

```python
async def on_account_value(accountValue):
    if accountValue.account == target_account:
        # Store in database
        await db.execute("""
            INSERT INTO account_values (account, tag, value, currency, timestamp)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (account, tag, currency) 
            DO UPDATE SET value = ?, timestamp = ?
        """, (
            accountValue.account,
            accountValue.tag,
            accountValue.value,
            accountValue.currency,
            datetime.utcnow(),
            accountValue.value,
            datetime.utcnow()
        ))
        
        # Broadcast to WebSocket clients
        await websocket_manager.broadcast({
            'type': 'account_update',
            'data': {
                'account': accountValue.account,
                'tag': accountValue.tag,
                'value': accountValue.value,
                'currency': accountValue.currency
            }
        })
```

This guide represents battle-tested approaches that work reliably in production environments.
