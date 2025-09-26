# Strategy Library

This library provides the core strategy interface and implementations for the trading bot. It supports both backtesting and live trading with a flexible, extensible architecture.

## Architecture

### Core Components

1. **BaseStrategy** (`base.py`) - Abstract base class that all strategies must inherit from
2. **StrategyRegistry** (`registry.py`) - Dynamic loading and management of strategy classes
3. **Strategy Implementations** (`sma_cross.py`, `mean_revert.py`) - Reference implementations demonstrating the interface
4. **Strategy Service** (`../services/strategy/`) - Live execution service that runs strategies

### Key Features

- **Multi-Symbol Support**: Strategies can trade multiple symbols simultaneously
- **Flexible Signal Generation**: Support for BUY/SELL/HOLD/EXIT signals with strength indicators
- **Parameter Management**: Hot-reloadable parameters via database configuration
- **Risk Integration**: Built-in position tracking and risk management hooks
- **Event-Driven**: Real-time bar processing with database persistence
- **Backtesting Ready**: Same interface works for both live and historical testing

## Strategy Interface

### Required Methods

```python
class MyStrategy(BaseStrategy):
    async def on_start(self, instruments: Dict[str, Any]) -> None:
        """Initialize strategy when starting"""
        pass
    
    async def on_bar(self, symbol: str, timeframe: str, bars: pd.DataFrame) -> List[StrategySignal]:
        """Process new bar data and generate signals"""
        pass
    
    async def on_stop(self) -> None:
        """Cleanup when strategy is stopping"""
        pass
```

### Optional Methods

```python
    async def on_tick(self, symbol: str, tick_data: Dict[str, Any]) -> None:
        """Handle tick-level data (for HFT strategies)"""
        pass
    
    async def on_signal_executed(self, signal: StrategySignal, success: bool) -> None:
        """Handle signal execution feedback"""
        pass
    
    async def on_parameter_update(self, new_params: Dict[str, Any]) -> None:
        """Handle hot parameter updates"""
        pass
```

## Configuration

### Strategy Configuration

```python
config = StrategyConfig(
    strategy_id="my_strategy_001",
    name="My Strategy",
    symbols=["AAPL", "MSFT", "SPY"],
    enabled=True,
    bar_timeframe="1 min",
    lookback_periods=50,
    parameters={
        "param1": "value1",
        "param2": 42
    }
)
```

### Database Configuration

Strategies are configured in the `strategies` table:

```sql
INSERT INTO strategies (strategy_id, name, enabled, params_json) VALUES (
    'sma_cross_001',
    'SMA_Crossover', 
    true,
    '{
        "symbols": ["AAPL", "MSFT"],
        "short_period": 20,
        "long_period": 50,
        "fixed_position_size": 100
    }'
);
```

## Strategy Implementations

### 1. SMA Crossover (`sma_cross.py`)

A trend-following strategy that generates signals based on moving average crossovers.

**Parameters:**
- `short_period`: Short-term SMA period (default: 20)
- `long_period`: Long-term SMA period (default: 50)
- `min_cross_strength`: Minimum cross strength threshold (default: 0.01)
- `volume_filter`: Enable volume filtering (default: true)
- `fixed_position_size`: Position size in shares (default: 100)

**Signals:**
- BUY: When short SMA crosses above long SMA
- SELL: When short SMA crosses below long SMA

### 2. Mean Reversion (`mean_revert.py`)

A mean reversion strategy using Bollinger Bands and RSI.

**Parameters:**
- `bb_period`: Bollinger Bands period (default: 20)
- `bb_std_dev`: Standard deviations for bands (default: 2.0)
- `rsi_period`: RSI calculation period (default: 14)
- `rsi_overbought`: RSI overbought level (default: 70)
- `rsi_oversold`: RSI oversold level (default: 30)
- `base_position_size`: Base position size (default: 100)

**Signals:**
- BUY: Price below lower BB + RSI oversold
- SELL: Price above upper BB + RSI overbought
- EXIT: Profit target, stop loss, or mean reversion complete

## Signal Generation

### StrategySignal Structure

```python
signal = StrategySignal(
    symbol="AAPL",
    signal_type=SignalType.BUY,
    strength=Decimal("0.75"),  # 0.0 to 1.0
    price=Decimal("150.00"),   # Suggested execution price
    quantity=100,              # Suggested position size
    metadata={                 # Additional context
        "reason": "sma_crossover",
        "short_sma": 149.50,
        "long_sma": 148.75
    }
)
```

### Signal Types

- `BUY`: Open long position
- `SELL`: Open short position  
- `HOLD`: No action (informational)
- `EXIT`: Close existing position

## Development Workflow

### 1. Create a New Strategy

```python
from strategy_lib import BaseStrategy, StrategyConfig, strategy

@strategy(name="My_Strategy", description="My custom strategy")
class MyStrategy(BaseStrategy):
    async def on_start(self, instruments):
        self.set_state(StrategyState.RUNNING)
        # Initialize strategy state
    
    async def on_bar(self, symbol, timeframe, bars):
        # Strategy logic here
        signals = []
        
        if self.should_buy(bars):
            signal = self.create_signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=Decimal("0.8"),
                quantity=100
            )
            signals.append(signal)
        
        return signals
    
    async def on_stop(self):
        self.set_state(StrategyState.STOPPED)
```

### 2. Test the Strategy

```python
# Use the test interface
python backend/src/strategy_lib/test_interface.py
```

### 3. Register in Database

```sql
INSERT INTO strategies (strategy_id, name, enabled, params_json) VALUES (
    'my_strategy_001',
    'My_Strategy',
    false,  -- Start disabled for testing
    '{
        "symbols": ["AAPL"],
        "param1": "value1"
    }'
);
```

### 4. Enable for Live Trading

```sql
UPDATE strategies SET enabled = true WHERE strategy_id = 'my_strategy_001';
```

## Service Integration

### Strategy Service (Port 8005)

The strategy service automatically:
- Loads enabled strategies from the database
- Fetches latest bar data every minute
- Processes bars through each strategy
- Stores generated signals in the database
- Places orders through the Trader service
- Broadcasts updates via WebSocket

### API Endpoints

- `GET /healthz` - Service health check
- `GET /strategies` - List active strategies
- `POST /strategies/{id}/reload` - Reload specific strategy
- `POST /strategies/reload-all` - Reload all strategies
- `WebSocket /ws` - Real-time strategy updates

### Database Integration

The service integrates with these tables:
- `strategies` - Strategy configurations
- `candles` - Historical bar data (input)
- `signals` - Generated signals (output)
- `orders` - Placed orders (via Trader service)

## Best Practices

### Strategy Development

1. **Keep strategies stateless where possible** - Store state in database
2. **Handle missing data gracefully** - Check for sufficient bars
3. **Use proper error handling** - Log errors without crashing
4. **Validate parameters** - Implement parameter schemas
5. **Test thoroughly** - Use the test interface before live trading

### Performance

1. **Efficient bar processing** - Only process new data
2. **Memory management** - Limit internal history storage
3. **Database efficiency** - Batch database operations
4. **Signal filtering** - Only generate actionable signals

### Risk Management

1. **Position tracking** - Use built-in position methods
2. **Signal strength** - Use meaningful strength values
3. **Parameter validation** - Validate all inputs
4. **Stop conditions** - Implement proper exit logic

## Troubleshooting

### Common Issues

1. **Strategy not loading**
   - Check strategy registration
   - Verify parameter schema
   - Check database configuration

2. **No signals generated**
   - Verify sufficient historical data
   - Check strategy logic conditions
   - Review parameter values

3. **Orders not placed**
   - Check strategy enabled flag
   - Verify Trader service connection
   - Review risk limit settings

### Debugging

1. **Enable debug logging**
2. **Use test interface for isolated testing**
3. **Check service health endpoints**
4. **Review database signal records**

## Future Enhancements

- Portfolio-level strategies
- Advanced order types (brackets, trailing stops)
- Real-time risk monitoring
- Strategy performance analytics
- Multi-timeframe strategies
- Options and futures support
