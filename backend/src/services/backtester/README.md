# Backtester Service

On-demand backtesting service that simulates strategy execution on historical data. Provides both CLI and REST API interfaces for running backtests, analyzing results, and comparing strategies.

## Features

- **Strategy Support**: Works with all strategies from `strategy_lib`
- **Multi-Symbol Support**: Full support for pairs trading, spread strategies, and portfolio strategies
- **Realistic Simulation**: Includes commission, slippage, and realistic order fills
- **Performance Metrics**: Comprehensive metrics including Sharpe ratio, max drawdown, win rate, profit factor
- **Database Storage**: Stores all results and trades for later analysis
- **Flexible Interface**: Both CLI tool and REST API
- **Historical Data**: Uses only local candles data (no external fetching)

## Multi-Symbol and Pairs Trading Support

The backtester fully supports strategies that need to analyze multiple symbols simultaneously, such as:
- **Pairs Trading**: Long one stock, short another based on ratio deviations
- **Spread Trading**: Trade spreads between related instruments
- **Portfolio Strategies**: Analyze and trade multiple symbols as a portfolio

### How It Works

1. **Traditional Single-Symbol Strategies**: Call `on_bar()` for each symbol individually
2. **Multi-Symbol Strategies**: Call `on_bar_multi()` with all symbols' data together

### Example: Pairs Trading

```python
class PairsTradingStrategy(BaseStrategy):
    @property
    def supports_multi_symbol(self) -> bool:
        return True  # Enable multi-symbol mode
    
    async def on_bar_multi(self, symbols, timeframe, bars_data):
        # Access both symbols together
        stock_a = symbols[0]
        stock_b = symbols[1]
        
        # Calculate price ratio
        ratio = bars_data[stock_a]['close'].iloc[-1] / bars_data[stock_b]['close'].iloc[-1]
        
        # Generate paired signals
        if ratio > threshold:
            return [
                self.create_signal(stock_a, SignalType.SELL, quantity=100),
                self.create_signal(stock_b, SignalType.BUY, quantity=100)
            ]
        return []
```

### Running Pairs Backtests

```bash
# CLI
python main.py cli \
  --strategy Pairs_Trading \
  --symbols AAPL MSFT \
  --timeframe "1 day" \
  --params '{"lookback_window": 20, "entry_threshold": 2.0}'

# Results show paired trades
# Trade 1: BUY 100 AAPL, P&L: $1,418.00
# Trade 2: SELL 100 MSFT, P&L: $-1,174.00
# Net Pair P&L: $244.00
```

The backtester tracks both legs independently and calculates individual P&L for each symbol, allowing you to analyze the performance of each leg of the pair.

## Architecture

### Components

1. **BacktestEngine** (`engine.py`) - Core simulation engine
   - Order execution simulation
   - Position tracking
   - P&L calculation
   - Performance metrics

2. **BacktesterService** (`main.py`) - Service layer
   - REST API endpoints
   - CLI interface
   - Database integration
   - Strategy loading

### Database Tables

- **backtest_runs**: Stores backtest metadata and summary metrics
  - `id`, `strategy_name`, `params_json`, `start_ts`, `end_ts`
  - `pnl`, `sharpe`, `maxdd`, `trades`, `created_at`

- **backtest_trades**: Stores individual trades from backtests
  - `id`, `run_id`, `symbol`, `side`, `qty`
  - `entry_ts`, `entry_px`, `exit_ts`, `exit_px`, `pnl`

## Usage

### CLI Interface

#### Basic Backtest

```bash
python main.py cli --strategy SMA_Crossover --symbols AAPL --timeframe "1 day"
```

#### With Custom Parameters

```bash
python main.py cli \
  --strategy SMA_Crossover \
  --symbols AAPL MSFT \
  --timeframe "1 day" \
  --params '{"short_period": 10, "long_period": 30}' \
  --initial-capital 50000 \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

#### CLI Options

- `--strategy`: Strategy name (from strategy registry)
- `--symbols`: List of symbols to trade (space-separated)
- `--timeframe`: Bar timeframe (default: "1 day")
- `--start-date`: Start date in YYYY-MM-DD format (optional)
- `--end-date`: End date in YYYY-MM-DD format (optional)
- `--initial-capital`: Starting capital (default: 100000)
- `--lookback`: Number of bars to provide to strategy (default: 100)
- `--params`: Strategy parameters as JSON string (optional)

### REST API

#### Start API Server

```bash
python main.py api --host 0.0.0.0 --port 8006
```

#### API Endpoints

**Health Check**
```bash
GET /healthz
```

**Run Backtest**
```bash
POST /backtests
Content-Type: application/json

{
  "strategy_name": "SMA_Crossover",
  "strategy_params": {
    "short_period": 10,
    "long_period": 30
  },
  "symbols": ["AAPL"],
  "timeframe": "1 day",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "initial_capital": 100000.0,
  "lookback_periods": 100
}
```

**Get Backtest Results**
```bash
GET /backtests/{run_id}
```

**List All Backtests**
```bash
GET /backtests?limit=50
```

**Get Backtest Trades**
```bash
GET /backtests/{run_id}/trades
```

### Python Integration

```python
import asyncio
from services.backtester import BacktesterService
from decimal import Decimal

async def run_backtest():
    service = BacktesterService()
    
    result = await service._run_backtest(
        strategy_name="SMA_Crossover",
        strategy_params={"short_period": 10, "long_period": 30},
        symbols=["AAPL"],
        timeframe="1 day",
        start_date=None,
        end_date=None,
        initial_capital=Decimal("100000"),
        lookback_periods=100
    )
    
    print(f"Run ID: {result['run_id']}")
    print(f"Total P&L: ${result['metrics']['total_pnl']:.2f}")
    print(f"Sharpe Ratio: {result['metrics']['sharpe_ratio']:.2f}")

asyncio.run(run_backtest())
```

## Configuration

Backtester uses settings from `common.config.BacktestSettings`:

- `BT_COMM_PER_SHARE`: Commission per share (default: 0.005)
- `BT_MIN_COMM_PER_ORDER`: Minimum commission per order (default: 1.0)
- `BT_DEFAULT_SLIPPAGE_TICKS`: Slippage in ticks (default: 1)
- `BT_TICK_SIZE_US_EQUITY`: Tick size for US equities (default: 0.01)

## Available Strategies

The backtester includes several pre-built strategies and supports custom strategy development:

### Single-Symbol Strategies

1. **SMA_Crossover** - Simple Moving Average crossover strategy
   - Configurable short/long periods
   - Volume filtering
   - Multiple position sizing methods
   - Example: `--strategy SMA_Crossover --symbols AAPL --params '{"short_period": 10, "long_period": 20}'`

2. **Mean_Reversion** - Mean reversion strategy with Bollinger Bands
   - Trades oversold/overbought conditions
   - Configurable lookback and standard deviation bands
   - Example: `--strategy Mean_Reversion --symbols SPY`

### Multi-Symbol Strategies

3. **Pairs_Trading** - Statistical arbitrage between two correlated stocks
   - Requires exactly 2 symbols
   - Z-score based entry/exit
   - Configurable thresholds and position sizing
   - Tracks both legs independently
   - Example: `--strategy Pairs_Trading --symbols AAPL MSFT --params '{"entry_threshold": 2.0, "position_size": 100}'`

### Creating Custom Strategies

All strategies inherit from `BaseStrategy` and can be easily added to the `strategy_lib` directory. See the strategy documentation for implementation details.

## Performance Metrics

The backtester calculates comprehensive performance metrics:

### Returns and Risk

- **Total P&L**: Net profit/loss in dollars
- **Total Return %**: Percentage return on initial capital
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Max Drawdown %**: Maximum peak-to-trough decline

### Trade Statistics

- **Total Trades**: Number of completed trades
- **Winning Trades**: Number of profitable trades
- **Losing Trades**: Number of losing trades
- **Win Rate**: Percentage of winning trades
- **Avg Win**: Average profit per winning trade
- **Avg Loss**: Average loss per losing trade
- **Profit Factor**: Gross profit / gross loss
- **Largest Win**: Single largest winning trade
- **Largest Loss**: Single largest losing trade
- **Avg Trade Duration**: Average holding period in days

### Costs

- **Total Commission**: Total commission paid
- **Total Slippage**: Total slippage cost

## Simulation Details

### Order Execution

- **Market Orders**: Fill at close price + slippage
- **Slippage**: Applied based on tick size and slippage_ticks setting
- **Commission**: Applied per share with minimum per order
- **Fills**: Assume 100% fill rate (no partial fills currently)

### Position Tracking

- Tracks long and short positions independently
- Calculates realized P&L on position closes
- Tracks unrealized P&L for open positions
- Handles position averaging and partial exits

### Capital Management

- Starts with specified initial capital
- Checks cash availability before buys
- Updates cash on every trade
- Tracks total equity (cash + positions)

## Example Output

### Single-Symbol Strategy (SMA Crossover)

```
================================================================================
Running Backtest: SMA_Crossover
================================================================================
Symbols: AAPL
Timeframe: 1 day
Initial Capital: $100,000.00
================================================================================

Progress: 0/250 bars, Equity: $100,000.00
Progress: 50/250 bars, Equity: $105,234.50
Progress: 100/250 bars, Equity: $112,456.78
Progress: 150/250 bars, Equity: $108,901.23
Progress: 200/250 bars, Equity: $115,678.90

================================================================================
Backtest Results (Run ID: 42)
================================================================================

Performance:
  Total P&L:              $15,678.90
  Total Return:                15.68%
  Sharpe Ratio:                  1.45
  Max Drawdown:                  8.23%

Trades:
  Total Trades:                    12
  Winning Trades:                   8
  Losing Trades:                    4
  Win Rate:                     66.67%

Trade Statistics:
  Avg Win:                   $2,456.78
  Avg Loss:                    -$987.65
  Largest Win:               $4,567.89
  Largest Loss:             -$1,234.56
  Profit Factor:                  2.48
  Avg Duration:                15.3 days

Costs:
  Total Commission:             $60.00
  Total Slippage:              $120.00

================================================================================
```

### Pairs Trading Strategy (Multi-Symbol)

```
================================================================================
Running Backtest: Pairs_Trading
================================================================================
Symbols: AAPL, MSFT
Timeframe: 1 day
Initial Capital: $100,000.00
================================================================================

Progress: 0/250 bars, Equity: $100,000.00
Progress: 50/250 bars, Equity: $100,000.00
Progress: 100/250 bars, Equity: $177,094.00
Progress: 150/250 bars, Equity: $191,067.00
Progress: 200/250 bars, Equity: $101,061.00

Closed trade: BUY 100 AAPL, P&L: $1,418.00
Closed trade: SELL 100 MSFT, P&L: $-1,174.00
Closed trade: BUY 100 AAPL, P&L: $1,156.00
Closed trade: SELL 100 MSFT, P&L: $-143.00

================================================================================
Backtest Results (Run ID: 18)
================================================================================

Performance:
  Total P&L:        $       3,323.00
  Total Return:                3.32%
  Sharpe Ratio:                0.92
  Max Drawdown:               50.94%

Trades:
  Total Trades:                  12
  Winning Trades:                 8
  Losing Trades:                  4
  Win Rate:                   66.67%

Trade Statistics:
  Avg Win:          $         944.38
  Avg Loss:         $      -1,058.00
  Largest Win:      $       1,782.00
  Largest Loss:     $      -2,716.00
  Profit Factor:               1.79
  Avg Duration:                13.3 days

Costs:
  Total Commission: $          12.00
  Total Slippage:   $          12.00

================================================================================
```

Note: In pairs trading, trades are shown individually for each leg. The 12 total trades represent 6 pairs (each pair has 2 legs).

## Testing

### Prerequisites

1. Historical data in `candles` table
2. Strategy registered in `strategy_lib`
3. Database connection configured

### Test Data Setup

```bash
# Request AAPL data (from historical service)
curl -X POST "http://localhost:8003/historical/request?symbol=AAPL&bar_size=1%20day&duration=1%20Y"
```

### Run Test Backtest

```bash
# CLI test
cd /app/src/services/backtester
python main.py cli --strategy SMA_Crossover --symbols AAPL --timeframe "1 day"

# API test
python main.py api --port 8006 &
curl -X POST http://localhost:8006/backtests -H "Content-Type: application/json" -d '{
  "strategy_name": "SMA_Crossover",
  "symbols": ["AAPL"],
  "timeframe": "1 day",
  "initial_capital": 100000.0
}'
```

## Limitations

- Currently only supports market orders (limit/stop orders TODO)
- Assumes 100% fill rate (no partial fills)
- Bar-based simulation (no intrabar price action)
- Simplified slippage model (fixed ticks)
- No overnight/weekend gap modeling
- Single-threaded execution

## Future Enhancements

- [ ] Limit and stop order simulation
- [ ] Partial fill simulation
- [ ] Intrabar price modeling
- [ ] Walk-forward analysis
- [ ] Parameter optimization integration
- [ ] Monte Carlo simulation
- [ ] Multi-strategy portfolio backtesting
- [ ] Custom commission models
- [ ] Benchmark comparison
- [ ] Performance visualization/plotting
- [ ] Export to CSV/Excel

## Integration with Other Services

### Historical Service

Backtester reads data from the `candles` table populated by the historical service:

```python
# Ensure data is loaded before backtesting
# Use historical service API or direct database check
```

### Strategy Service

Backtester uses the same strategy classes as the live strategy service, ensuring consistency between backtesting and live trading.

### Optimizer Service (Future)

The backtester will integrate with the optimizer service for parameter optimization and walk-forward analysis.

## Troubleshooting

### No Data Available

```
Error: No historical data found for symbols: ['AAPL']
```

**Solution**: Request data from historical service first:
```bash
curl -X POST "http://localhost:8003/historical/request?symbol=AAPL&bar_size=1%20day&duration=1%20Y"
```

### Strategy Not Found

```
Error: Strategy 'MyStrategy' not found in registry
```

**Solution**: Ensure strategy is registered in `strategy_lib/__init__.py` and uses the `@strategy` decorator.

### Insufficient Data

```
Warning: Insufficient data for AAPL: 25 bars
```

**Solution**: Strategy requires more bars than available. Either:
- Request more historical data
- Reduce `lookback_periods` parameter
- Adjust strategy period requirements

## API Documentation

Full API documentation available at `/docs` when running in API mode:

```bash
python main.py api --port 8006
# Visit http://localhost:8006/docs
```

