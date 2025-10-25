"""
Backtester Service - On-demand backtesting service with CLI and REST API

This service provides both a command-line interface and REST API for running
backtests on historical data. It supports all strategies from the strategy_lib
and stores results in the database for analysis.
"""

import asyncio
import sys
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any
import pandas as pd
import argparse
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from common.config import get_settings
from common.db import get_db_session, execute_with_retry
from common.models import Candle, BacktestRun, BacktestTrade, Symbol
from common.logging import configure_service_logging
from strategy_lib import get_strategy_registry, StrategyConfig, create_strategy_from_db_config
from common.schemas import OrderSide

# Import engine - handle both relative and absolute imports
try:
    from .engine import BacktestEngine, BacktestMetrics
except ImportError:
    from services.backtester.engine import BacktestEngine, BacktestMetrics


# Pydantic models for API
class BacktestRequest(BaseModel):
    """Request to run a backtest."""
    strategy_name: str = Field(..., description="Name of strategy to backtest")
    strategy_params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    symbols: List[str] = Field(..., description="List of symbols to trade")
    timeframe: str = Field(default="1 day", description="Timeframe for bars")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(default=100000.0, description="Starting capital")
    lookback_periods: int = Field(default=100, description="Number of bars to provide to strategy")


class BacktestResponse(BaseModel):
    """Response from backtest execution."""
    run_id: int
    status: str
    metrics: Dict[str, Any]
    trades_count: int
    message: str


class BacktestStatus(BaseModel):
    """Status of a backtest run."""
    run_id: int
    strategy_name: str
    status: str
    start_date: Optional[str]
    end_date: Optional[str]
    total_pnl: Optional[float]
    sharpe_ratio: Optional[float]
    max_drawdown_pct: Optional[float]
    total_trades: int
    created_at: str


class BacktesterService:
    """Backtester service with CLI and REST API."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = configure_service_logging("backtester")
        self.app = FastAPI(title="Backtester Service", description="On-demand backtesting service")
        self.registry = get_strategy_registry()
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/healthz")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "backtester",
                "available_strategies": self.registry.list_strategies()
            }
        
        @self.app.post("/backtests", response_model=BacktestResponse)
        async def run_backtest(request: BacktestRequest):
            """Run a backtest."""
            try:
                self.logger.info(f"Running backtest: {request.strategy_name} on {request.symbols}")
                
                # Parse dates
                start_date = datetime.fromisoformat(request.start_date) if request.start_date else None
                end_date = datetime.fromisoformat(request.end_date) if request.end_date else None
                
                # Run backtest
                result = await self._run_backtest(
                    strategy_name=request.strategy_name,
                    strategy_params=request.strategy_params,
                    symbols=request.symbols,
                    timeframe=request.timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=Decimal(str(request.initial_capital)),
                    lookback_periods=request.lookback_periods
                )
                
                return BacktestResponse(
                    run_id=result['run_id'],
                    status="completed",
                    metrics=result['metrics'],
                    trades_count=result['trades_count'],
                    message=f"Backtest completed successfully"
                )
                
            except Exception as e:
                self.logger.error(f"Backtest failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/backtests/{run_id}", response_model=BacktestStatus)
        async def get_backtest(run_id: int):
            """Get backtest run details."""
            try:
                def _get_run(session):
                    return session.query(BacktestRun).filter(BacktestRun.id == run_id).first()
                
                run = execute_with_retry(_get_run)
                
                if not run:
                    raise HTTPException(status_code=404, detail=f"Backtest run {run_id} not found")
                
                return BacktestStatus(
                    run_id=run.id,
                    strategy_name=run.strategy_name,
                    status="completed",
                    start_date=run.start_ts.isoformat() if run.start_ts else None,
                    end_date=run.end_ts.isoformat() if run.end_ts else None,
                    total_pnl=float(run.pnl) if run.pnl else None,
                    sharpe_ratio=float(run.sharpe) if run.sharpe else None,
                    max_drawdown_pct=float(run.maxdd) if run.maxdd else None,
                    total_trades=run.trades,
                    created_at=run.created_at.isoformat()
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to get backtest: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/backtests")
        async def list_backtests(limit: int = 50):
            """List recent backtest runs."""
            try:
                def _list_runs(session):
                    runs = session.query(BacktestRun).order_by(
                        BacktestRun.created_at.desc()
                    ).limit(limit).all()
                    
                    return [
                        {
                            'run_id': run.id,
                            'strategy_name': run.strategy_name,
                            'start_date': run.start_ts.isoformat() if run.start_ts else None,
                            'end_date': run.end_ts.isoformat() if run.end_ts else None,
                            'total_pnl': float(run.pnl) if run.pnl else None,
                            'sharpe_ratio': float(run.sharpe) if run.sharpe else None,
                            'total_trades': run.trades,
                            'created_at': run.created_at.isoformat()
                        }
                        for run in runs
                    ]
                
                runs = execute_with_retry(_list_runs)
                return {"backtests": runs, "count": len(runs)}
                
            except Exception as e:
                self.logger.error(f"Failed to list backtests: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/backtests/{run_id}/trades")
        async def get_backtest_trades(run_id: int):
            """Get trades from a backtest run."""
            try:
                def _get_trades(session):
                    trades = session.query(BacktestTrade).filter(
                        BacktestTrade.run_id == run_id
                    ).order_by(BacktestTrade.entry_ts).all()
                    
                    return [
                        {
                            'id': trade.id,
                            'symbol': trade.symbol,
                            'side': trade.side,
                            'quantity': float(trade.qty) if trade.qty else None,
                            'entry_time': trade.entry_ts.isoformat() if trade.entry_ts else None,
                            'entry_price': float(trade.entry_px) if trade.entry_px else None,
                            'exit_time': trade.exit_ts.isoformat() if trade.exit_ts else None,
                            'exit_price': float(trade.exit_px) if trade.exit_px else None,
                            'pnl': float(trade.pnl) if trade.pnl else None
                        }
                        for trade in trades
                    ]
                
                trades = execute_with_retry(_get_trades)
                return {"trades": trades, "count": len(trades)}
                
            except Exception as e:
                self.logger.error(f"Failed to get trades: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _run_backtest(
        self,
        strategy_name: str,
        strategy_params: Dict[str, Any],
        symbols: List[str],
        timeframe: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        initial_capital: Decimal,
        lookback_periods: int
    ) -> Dict[str, Any]:
        """Execute a backtest and store results."""
        
        # Load historical data
        bars_data = await self._load_historical_data(
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if not bars_data:
            raise ValueError(f"No historical data found for symbols: {symbols}")
        
        # Create strategy instance
        strategy = self._create_strategy(
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            symbols=symbols,
            timeframe=timeframe,
            lookback_periods=lookback_periods
        )
        
        # Create backtest engine
        engine = BacktestEngine(
            strategy=strategy,
            initial_capital=initial_capital,
            commission_per_share=Decimal(str(self.settings.backtest.comm_per_share)),
            min_commission=Decimal(str(self.settings.backtest.min_comm_per_order)),
            slippage_ticks=self.settings.backtest.default_slippage_ticks,
            tick_size=Decimal(str(self.settings.backtest.tick_size_us_equity))
        )
        
        # Run backtest
        metrics = await engine.run(bars_data, start_date, end_date)
        
        # Store results in database
        run_id = self._store_backtest_results(
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            metrics=metrics,
            trades=engine.trades
        )
        
        return {
            'run_id': run_id,
            'metrics': self._metrics_to_dict(metrics),
            'trades_count': len(engine.trades)
        }
    
    async def _load_historical_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load historical data from database."""
        
        def _load_data(session):
            data = {}
            
            for symbol in symbols:
                # Query candles for this symbol and timeframe
                query = session.query(Candle).filter(
                    Candle.symbol == symbol,
                    Candle.tf == timeframe
                )
                if start_date:
                    query = query.filter(Candle.ts >= start_date)
                if end_date:
                    query = query.filter(Candle.ts <= end_date)
                candles = query.order_by(Candle.ts).all()
                
                if not candles:
                    self.logger.warning(f"No data found for {symbol} {timeframe}")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame([
                    {
                        'timestamp': candle.ts,
                        'open': float(candle.open),
                        'high': float(candle.high),
                        'low': float(candle.low),
                        'close': float(candle.close),
                        'volume': candle.volume
                    }
                    for candle in candles
                ])
                
                data[symbol] = df
                if start_date or end_date:
                    self.logger.info(
                        f"Loaded {len(df)} bars for {symbol} between "
                        f"{start_date.isoformat() if start_date else 'start'} and "
                        f"{end_date.isoformat() if end_date else 'end'}"
                    )
                else:
                    self.logger.info(f"Loaded {len(df)} bars for {symbol}")
            
            return data
        
        return execute_with_retry(_load_data)
    
    def _create_strategy(
        self,
        strategy_name: str,
        strategy_params: Dict[str, Any],
        symbols: List[str],
        timeframe: str,
        lookback_periods: int
    ):
        """Create strategy instance."""
        
        # Get strategy class from registry
        strategy_class = self.registry.get_strategy_class(strategy_name)
        if not strategy_class:
            raise ValueError(f"Strategy '{strategy_name}' not found in registry")
        
        # Build config
        config_data = {
            'strategy_id': f'backtest_{strategy_name}',
            'name': strategy_name,
            'symbols': symbols,
            'enabled': True,
            'bar_timeframe': timeframe,
            'lookback_periods': lookback_periods,
            'parameters': strategy_params
        }
        
        # Merge strategy-specific params into top level
        config_data.update(strategy_params)
        
        # Create strategy config
        config = StrategyConfig(**config_data)
        
        # Instantiate strategy
        strategy = strategy_class(config)
        
        self.logger.info(f"Created strategy: {strategy_name} with params: {strategy_params}")
        
        return strategy
    
    def _store_backtest_results(
        self,
        strategy_name: str,
        strategy_params: Dict[str, Any],
        metrics: BacktestMetrics,
        trades: List
    ) -> int:
        """Store backtest results in database."""
        
        def _store_results(session):
            # Create backtest run record with comprehensive metrics
            run = BacktestRun(
                strategy_name=strategy_name,
                params_json=strategy_params,
                start_ts=metrics.start_date,
                end_ts=metrics.end_date,
                
                # Core performance metrics
                pnl=metrics.total_pnl,
                total_return_pct=metrics.total_return_pct,
                sharpe=metrics.sharpe_ratio,
                sortino_ratio=metrics.sortino_ratio,
                annualized_volatility_pct=metrics.annualized_volatility_pct,
                value_at_risk_pct=metrics.value_at_risk_pct,
                maxdd=metrics.max_drawdown_pct,
                max_drawdown_duration_days=metrics.max_drawdown_duration_days,
                
                # Trade statistics
                trades=metrics.total_trades,
                winning_trades=metrics.winning_trades,
                losing_trades=metrics.losing_trades,
                win_rate=metrics.win_rate,
                profit_factor=metrics.profit_factor,
                
                # Trade performance
                avg_win=metrics.avg_win,
                avg_loss=metrics.avg_loss,
                largest_win=metrics.largest_win,
                largest_loss=metrics.largest_loss,
                
                # Trade timing
                avg_trade_duration_days=metrics.avg_trade_duration_days,
                avg_holding_period_hours=metrics.avg_holding_period_hours,
                
                # Costs
                total_commission=metrics.total_commission,
                total_slippage=metrics.total_slippage,
                
                # Additional metadata
                total_days=metrics.total_days
            )
            
            session.add(run)
            session.flush()  # Get run.id
            
            # Store individual trades
            for trade in trades:
                bt_trade = BacktestTrade(
                    run_id=run.id,
                    symbol=trade.symbol,
                    side=trade.side.value,
                    qty=Decimal(str(trade.quantity)),
                    entry_ts=trade.entry_time,
                    entry_px=trade.entry_price,
                    exit_ts=trade.exit_time,
                    exit_px=trade.exit_price,
                    pnl=trade.net_pnl
                )
                session.add(bt_trade)
            
            session.commit()
            
            self.logger.info(f"Stored backtest run {run.id} with {len(trades)} trades")
            
            return run.id
        
        return execute_with_retry(_store_results)
    
    def _metrics_to_dict(self, metrics: BacktestMetrics) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_pnl': float(metrics.total_pnl),
            'total_return_pct': float(metrics.total_return_pct),
            'sharpe_ratio': float(metrics.sharpe_ratio) if metrics.sharpe_ratio else None,
            'sortino_ratio': float(metrics.sortino_ratio) if metrics.sortino_ratio else None,
            'annualized_volatility_pct': float(metrics.annualized_volatility_pct) if metrics.annualized_volatility_pct is not None else None,
            'value_at_risk_pct': float(metrics.value_at_risk_pct) if metrics.value_at_risk_pct is not None else None,
            'max_drawdown_pct': float(metrics.max_drawdown_pct),
            'total_trades': metrics.total_trades,
            'winning_trades': metrics.winning_trades,
            'losing_trades': metrics.losing_trades,
            'win_rate': float(metrics.win_rate),
            'avg_win': float(metrics.avg_win),
            'avg_loss': float(metrics.avg_loss),
            'profit_factor': float(metrics.profit_factor) if metrics.profit_factor else None,
            'largest_win': float(metrics.largest_win),
            'largest_loss': float(metrics.largest_loss),
            'avg_trade_duration_days': float(metrics.avg_trade_duration_days),
            'avg_holding_period_hours': float(metrics.avg_holding_period_hours),
            'total_commission': float(metrics.total_commission),
            'total_slippage': float(metrics.total_slippage),
            'start_date': metrics.start_date.isoformat() if metrics.start_date else None,
            'end_date': metrics.end_date.isoformat() if metrics.end_date else None,
            'total_days': metrics.total_days
        }


# CLI Interface
async def cli_backtest(args):
    """Run backtest from command line."""
    service = BacktesterService()
    
    # Parse dates
    start_date = datetime.fromisoformat(args.start_date) if args.start_date else None
    end_date = datetime.fromisoformat(args.end_date) if args.end_date else None
    
    # Parse strategy parameters
    strategy_params = {}
    if args.params:
        try:
            strategy_params = json.loads(args.params)
        except json.JSONDecodeError as e:
            print(f"Error parsing strategy parameters: {e}")
            sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Running Backtest: {args.strategy}")
    print(f"{'='*80}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    if start_date:
        print(f"Start Date: {start_date.date()}")
    if end_date:
        print(f"End Date: {end_date.date()}")
    print(f"{'='*80}\n")
    
    try:
        result = await service._run_backtest(
            strategy_name=args.strategy,
            strategy_params=strategy_params,
            symbols=args.symbols,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=Decimal(str(args.initial_capital)),
            lookback_periods=args.lookback
        )
        
        # Print results
        metrics = result['metrics']
        print(f"\n{'='*80}")
        print(f"Backtest Results (Run ID: {result['run_id']})")
        print(f"{'='*80}")
        print(f"\nPerformance:")
        print(f"  Total P&L:        ${metrics['total_pnl']:>15,.2f}")
        print(f"  Total Return:     {metrics['total_return_pct']:>15,.2f}%")
        print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:>15.2f}" if metrics['sharpe_ratio'] else "  Sharpe Ratio:     N/A")
        print(
            f"  Sortino Ratio:    {metrics['sortino_ratio']:>15.2f}"
            if metrics.get('sortino_ratio') is not None
            else "  Sortino Ratio:    N/A"
        )
        if metrics.get('annualized_volatility_pct') is not None:
            print(f"  Volatility (ann): {metrics['annualized_volatility_pct']:>15,.2f}%")
        else:
            print(f"  Volatility (ann): {'N/A':>15}")
        if metrics.get('value_at_risk_pct') is not None:
            print(f"  Value at Risk 95%: {metrics['value_at_risk_pct']:>14,.2f}%")
        else:
            print(f"  Value at Risk 95%: {'N/A':>14}")
        print(f"  Max Drawdown:     {metrics['max_drawdown_pct']:>15,.2f}%")
        
        print(f"\nTrades:")
        print(f"  Total Trades:     {metrics['total_trades']:>15,}")
        print(f"  Winning Trades:   {metrics['winning_trades']:>15,}")
        print(f"  Losing Trades:    {metrics['losing_trades']:>15,}")
        print(f"  Win Rate:         {metrics['win_rate']:>15,.2f}%")
        
        if metrics['total_trades'] > 0:
            print(f"\nTrade Statistics:")
            print(f"  Avg Win:          ${metrics['avg_win']:>15,.2f}")
            print(f"  Avg Loss:         ${metrics['avg_loss']:>15,.2f}")
            print(f"  Largest Win:      ${metrics['largest_win']:>15,.2f}")
            print(f"  Largest Loss:     ${metrics['largest_loss']:>15,.2f}")
            if metrics['profit_factor']:
                print(f"  Profit Factor:    {metrics['profit_factor']:>15.2f}")
            print(f"  Avg Duration:     {metrics['avg_trade_duration_days']:>15,.1f} days")
            print(f"  Avg Holding Time: {metrics['avg_holding_period_hours']:>15,.2f} hrs")
        
        print(f"\nCosts:")
        print(f"  Total Commission: ${metrics['total_commission']:>15,.2f}")
        print(f"  Total Slippage:   ${metrics['total_slippage']:>15,.2f}")
        
        print(f"\n{'='*80}\n")
        
        return 0
        
    except Exception as e:
        print(f"\nBacktest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backtesting Service - Run backtests via CLI or API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run via CLI
  python main.py cli --strategy SMA_Crossover --symbols AAPL --timeframe "1 day"
  
  # Run via CLI with custom parameters
  python main.py cli --strategy SMA_Crossover --symbols AAPL \\
    --params '{"short_period": 10, "long_period": 30}'
  
  # Start REST API server
  python main.py api --port 8006
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Run mode')
    
    # CLI mode
    cli_parser = subparsers.add_parser('cli', help='Run backtest from command line')
    cli_parser.add_argument('--strategy', required=True, help='Strategy name')
    cli_parser.add_argument('--symbols', nargs='+', required=True, help='Symbols to trade')
    cli_parser.add_argument('--timeframe', default='1 day', help='Timeframe (default: 1 day)')
    cli_parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    cli_parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    cli_parser.add_argument('--initial-capital', type=float, default=100000.0, help='Initial capital (default: 100000)')
    cli_parser.add_argument('--lookback', type=int, default=100, help='Lookback periods (default: 100)')
    cli_parser.add_argument('--params', help='Strategy parameters as JSON')
    
    # API mode
    api_parser = subparsers.add_parser('api', help='Start REST API server')
    api_parser.add_argument('--host', default='0.0.0.0', help='Host (default: 0.0.0.0)')
    api_parser.add_argument('--port', type=int, default=8006, help='Port (default: 8006)')
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return 1
    
    if args.mode == 'cli':
        # Run CLI backtest
        exit_code = asyncio.run(cli_backtest(args))
        return exit_code
    
    elif args.mode == 'api':
        # Start API server
        service = BacktesterService()
        
        print(f"\n{'='*80}")
        print(f"Starting Backtester API Server")
        print(f"{'='*80}")
        print(f"Host: {args.host}")
        print(f"Port: {args.port}")
        print(f"Docs: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/docs")
        print(f"{'='*80}\n")
        
        config = uvicorn.Config(
            service.app,
            host=args.host,
            port=args.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        asyncio.run(server.serve())
        
        return 0


if __name__ == "__main__":
    sys.exit(main())
