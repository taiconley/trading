"""Parallel execution engine for running backtests in parallel."""

import multiprocessing as mp
from multiprocessing import Pool
import logging
from typing import Dict, Any, List, Callable
from dataclasses import dataclass
import traceback

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result from executing a single backtest task."""
    params: Dict[str, Any]
    success: bool
    result: Any = None
    error: str = None


def run_backtest_task(args: tuple) -> TaskResult:
    """
    Worker function to run a single backtest.
    
    This function runs in a separate process.
    
    Args:
        args: Tuple of (params, strategy_name, symbols, timeframe, lookback, objective, config)
    
    Returns:
        TaskResult with backtest results or error
    """
    params, strategy_name, symbols, timeframe, lookback, objective, config = args
    
    try:
        # Import here to avoid pickling issues with multiprocessing
        import sys
        import os
        import pandas as pd
        from decimal import Decimal
        from datetime import datetime, timedelta
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        
        from services.backtester.engine import BacktestEngine
        from common.config import settings
        from common.db import create_database_engine, close_database_connections
        from common.models import Candle
        from strategy_lib.base import StrategyConfig
        from strategy_lib import get_strategy_registry
        from sqlalchemy.orm import sessionmaker
        
        # Ensure symbols is a list
        if not isinstance(symbols, list):
            symbols = [symbols]
        
        # Create a NEW database engine for THIS worker process
        # This is critical - each process needs its own connection pool
        engine = create_database_engine()
        SessionLocal = sessionmaker(bind=engine)
        
        # Load historical data from database
        bars_data = {}
        session = SessionLocal()
        try:
            for symbol in symbols:
                candles_list = session.query(
                    Candle.ts,
                    Candle.open,
                    Candle.high,
                    Candle.low,
                    Candle.close,
                    Candle.volume
                ).filter(
                    Candle.symbol == symbol,
                    Candle.tf == timeframe
                ).order_by(Candle.ts).all()
                
                if not candles_list:
                    return TaskResult(
                        params=params,
                        success=False,
                        error=f"No historical data found for {symbol} {timeframe}"
                    )
                
                # Convert to DataFrame immediately (while still in session)
                df = pd.DataFrame([
                    {
                        'timestamp': candle.ts,
                        'open': float(candle.open),
                        'high': float(candle.high),
                        'low': float(candle.low),
                        'close': float(candle.close),
                        'volume': candle.volume
                    }
                    for candle in candles_list
                ])
                # Don't set index - keep timestamp as a column
                bars_data[symbol] = df
        finally:
            # Close session and dispose of engine for this worker
            session.close()
            engine.dispose()
        
        # Get strategy class from registry
        registry = get_strategy_registry()
        strategy_class = registry.get_strategy_class(strategy_name)
        if not strategy_class:
            return TaskResult(
                params=params,
                success=False,
                error=f"Strategy '{strategy_name}' not found in registry"
            )
        
        # Create strategy config
        config_data = {
            'strategy_id': f'opt_{strategy_name}',
            'name': strategy_name,
            'symbols': symbols,
            'enabled': True,
            'bar_timeframe': timeframe,
            'lookback_periods': lookback,
            'parameters': params
        }
        config_data.update(params)
        strategy_config = StrategyConfig(**config_data)
        
        # Instantiate strategy
        strategy = strategy_class(strategy_config)
        
        # Create backtest engine
        engine = BacktestEngine(
            strategy=strategy,
            initial_capital=Decimal(str(config.get('initial_capital', 100000.0))),
            commission_per_share=Decimal(str(config.get('commission_per_share', settings.backtest.comm_per_share))),
            min_commission=Decimal(str(config.get('min_commission', settings.backtest.min_comm_per_order))),
            slippage_ticks=config.get('slippage_ticks', settings.backtest.default_slippage_ticks),
            tick_size=Decimal(str(config.get('tick_size', settings.backtest.tick_size_us_equity)))
        )
        
        # Run backtest (run is async but we'll use asyncio to run it)
        import asyncio
        metrics = asyncio.run(engine.run(bars_data, start_date=None, end_date=None))
        
        if not metrics:
            return TaskResult(
                params=params,
                success=False,
                error="Backtest returned no results"
            )
        
        # Extract objective score from BacktestMetrics object
        if objective == 'sharpe_ratio':
            score = float(metrics.sharpe_ratio) if metrics.sharpe_ratio is not None else 0.0
        elif objective == 'total_return':
            score = float(metrics.total_return_pct) if metrics.total_return_pct is not None else 0.0
        elif objective == 'profit_factor':
            score = float(metrics.profit_factor) if metrics.profit_factor is not None else 0.0
        elif objective == 'win_rate':
            score = float(metrics.win_rate) if metrics.win_rate is not None else 0.0
        else:
            # Default to Sharpe ratio
            score = float(metrics.sharpe_ratio) if metrics.sharpe_ratio is not None else 0.0
        
        # Convert metrics to dict for storage
        metrics_dict = {
            'sharpe_ratio': float(metrics.sharpe_ratio) if metrics.sharpe_ratio else None,
            'total_return_pct': float(metrics.total_return_pct) if metrics.total_return_pct else None,
            'max_drawdown_pct': float(metrics.max_drawdown_pct) if metrics.max_drawdown_pct else None,
            'win_rate': float(metrics.win_rate) if metrics.win_rate else None,
            'profit_factor': float(metrics.profit_factor) if metrics.profit_factor else None,
            'total_trades': metrics.total_trades,
            'winning_trades': metrics.winning_trades,
            'losing_trades': metrics.losing_trades,
            'avg_win': float(metrics.avg_win) if metrics.avg_win else None,
            'avg_loss': float(metrics.avg_loss) if metrics.avg_loss else None,
            'largest_win': float(metrics.largest_win) if metrics.largest_win else None,
            'largest_loss': float(metrics.largest_loss) if metrics.largest_loss else None,
            'total_commission': float(metrics.total_commission) if metrics.total_commission else None,
            'total_slippage': float(metrics.total_slippage) if metrics.total_slippage else None
        }
        
        return TaskResult(
            params=params,
            success=True,
            result={
                'score': score,
                'metrics': metrics_dict,
                'backtest_run_id': None  # We don't store individual backtests from optimizer
            }
        )
        
    except Exception as e:
        logger.error(f"Backtest failed for params {params}: {str(e)}")
        return TaskResult(
            params=params,
            success=False,
            error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        )


class ParallelExecutor:
    """
    Parallel execution engine for running multiple backtests.
    
    Uses multiprocessing to run backtests in parallel across multiple CPU cores.
    """
    
    def __init__(self, num_workers: int = None):
        """
        Initialize parallel executor.
        
        Args:
            num_workers: Number of worker processes (None for CPU count)
        """
        if num_workers is None:
            self.num_workers = mp.cpu_count()
        else:
            self.num_workers = min(num_workers, mp.cpu_count())
        
        logger.info(f"Initialized ParallelExecutor with {self.num_workers} workers")
    
    def execute_batch(
        self,
        param_combinations: List[Dict[str, Any]],
        strategy_name: str,
        symbols: List[str],
        timeframe: str,
        lookback: int,
        objective: str,
        config: Dict[str, Any] = None,
        callback: Callable[[TaskResult], None] = None
    ) -> List[TaskResult]:
        """
        Execute a batch of backtests in parallel.
        
        Args:
            param_combinations: List of parameter dictionaries to test
            strategy_name: Strategy to optimize
            symbols: List of symbols for backtest
            timeframe: Timeframe for backtest
            lookback: Lookback period in days
            objective: Objective function ('sharpe_ratio', 'total_return', etc.)
            config: Additional configuration (commission, slippage, etc.)
            callback: Optional callback function called for each completed task
        
        Returns:
            List of TaskResult objects
        """
        if config is None:
            config = {}
        
        if not param_combinations:
            logger.warning("No parameter combinations to execute")
            return []
        
        logger.info(
            f"Executing {len(param_combinations)} backtests in parallel "
            f"with {self.num_workers} workers"
        )
        
        # Prepare tasks
        tasks = [
            (params, strategy_name, symbols, timeframe, lookback, objective, config)
            for params in param_combinations
        ]
        
        results = []
        
        # Execute in parallel using multiprocessing pool
        with Pool(processes=self.num_workers) as pool:
            # Use imap_unordered for better memory efficiency
            for i, result in enumerate(pool.imap_unordered(run_backtest_task, tasks)):
                results.append(result)
                
                # Call callback if provided
                if callback:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Callback failed: {e}")
                
                # Log progress
                if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                    logger.info(f"Completed {i + 1}/{len(tasks)} backtests")
        
        # Count successes and failures
        successes = sum(1 for r in results if r.success)
        failures = len(results) - successes
        
        logger.info(
            f"Batch execution complete: {successes} succeeded, {failures} failed"
        )
        
        return results
    
    def execute_sequential(
        self,
        param_combinations: List[Dict[str, Any]],
        strategy_name: str,
        symbols: List[str],
        timeframe: str,
        lookback: int,
        objective: str,
        config: Dict[str, Any] = None,
        callback: Callable[[TaskResult], None] = None
    ) -> List[TaskResult]:
        """
        Execute backtests sequentially (for debugging or single-core machines).
        
        Args:
            Same as execute_batch
        
        Returns:
            List of TaskResult objects
        """
        if config is None:
            config = {}
        
        logger.info(
            f"Executing {len(param_combinations)} backtests sequentially"
        )
        
        results = []
        for i, params in enumerate(param_combinations):
            task = (params, strategy_name, symbols, timeframe, lookback, objective, config)
            result = run_backtest_task(task)
            results.append(result)
            
            # Call callback if provided
            if callback:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Callback failed: {e}")
            
            # Log progress
            if (i + 1) % 10 == 0 or (i + 1) == len(param_combinations):
                logger.info(f"Completed {i + 1}/{len(param_combinations)} backtests")
        
        return results

