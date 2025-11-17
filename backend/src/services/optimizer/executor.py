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
        args: Tuple of (params, strategy_name, symbols, timeframe, lookback, objective, config, bars_data, start_date, end_date)
              bars_data is optional - if None, data will be loaded from database
              start_date and end_date are optional - if provided, filter data by these dates
    
    Returns:
        TaskResult with backtest results or error
    """
    import logging
    import sys
    
    # Configure logging for this worker process (multiprocessing workers don't inherit logging config)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
        force=True  # Override any existing config
    )
    
    worker_logger = logging.getLogger(f"{__name__}.worker")
    
    # Also use print as fallback (always works, even if logging fails)
    print("=" * 60, flush=True)
    print("WORKER: Starting backtest task", flush=True)
    worker_logger.info("=" * 60)
    worker_logger.info("WORKER: Starting backtest task")
    
    params, strategy_name, symbols, timeframe, lookback, objective, config, bars_data_arg, start_date, end_date = args
    
    try:
        symbol_count = len(symbols) if isinstance(symbols, list) else 1
        msg = f"WORKER: Task params - strategy={strategy_name}, symbols={symbol_count}, timeframe={timeframe}"
        print(msg, flush=True)
        worker_logger.info(msg)
        
        # Import here to avoid pickling issues with multiprocessing
        import os
        import pandas as pd
        from decimal import Decimal
        from datetime import datetime, timedelta
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        
        print("WORKER: Imports complete", flush=True)
        worker_logger.info("WORKER: Imports complete")
        
        from services.backtester.engine import BacktestEngine
        from common.config import settings
        from common.db import create_database_engine, close_database_connections
        from common.models import Candle
        from strategy_lib.base import StrategyConfig
        from strategy_lib import get_strategy_registry
        from sqlalchemy.orm import sessionmaker
        
        print("WORKER: Module imports complete", flush=True)
        worker_logger.info("WORKER: Module imports complete")
        
        # Ensure symbols is a list
        if not isinstance(symbols, list):
            symbols = [symbols]
        
        msg = f"WORKER: Processing {len(symbols)} symbols"
        print(msg, flush=True)
        worker_logger.info(msg)
        
        # Use pre-loaded data if provided, otherwise load from database
        if bars_data_arg is not None:
            print("WORKER: Using pre-loaded data", flush=True)
            worker_logger.info("WORKER: Using pre-loaded data")
            bars_data = bars_data_arg
        else:
            print("WORKER: Loading data from database...", flush=True)
            worker_logger.info("WORKER: Loading data from database...")
            # Create a NEW database engine for THIS worker process
            # This is critical - each process needs its own connection pool
            engine = create_database_engine()
            print("WORKER: Database engine created", flush=True)
            worker_logger.info("WORKER: Database engine created")
            SessionLocal = sessionmaker(bind=engine)
            
            # Load historical data from database
            bars_data = {}
            session = SessionLocal()
            try:
                for i, symbol in enumerate(symbols):
                    msg = f"WORKER: Loading data for symbol {i+1}/{len(symbols)}: {symbol}"
                    print(msg, flush=True)
                    worker_logger.info(msg)
                    
                    # Build query with optional date filtering
                    query = session.query(
                        Candle.ts,
                        Candle.open,
                        Candle.high,
                        Candle.low,
                        Candle.close,
                        Candle.volume
                    ).filter(
                        Candle.symbol == symbol,
                        Candle.tf == timeframe
                    )
                    
                    # Apply date filters if provided
                    if start_date is not None:
                        query = query.filter(Candle.ts >= start_date)
                    if end_date is not None:
                        query = query.filter(Candle.ts <= end_date)
                    
                    candles_list = query.order_by(Candle.ts).all()
                    
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
                    msg = f"WORKER: Loaded {len(df)} bars for {symbol}"
                    print(msg, flush=True)
                    worker_logger.info(msg)
            finally:
                # Close session and dispose of engine for this worker
                session.close()
                engine.dispose()
                print("WORKER: Database connection closed", flush=True)
                worker_logger.info("WORKER: Database connection closed")
        
        msg = f"WORKER: Data loading complete. Total symbols: {len(bars_data)}"
        print(msg, flush=True)
        worker_logger.info(msg)
        
        # Get strategy class from registry
        print("WORKER: Getting strategy from registry...", flush=True)
        worker_logger.info("WORKER: Getting strategy from registry...")
        registry = get_strategy_registry()
        print("WORKER: Strategy registry obtained", flush=True)
        worker_logger.info("WORKER: Strategy registry obtained")
        strategy_class = registry.get_strategy_class(strategy_name)
        if not strategy_class:
            return TaskResult(
                params=params,
                success=False,
                error=f"Strategy '{strategy_name}' not found in registry"
            )
        
        msg = f"WORKER: Strategy class obtained: {strategy_class.__name__}"
        print(msg, flush=True)
        worker_logger.info(msg)
        
        # Handle pair_selection parameter if present (before creating config_data)
        filtered_params = params.copy()  # Work with a copy
        if 'pair_selection' in filtered_params and isinstance(filtered_params['pair_selection'], dict):
            # Get all pairs from config
            all_pairs = []
            if config and 'pairs' in config:
                all_pairs = config['pairs']
            
            # Filter pairs based on selection
            pair_selection = filtered_params['pair_selection']
            filtered_pairs = []
            for pair in all_pairs:
                pair_key = f"{pair[0]}/{pair[1]}"
                # Include pair if selection is True
                if pair_key in pair_selection and pair_selection[pair_key]:
                    filtered_pairs.append(pair)
                elif pair_key not in pair_selection:
                    # If not in selection dict, include by default
                    filtered_pairs.append(pair)
            
            # Remove pair_selection from params so it doesn't get passed to strategy
            filtered_params = {k: v for k, v in filtered_params.items() if k != 'pair_selection'}
        else:
            # Get pairs from config if not using pair_selection
            filtered_pairs = None
            if config and 'pairs' in config:
                filtered_pairs = config['pairs']
        
        # Create strategy config
        config_data = {
            'strategy_id': f'opt_{strategy_name}',
            'name': strategy_name,
            'symbols': symbols,
            'enabled': True,
            'bar_timeframe': timeframe,
            'lookback_periods': lookback,
            'parameters': filtered_params
        }
        # Merge config first (contains pairs, etc.), then params (params override config for optimization)
        # This ensures pairs from config are preserved, but param values override defaults
        if config:
            config_data.update(config)
        # Params come last so optimization parameters override any defaults in config
        config_data.update(filtered_params)
        
        # Set filtered pairs if we have them
        if filtered_pairs is not None:
            config_data['pairs'] = filtered_pairs
        # Ensure pairs is preserved if not already set
        elif 'pairs' not in config_data and config and 'pairs' in config:
            config_data['pairs'] = config['pairs']
        
        print("WORKER: Creating strategy config...", flush=True)
        worker_logger.info("WORKER: Creating strategy config...")
        strategy_config = StrategyConfig(**config_data)
        print("WORKER: Strategy config created", flush=True)
        worker_logger.info("WORKER: Strategy config created")
        
        # Instantiate strategy
        print("WORKER: Instantiating strategy...", flush=True)
        worker_logger.info("WORKER: Instantiating strategy...")
        strategy = strategy_class(strategy_config)
        print("WORKER: Strategy instantiated", flush=True)
        worker_logger.info("WORKER: Strategy instantiated")
        
        # Create backtest engine
        print("WORKER: Creating backtest engine...", flush=True)
        worker_logger.info("WORKER: Creating backtest engine...")
        engine = BacktestEngine(
            strategy=strategy,
            initial_capital=Decimal(str(config.get('initial_capital', 100000.0))),
            commission_per_share=Decimal(str(config.get('commission_per_share', settings.backtest.comm_per_share))),
            min_commission=Decimal(str(config.get('min_commission', settings.backtest.min_comm_per_order))),
            slippage_ticks=config.get('slippage_ticks', settings.backtest.default_slippage_ticks),
            tick_size=Decimal(str(config.get('tick_size', settings.backtest.tick_size_us_equity)))
        )
        print("WORKER: Backtest engine created", flush=True)
        worker_logger.info("WORKER: Backtest engine created")
        
        # Run backtest (run is async but we'll use asyncio to run it)
        print("WORKER: Starting backtest execution...", flush=True)
        worker_logger.info("WORKER: Starting backtest execution...")
        import asyncio
        metrics = asyncio.run(engine.run(bars_data, start_date=None, end_date=None))
        print("WORKER: Backtest execution complete", flush=True)
        worker_logger.info("WORKER: Backtest execution complete")
        
        if not metrics:
            return TaskResult(
                params=params,
                success=False,
                error="Backtest returned no results"
            )
        
        # Extract objective score from BacktestMetrics object
        if objective == 'sharpe_ratio':
            score = float(metrics.sharpe_ratio) if metrics.sharpe_ratio is not None else 0.0
        elif objective in ('sortino_ratio', 'sortino'):
            score = float(metrics.sortino_ratio) if metrics.sortino_ratio is not None else 0.0
        elif objective == 'total_return':
            score = float(metrics.total_return_pct) if metrics.total_return_pct is not None else 0.0
        elif objective == 'profit_factor':
            score = float(metrics.profit_factor) if metrics.profit_factor is not None else 0.0
        elif objective == 'win_rate':
            score = float(metrics.win_rate) if metrics.win_rate is not None else 0.0
        elif objective in ('volatility', 'annualized_volatility'):
            volatility = float(metrics.annualized_volatility_pct) if metrics.annualized_volatility_pct is not None else None
            score = -volatility if volatility is not None else 0.0
        elif objective in ('value_at_risk', 'var', 'var_95'):
            var_value = float(metrics.value_at_risk_pct) if metrics.value_at_risk_pct is not None else None
            score = -var_value if var_value is not None else 0.0
        elif objective in ('avg_holding_time', 'avg_holding_period'):
            holding_hours = float(metrics.avg_holding_period_hours) if metrics.avg_holding_period_hours is not None else None
            score = holding_hours if holding_hours is not None else 0.0
        else:
            # Default to Sharpe ratio
            score = float(metrics.sharpe_ratio) if metrics.sharpe_ratio is not None else 0.0
        
        # Convert metrics to dict for storage
        metrics_dict = {
            'sharpe_ratio': float(metrics.sharpe_ratio) if metrics.sharpe_ratio else None,
            'sortino_ratio': float(metrics.sortino_ratio) if metrics.sortino_ratio else None,
            'total_return_pct': float(metrics.total_return_pct) if metrics.total_return_pct else None,
            'max_drawdown_pct': float(metrics.max_drawdown_pct) if metrics.max_drawdown_pct else None,
            'annualized_volatility_pct': float(metrics.annualized_volatility_pct) if metrics.annualized_volatility_pct is not None else None,
            'value_at_risk_pct': float(metrics.value_at_risk_pct) if metrics.value_at_risk_pct is not None else None,
            'win_rate': float(metrics.win_rate) if metrics.win_rate else None,
            'profit_factor': float(metrics.profit_factor) if metrics.profit_factor else None,
            'total_trades': metrics.total_trades,
            'winning_trades': metrics.winning_trades,
            'losing_trades': metrics.losing_trades,
            'avg_win': float(metrics.avg_win) if metrics.avg_win else None,
            'avg_loss': float(metrics.avg_loss) if metrics.avg_loss else None,
            'largest_win': float(metrics.largest_win) if metrics.largest_win else None,
            'largest_loss': float(metrics.largest_loss) if metrics.largest_loss else None,
            'avg_trade_duration_days': float(metrics.avg_trade_duration_days),
            'avg_holding_period_hours': float(metrics.avg_holding_period_hours),
            'total_commission': float(metrics.total_commission) if metrics.total_commission else None,
            'total_slippage': float(metrics.total_slippage) if metrics.total_slippage else None
        }
        
        msg = f"WORKER: Backtest completed successfully. Score: {score}"
        print(msg, flush=True)
        print("=" * 60, flush=True)
        worker_logger.info(msg)
        worker_logger.info("=" * 60)
        
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
        error_msg = f"WORKER: Backtest failed for params {params}: {str(e)}"
        print(error_msg, flush=True)
        print(f"WORKER: Traceback: {traceback.format_exc()}", flush=True)
        print("=" * 60, flush=True)
        worker_logger.error(error_msg)
        worker_logger.error(f"WORKER: Traceback: {traceback.format_exc()}")
        worker_logger.info("=" * 60)
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
        callback: Callable[[TaskResult], None] = None,
        bars_data: Dict[str, Any] = None,
        start_date: Any = None,
        end_date: Any = None
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
            bars_data: Optional pre-loaded bars data (for validation methods)
            start_date: Optional start date for filtering data
            end_date: Optional end date for filtering data
        
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
            (params, strategy_name, symbols, timeframe, lookback, objective, config, bars_data, start_date, end_date)
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
        callback: Callable[[TaskResult], None] = None,
        bars_data: Dict[str, Any] = None,
        start_date: Any = None,
        end_date: Any = None
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
            task = (params, strategy_name, symbols, timeframe, lookback, objective, config, bars_data, start_date, end_date)
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
