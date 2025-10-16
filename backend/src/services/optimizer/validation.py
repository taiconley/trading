"""
Validation methods for parameter optimization.

This module provides various validation techniques to assess
parameter robustness and prevent overfitting:

1. Walk-forward Analysis: Rolling optimization windows
2. Out-of-sample Testing: Train/test split validation
3. Cross-validation: K-fold time series validation
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from decimal import Decimal
import pandas as pd

from ...common.models import StrategyConfig
from ...common.database import get_db
from .algorithms import BaseOptimizer, ParameterSpace
from .executor import ParallelExecutor

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window."""
    window_id: int
    in_sample_start: datetime
    in_sample_end: datetime
    out_sample_start: datetime
    out_sample_end: datetime
    best_params: Optional[Dict[str, Any]] = None
    in_sample_score: Optional[float] = None
    out_sample_score: Optional[float] = None


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""
    in_sample_days: int  # Number of days for optimization
    out_sample_days: int  # Number of days for validation
    step_days: int  # Step size for rolling windows
    anchored: bool = False  # If True, start date is fixed
    min_trades_required: int = 1  # Minimum trades needed for valid window


class WalkForwardAnalysis:
    """
    Walk-forward analysis for parameter optimization.
    
    Splits historical data into rolling windows with in-sample (training)
    and out-of-sample (testing) periods. Optimizes parameters on in-sample
    data and validates on out-of-sample data to assess robustness.
    
    Example:
        Total data: 365 days
        In-sample: 180 days
        Out-sample: 60 days
        Step: 60 days
        
        Window 1: Train[Day 1-180],   Test[Day 181-240]
        Window 2: Train[Day 61-240],  Test[Day 241-300]
        Window 3: Train[Day 121-300], Test[Day 301-360]
    """
    
    def __init__(
        self,
        config: WalkForwardConfig,
        strategy_name: str,
        symbols: List[str],
        timeframe: str,
        param_space: ParameterSpace,
        algorithm: str = 'grid_search',
        objective: str = 'sharpe_ratio',
        constraints: Optional[List[str]] = None,
        lookback: int = 100,
        num_workers: int = 4,
        random_seed: Optional[int] = None
    ):
        """
        Initialize walk-forward analysis.
        
        Args:
            config: Walk-forward configuration
            strategy_name: Name of strategy to optimize
            symbols: List of symbols to trade
            timeframe: Bar timeframe
            param_space: Parameter space definition
            algorithm: Optimization algorithm
            objective: Objective function to maximize
            constraints: Parameter constraints
            lookback: Lookback period for strategy
            num_workers: Number of parallel workers
            random_seed: Random seed for reproducibility
        """
        self.config = config
        self.strategy_name = strategy_name
        self.symbols = symbols
        self.timeframe = timeframe
        self.param_space = param_space
        self.algorithm = algorithm
        self.objective = objective
        self.constraints = constraints or []
        self.lookback = lookback
        self.num_workers = num_workers
        self.random_seed = random_seed
        
        self.windows: List[WalkForwardWindow] = []
        self.results: Dict[str, Any] = {}
    
    def create_windows(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[WalkForwardWindow]:
        """
        Create walk-forward windows from date range.
        
        Args:
            start_date: Start of historical data
            end_date: End of historical data
            
        Returns:
            List of walk-forward windows
        """
        windows = []
        window_id = 1
        
        # Calculate first window
        in_start = start_date
        in_end = start_date + timedelta(days=self.config.in_sample_days)
        out_start = in_end + timedelta(days=1)
        out_end = out_start + timedelta(days=self.config.out_sample_days - 1)
        
        while out_end <= end_date:
            window = WalkForwardWindow(
                window_id=window_id,
                in_sample_start=in_start,
                in_sample_end=in_end,
                out_sample_start=out_start,
                out_sample_end=out_end
            )
            windows.append(window)
            
            # Move to next window
            if self.config.anchored:
                # Anchored: keep start date, extend end date
                in_end = in_end + timedelta(days=self.config.step_days)
                out_start = in_end + timedelta(days=1)
                out_end = out_start + timedelta(days=self.config.out_sample_days - 1)
            else:
                # Rolling: move both start and end dates
                in_start = in_start + timedelta(days=self.config.step_days)
                in_end = in_start + timedelta(days=self.config.in_sample_days)
                out_start = in_end + timedelta(days=1)
                out_end = out_start + timedelta(days=self.config.out_sample_days - 1)
            
            window_id += 1
        
        logger.info(f"Created {len(windows)} walk-forward windows")
        self.windows = windows
        return windows
    
    async def run_window(
        self,
        window: WalkForwardWindow,
        bars_data: Dict[str, pd.DataFrame]
    ) -> WalkForwardWindow:
        """
        Run optimization and validation for a single window.
        
        Args:
            window: Walk-forward window to process
            bars_data: Historical bars data for all symbols
            
        Returns:
            Window with results populated
        """
        logger.info(
            f"Processing window {window.window_id}: "
            f"In-sample [{window.in_sample_start.date()} to {window.in_sample_end.date()}], "
            f"Out-sample [{window.out_sample_start.date()} to {window.out_sample_end.date()}]"
        )
        
        # Filter data for in-sample period
        in_sample_data = {}
        for symbol, df in bars_data.items():
            mask = (df['timestamp'] >= window.in_sample_start) & (df['timestamp'] <= window.in_sample_end)
            in_sample_data[symbol] = df[mask].copy()
        
        # Check if we have enough data
        min_bars = max([len(df) for df in in_sample_data.values()])
        if min_bars < self.lookback:
            logger.warning(f"Window {window.window_id}: Insufficient in-sample data ({min_bars} bars)")
            return window
        
        # Run optimization on in-sample data
        from .engine import OptimizationEngine
        
        optimizer = OptimizationEngine(
            strategy_name=self.strategy_name,
            symbols=self.symbols,
            timeframe=self.timeframe,
            lookback=self.lookback,
            param_ranges=self.param_space.to_dict(),
            algorithm=self.algorithm,
            objective=self.objective,
            constraints=self.constraints,
            max_iterations=None,  # Use all combinations for walk-forward
            num_workers=self.num_workers,
            random_seed=self.random_seed
        )
        
        # Run in-sample optimization
        with get_db() as db:
            best_result = await optimizer._run_optimization_direct(
                db=db,
                bars_data=in_sample_data,
                window_id=f"wf{window.window_id}_in"
            )
        
        if not best_result:
            logger.warning(f"Window {window.window_id}: No valid optimization results")
            return window
        
        window.best_params = best_result.params
        window.in_sample_score = best_result.score
        
        logger.info(
            f"Window {window.window_id} in-sample: "
            f"Best params={best_result.params}, Score={best_result.score:.4f}"
        )
        
        # Validate on out-of-sample data
        out_sample_data = {}
        for symbol, df in bars_data.items():
            mask = (df['timestamp'] >= window.out_sample_start) & (df['timestamp'] <= window.out_sample_end)
            out_sample_data[symbol] = df[mask].copy()
        
        # Check if we have enough out-of-sample data
        min_bars_out = max([len(df) for df in out_sample_data.values()])
        if min_bars_out < self.lookback:
            logger.warning(f"Window {window.window_id}: Insufficient out-of-sample data ({min_bars_out} bars)")
            return window
        
        # Run backtest with best parameters on out-of-sample data
        executor = ParallelExecutor(num_workers=1)  # Single backtest, no parallelization needed
        
        try:
            result = await executor.evaluate_parameters(
                strategy_name=self.strategy_name,
                params=window.best_params,
                symbols=self.symbols,
                timeframe=self.timeframe,
                bars_data=out_sample_data,
                lookback=self.lookback,
                objective=self.objective,
                config={}
            )
            
            window.out_sample_score = result['score']
            
            logger.info(
                f"Window {window.window_id} out-of-sample: "
                f"Score={result['score']:.4f}, Trades={result['total_trades']}"
            )
            
            # Check minimum trades requirement
            if result['total_trades'] < self.config.min_trades_required:
                logger.warning(
                    f"Window {window.window_id}: Insufficient trades "
                    f"({result['total_trades']} < {self.config.min_trades_required})"
                )
        
        except Exception as e:
            logger.error(f"Window {window.window_id} out-of-sample validation failed: {e}")
        
        return window
    
    async def run(
        self,
        start_date: datetime,
        end_date: datetime,
        bars_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Run walk-forward analysis across all windows.
        
        Args:
            start_date: Start of historical data
            end_date: End of historical data
            bars_data: Historical bars data for all symbols
            
        Returns:
            Analysis results with metrics
        """
        logger.info(f"Starting walk-forward analysis: {start_date.date()} to {end_date.date()}")
        
        # Create windows
        windows = self.create_windows(start_date, end_date)
        
        if not windows:
            raise ValueError("No walk-forward windows could be created with given configuration")
        
        # Process each window
        for window in windows:
            await self.run_window(window, bars_data)
        
        # Calculate aggregate metrics
        valid_windows = [w for w in windows if w.out_sample_score is not None]
        
        if not valid_windows:
            logger.warning("No valid walk-forward windows with out-of-sample results")
            self.results = {
                'total_windows': len(windows),
                'valid_windows': 0,
                'windows': windows
            }
            return self.results
        
        in_sample_scores = [w.in_sample_score for w in valid_windows if w.in_sample_score is not None]
        out_sample_scores = [w.out_sample_score for w in valid_windows]
        
        self.results = {
            'total_windows': len(windows),
            'valid_windows': len(valid_windows),
            'avg_in_sample_score': sum(in_sample_scores) / len(in_sample_scores) if in_sample_scores else 0,
            'avg_out_sample_score': sum(out_sample_scores) / len(out_sample_scores),
            'score_degradation': (
                (sum(in_sample_scores) / len(in_sample_scores) - sum(out_sample_scores) / len(out_sample_scores))
                if in_sample_scores else 0
            ),
            'score_stability': self._calculate_stability(out_sample_scores),
            'windows': windows
        }
        
        logger.info(
            f"Walk-forward analysis complete: "
            f"{len(valid_windows)}/{len(windows)} valid windows, "
            f"Avg out-sample score={self.results['avg_out_sample_score']:.4f}, "
            f"Degradation={self.results['score_degradation']:.4f}"
        )
        
        return self.results
    
    def _calculate_stability(self, scores: List[float]) -> float:
        """
        Calculate stability metric from scores.
        
        Higher stability = lower variance across windows
        
        Args:
            scores: List of out-of-sample scores
            
        Returns:
            Stability score (0-1, higher is better)
        """
        if len(scores) < 2:
            return 1.0
        
        import numpy as np
        std = np.std(scores)
        mean = np.mean(scores)
        
        if mean == 0:
            return 0.0
        
        # Coefficient of variation, inverted
        cv = std / abs(mean)
        stability = 1.0 / (1.0 + cv)
        
        return stability


# TODO: Implement OutOfSampleTesting class
# TODO: Implement CrossValidation class

