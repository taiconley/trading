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

from common.db import get_db_session
from services.optimizer.algorithms import BaseOptimizer, ParameterSpace
from services.optimizer.executor import ParallelExecutor

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
        from services.optimizer.engine import OptimizationEngine
        
        optimizer = OptimizationEngine(
            strategy_name=self.strategy_name,
            symbols=self.symbols,
            timeframe=self.timeframe,
            lookback=self.lookback,
            param_ranges=self.param_space.ranges,
            algorithm=self.algorithm,
            objective=self.objective,
            constraints=self.constraints,
            max_iterations=None,  # Use all combinations for walk-forward
            num_workers=self.num_workers,
            random_seed=self.random_seed
        )
        
        # Run in-sample optimization
        with get_db_session() as db:
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
            results = executor.execute_batch(
                param_combinations=[window.best_params],
                strategy_name=self.strategy_name,
                symbols=self.symbols,
                timeframe=self.timeframe,
                lookback=self.lookback,
                objective=self.objective,
                config={},
                bars_data=out_sample_data
            )
            
            if not results or not results[0].success:
                logger.error(f"Failed to validate window on out-of-sample data")
                return window
            
            result = results[0].result
            
            window.out_sample_score = result['score']
            
            logger.info(
                f"Window {window.window_id} out-of-sample: "
                f"Score={result['score']:.4f}, Trades={result['metrics']['total_trades']}"
            )
            
            # Check minimum trades requirement
            if result['metrics']['total_trades'] < self.config.min_trades_required:
                logger.warning(
                    f"Window {window.window_id}: Insufficient trades "
                    f"({result['metrics']['total_trades']} < {self.config.min_trades_required})"
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


class OutOfSampleTesting:
    """
    Out-of-sample testing with train/test split.
    
    Optimizes parameters on training data and validates on testing data
    to detect overfitting and assess generalization.
    """
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        strategy_name: str = None,
        symbols: List[str] = None,
        timeframe: str = None,
        param_space: ParameterSpace = None,
        algorithm: str = 'grid_search',
        objective: str = 'sharpe_ratio',
        constraints: Optional[List[str]] = None,
        lookback: int = 100,
        num_workers: int = 4,
        random_seed: Optional[int] = None
    ):
        """
        Initialize out-of-sample testing.
        
        Args:
            train_ratio: Fraction of data for training (0-1)
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
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        
        self.train_ratio = train_ratio
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
        
        self.train_data: Optional[Dict[str, pd.DataFrame]] = None
        self.test_data: Optional[Dict[str, pd.DataFrame]] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.train_score: Optional[float] = None
        self.test_score: Optional[float] = None
    
    def split_data(
        self,
        bars_data: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Split data into train and test sets.
        
        Args:
            bars_data: Historical bars data for all symbols
            
        Returns:
            Tuple of (train_data, test_data)
        """
        train_data = {}
        test_data = {}
        
        for symbol, df in bars_data.items():
            split_idx = int(len(df) * self.train_ratio)
            
            if split_idx < self.lookback:
                raise ValueError(
                    f"Training set too small for {symbol}: {split_idx} bars < {self.lookback} lookback"
                )
            
            if len(df) - split_idx < self.lookback:
                raise ValueError(
                    f"Testing set too small for {symbol}: {len(df) - split_idx} bars < {self.lookback} lookback"
                )
            
            train_data[symbol] = df.iloc[:split_idx].copy()
            test_data[symbol] = df.iloc[split_idx:].copy()
        
        logger.info(
            f"Split data: {len(train_data[self.symbols[0]])} train bars, "
            f"{len(test_data[self.symbols[0]])} test bars"
        )
        
        self.train_data = train_data
        self.test_data = test_data
        
        return train_data, test_data
    
    async def run(
        self,
        bars_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Run out-of-sample testing.
        
        Args:
            bars_data: Historical bars data for all symbols
            
        Returns:
            Testing results with metrics
        """
        logger.info("Starting out-of-sample testing")
        
        # Split data
        train_data, test_data = self.split_data(bars_data)
        
        # Optimize on training data
        from services.optimizer.engine import OptimizationEngine
        
        optimizer = OptimizationEngine(
            strategy_name=self.strategy_name,
            symbols=self.symbols,
            timeframe=self.timeframe,
            lookback=self.lookback,
            param_ranges=self.param_space.ranges,
            algorithm=self.algorithm,
            objective=self.objective,
            constraints=self.constraints,
            num_workers=self.num_workers,
            random_seed=self.random_seed
        )
        
        with get_db_session() as db:
            best_result = await optimizer._run_optimization_direct(
                db=db,
                bars_data=train_data,
                window_id="train"
            )
        
        if not best_result:
            raise RuntimeError("No valid optimization results on training data")
        
        self.best_params = best_result.params
        self.train_score = best_result.score
        
        logger.info(
            f"Training complete: Best params={best_result.params}, Score={best_result.score:.4f}"
        )
        
        # Validate on testing data
        executor = ParallelExecutor(num_workers=1)
        
        test_results = executor.execute_batch(
            param_combinations=[self.best_params],
            strategy_name=self.strategy_name,
            symbols=self.symbols,
            timeframe=self.timeframe,
            lookback=self.lookback,
            objective=self.objective,
            config={},
            bars_data=test_data
        )
        
        if not test_results or not test_results[0].success:
            raise ValueError("Failed to evaluate parameters on test data")
        
        test_result = test_results[0].result
        
        self.test_score = test_result['score']
        
        logger.info(
            f"Testing complete: Score={test_result['score']:.4f}, Trades={test_result['metrics']['total_trades']}"
        )
        
        # Calculate overfitting metrics
        degradation = self.train_score - self.test_score
        degradation_pct = (degradation / self.train_score * 100) if self.train_score != 0 else 0
        
        results = {
            'best_params': self.best_params,
            'train_score': self.train_score,
            'test_score': self.test_score,
            'score_degradation': degradation,
            'score_degradation_pct': degradation_pct,
            'train_metrics': {
                'sharpe_ratio': best_result.score if self.objective == 'sharpe_ratio' else None
            },
            'test_metrics': test_result['metrics'],
            'overfitting_detected': degradation_pct > 50  # Flag if >50% degradation
        }
        
        logger.info(
            f"Out-of-sample testing complete: "
            f"Train={self.train_score:.4f}, Test={self.test_score:.4f}, "
            f"Degradation={degradation_pct:.1f}%"
        )
        
        return results


class TimeSeriesCrossValidation:
    """
    K-fold cross-validation for time series data.
    
    Uses purged and embargoed splits to prevent lookahead bias
    in time series validation.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size_ratio: float = 0.2,
        purge_days: int = 0,
        embargo_days: int = 0,
        strategy_name: str = None,
        symbols: List[str] = None,
        timeframe: str = None,
        param_space: ParameterSpace = None,
        algorithm: str = 'grid_search',
        objective: str = 'sharpe_ratio',
        constraints: Optional[List[str]] = None,
        lookback: int = 100,
        num_workers: int = 4,
        random_seed: Optional[int] = None
    ):
        """
        Initialize time series cross-validation.
        
        Args:
            n_splits: Number of folds
            test_size_ratio: Ratio of test set size (0-1)
            purge_days: Days to purge after training set (prevent leakage)
            embargo_days: Days to embargo before test set (prevent leakage)
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
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not 0 < test_size_ratio < 1:
            raise ValueError("test_size_ratio must be between 0 and 1")
        
        self.n_splits = n_splits
        self.test_size_ratio = test_size_ratio
        self.purge_days = purge_days
        self.embargo_days = embargo_days
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
        
        self.fold_results: List[Dict[str, Any]] = []
    
    def create_folds(
        self,
        bars_data: Dict[str, pd.DataFrame]
    ) -> List[Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]]:
        """
        Create time series cross-validation folds.
        
        Args:
            bars_data: Historical bars data for all symbols
            
        Returns:
            List of (train_data, test_data) tuples for each fold
        """
        # Get total number of bars (assume all symbols have same length)
        total_bars = len(bars_data[self.symbols[0]])
        test_size = int(total_bars * self.test_size_ratio)
        
        # Calculate fold positions
        folds = []
        fold_step = total_bars // self.n_splits
        
        for i in range(self.n_splits):
            # Test set is at position i * fold_step
            test_start = i * fold_step
            test_end = test_start + test_size
            
            if test_end > total_bars:
                break
            
            # Training set is everything before test (with purge)
            train_end = test_start - self.purge_days - 1
            
            if train_end < self.lookback:
                logger.warning(f"Fold {i+1}: Insufficient training data, skipping")
                continue
            
            # Apply embargo to test set
            actual_test_start = test_start + self.embargo_days
            
            if actual_test_start >= test_end:
                logger.warning(f"Fold {i+1}: Embargo too large, skipping")
                continue
            
            # Create train and test data
            train_data = {}
            test_data = {}
            
            for symbol, df in bars_data.items():
                train_data[symbol] = df.iloc[:train_end].copy()
                test_data[symbol] = df.iloc[actual_test_start:test_end].copy()
            
            folds.append((train_data, test_data))
            
            logger.info(
                f"Fold {i+1}: Train[0:{train_end}], Test[{actual_test_start}:{test_end}] "
                f"(purge={self.purge_days}, embargo={self.embargo_days})"
            )
        
        logger.info(f"Created {len(folds)} cross-validation folds")
        return folds
    
    async def run(
        self,
        bars_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Run cross-validation.
        
        Args:
            bars_data: Historical bars data for all symbols
            
        Returns:
            Cross-validation results with metrics
        """
        logger.info(f"Starting {self.n_splits}-fold time series cross-validation")
        
        # Create folds
        folds = self.create_folds(bars_data)
        
        if not folds:
            raise ValueError("No valid folds could be created with given configuration")
        
        # Process each fold
        fold_results = []
        
        for fold_idx, (train_data, test_data) in enumerate(folds, 1):
            logger.info(f"Processing fold {fold_idx}/{len(folds)}")
            
            # Optimize on training data
            from services.optimizer.engine import OptimizationEngine
            
            optimizer = OptimizationEngine(
                strategy_name=self.strategy_name,
                symbols=self.symbols,
                timeframe=self.timeframe,
                lookback=self.lookback,
                param_ranges=self.param_space.ranges,
                algorithm=self.algorithm,
                objective=self.objective,
                constraints=self.constraints,
                num_workers=self.num_workers,
                random_seed=self.random_seed
            )
            
            with get_db_session() as db:
                best_result = await optimizer._run_optimization_direct(
                    db=db,
                    bars_data=train_data,
                    window_id=f"cv_fold{fold_idx}_train"
                )
            
            if not best_result:
                logger.warning(f"Fold {fold_idx}: No valid optimization results")
                continue
            
            # Validate on test data
            executor = ParallelExecutor(num_workers=1)
            
            test_results = executor.execute_batch(
                param_combinations=[best_result.params],
                strategy_name=self.strategy_name,
                symbols=self.symbols,
                timeframe=self.timeframe,
                lookback=self.lookback,
                objective=self.objective,
                config={},
                bars_data=test_data
            )
            
            if not test_results or not test_results[0].success:
                logger.warning(f"Fold {i+1}: Failed to validate on test data")
                continue
            
            test_result = test_results[0].result
            
            fold_result = {
                'fold': fold_idx,
                'best_params': best_result.params,
                'train_score': best_result.score,
                'test_score': test_result['score'],
                'test_trades': test_result['metrics']['total_trades']
            }
            
            fold_results.append(fold_result)
            
            logger.info(
                f"Fold {fold_idx} complete: "
                f"Train={best_result.score:.4f}, Test={test_result['score']:.4f}"
            )
        
        if not fold_results:
            raise RuntimeError("No valid fold results")
        
        # Calculate aggregate metrics
        train_scores = [f['train_score'] for f in fold_results]
        test_scores = [f['test_score'] for f in fold_results]
        
        import numpy as np
        
        results = {
            'n_folds': len(fold_results),
            'avg_train_score': np.mean(train_scores),
            'avg_test_score': np.mean(test_scores),
            'std_test_score': np.std(test_scores),
            'score_degradation': np.mean(train_scores) - np.mean(test_scores),
            'fold_results': fold_results,
            'stability': 1.0 / (1.0 + np.std(test_scores) / abs(np.mean(test_scores))) if np.mean(test_scores) != 0 else 0
        }
        
        logger.info(
            f"Cross-validation complete: "
            f"Avg train={results['avg_train_score']:.4f}, "
            f"Avg test={results['avg_test_score']:.4f}, "
            f"Std={results['std_test_score']:.4f}, "
            f"Stability={results['stability']:.4f}"
        )
        
        self.fold_results = fold_results
        return results

