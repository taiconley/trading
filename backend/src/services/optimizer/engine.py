"""Core optimization engine that coordinates the optimization process."""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from common.db import get_db_session
from common.models import OptimizationRun, OptimizationResult
from .algorithms import (
    BaseOptimizer,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    ParameterSpace,
    OptimizationResult as AlgoResult
)
from .executor import ParallelExecutor, TaskResult

logger = logging.getLogger(__name__)


class OptimizationEngine:
    """
    Core optimization engine.
    
    Coordinates the optimization process:
    1. Creates optimization run in database
    2. Generates parameter combinations using selected algorithm
    3. Executes backtests in parallel
    4. Stores results in database
    5. Tracks progress and best parameters
    """
    
    def __init__(
        self,
        strategy_name: str,
        symbols: List[str],
        timeframe: str,
        lookback: int,
        param_ranges: Dict[str, Any],
        algorithm: str = 'grid_search',
        objective: str = 'sharpe_ratio',
        constraints: List[str] = None,
        num_workers: int = None,
        max_iterations: int = None,
        random_seed: int = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize optimization engine.
        
        Args:
            strategy_name: Name of strategy to optimize
            symbols: List of symbols for backtest
            timeframe: Timeframe for backtest (e.g., "1 day")
            lookback: Lookback period in days
            param_ranges: Parameter ranges to search
            algorithm: Optimization algorithm ('grid_search', 'random_search')
            objective: Objective function ('sharpe_ratio', 'total_return', 'profit_factor', 'win_rate')
            constraints: List of constraint expressions
            num_workers: Number of parallel workers (None for CPU count)
            max_iterations: Maximum iterations for random search
            random_seed: Random seed for reproducibility
            config: Additional config (commission, slippage, etc.)
        """
        self.strategy_name = strategy_name
        self.symbols = symbols
        self.timeframe = timeframe
        self.lookback = lookback
        self.param_ranges = param_ranges
        self.algorithm = algorithm
        self.objective = objective
        self.constraints = constraints or []
        self.num_workers = num_workers
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        self.config = config or {}
        
        # Create parameter space
        self.param_space = ParameterSpace(ranges=param_ranges)
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create parallel executor
        self.executor = ParallelExecutor(num_workers=num_workers)
        
        # State
        self.run_id: Optional[int] = None
        self.db_session: Optional[Session] = None
    
    def _create_optimizer(self) -> BaseOptimizer:
        """Create optimizer instance based on algorithm."""
        if self.algorithm == 'grid_search':
            return GridSearchOptimizer(
                param_space=self.param_space,
                constraints=self.constraints,
                max_iterations=self.max_iterations,
                random_seed=self.random_seed
            )
        elif self.algorithm == 'random_search':
            if self.max_iterations is None:
                self.max_iterations = 100  # Default for random search
            return RandomSearchOptimizer(
                param_space=self.param_space,
                constraints=self.constraints,
                max_iterations=self.max_iterations,
                random_seed=self.random_seed
            )
        else:
            raise ValueError(
                f"Unknown algorithm: {self.algorithm}. "
                "Supported: grid_search, random_search"
            )
    
    def _create_run_record(self, db: Session) -> OptimizationRun:
        """Create optimization run record in database."""
        run = OptimizationRun(
            strategy_name=self.strategy_name,
            algorithm=self.algorithm,
            symbols=self.symbols,
            timeframe=self.timeframe,
            param_ranges=self.param_ranges,
            objective=self.objective,
            status='pending',
            total_combinations=self.param_space.count_combinations() if self.algorithm == 'grid_search' else self.max_iterations,
            completed_combinations=0,
            config={
                'lookback': self.lookback,
                'constraints': self.constraints,
                'num_workers': self.num_workers,
                'random_seed': self.random_seed,
                **self.config
            }
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return run
    
    def _update_progress(
        self,
        db: Session,
        run: OptimizationRun,
        completed: int,
        best_params: Dict[str, Any] = None,
        best_score: float = None
    ):
        """Update optimization run progress in database."""
        run.completed_combinations = completed
        if best_params is not None:
            run.best_params = best_params
        if best_score is not None:
            run.best_score = best_score
        db.commit()
    
    def _store_result(
        self,
        db: Session,
        run_id: int,
        params: Dict[str, Any],
        score: float,
        metrics: Dict[str, Any],
        backtest_run_id: int = None
    ):
        """Store optimization result in database."""
        result = OptimizationResult(
            run_id=run_id,
            params_json=params,
            backtest_run_id=backtest_run_id,
            score=score,
            sharpe_ratio=metrics.get('sharpe_ratio'),
            total_return=metrics.get('total_return_pct'),
            max_drawdown=metrics.get('max_drawdown_pct'),
            win_rate=metrics.get('win_rate'),
            profit_factor=metrics.get('profit_factor'),
            total_trades=metrics.get('total_trades')
        )
        db.add(result)
        db.commit()
        return result
    
    def run(self, batch_size: int = 50) -> Dict[str, Any]:
        """
        Run the optimization.
        
        Args:
            batch_size: Number of parameter combinations to evaluate in each batch
        
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting optimization: {self.strategy_name} with {self.algorithm}")
        logger.info(f"Parameter space: {self.param_ranges}")
        logger.info(f"Total combinations: {self.param_space.count_combinations()}")
        
        # Create database session
        with get_db_session() as db:
            # Create run record
            run = self._create_run_record(db)
            self.run_id = run.id
            logger.info(f"Created optimization run ID: {self.run_id}")
            
            # Update status to running
            run.status = 'running'
            run.start_time = datetime.now(timezone.utc)
            db.commit()
            
            try:
                # Generate and evaluate parameter combinations in batches
                batch = []
                
                for params in self.optimizer.generate_candidates():
                    batch.append(params)
                    
                    # Process batch when full
                    if len(batch) >= batch_size:
                        self._process_batch(db, run, batch)
                        batch = []
                
                # Process remaining batch
                if batch:
                    self._process_batch(db, run, batch)
                
                # Mark as completed
                run.status = 'completed'
                run.end_time = datetime.now(timezone.utc)
                
                # Get final best result
                best = self.optimizer.get_best()
                if best:
                    run.best_params = best.params
                    run.best_score = best.score
                
                db.commit()
                
                logger.info(
                    f"Optimization completed: {run.completed_combinations} combinations tested"
                )
                if best:
                    logger.info(f"Best score: {best.score:.4f}")
                    logger.info(f"Best params: {best.params}")
                
                return {
                    'run_id': self.run_id,
                    'status': 'completed',
                    'completed_combinations': run.completed_combinations,
                    'best_params': run.best_params,
                    'best_score': float(run.best_score) if run.best_score else None,
                    'duration_seconds': (run.end_time - run.start_time).total_seconds()
                }
                
            except Exception as e:
                # Mark as failed
                run.status = 'failed'
                run.end_time = datetime.now(timezone.utc)
                run.error_message = str(e)
                db.commit()
                
                logger.error(f"Optimization failed: {e}", exc_info=True)
                raise
    
    def _process_batch(self, db: Session, run: OptimizationRun, batch: List[Dict[str, Any]]):
        """Process a batch of parameter combinations."""
        logger.info(f"Processing batch of {len(batch)} combinations")
        
        # Execute backtests in parallel
        results = self.executor.execute_batch(
            param_combinations=batch,
            strategy_name=self.strategy_name,
            symbols=self.symbols,
            timeframe=self.timeframe,
            lookback=self.lookback,
            objective=self.objective,
            config=self.config
        )
        
        # Process results
        for task_result in results:
            if task_result.success:
                # Create algorithm result
                algo_result = AlgoResult(
                    params=task_result.params,
                    score=task_result.result['score'],
                    metrics=task_result.result['metrics'],
                    backtest_run_id=task_result.result.get('backtest_run_id')
                )
                
                # Update optimizer state
                self.optimizer.update(algo_result)
                
                # Store in database
                self._store_result(
                    db=db,
                    run_id=run.id,
                    params=task_result.params,
                    score=task_result.result['score'],
                    metrics=task_result.result['metrics'],
                    backtest_run_id=task_result.result.get('backtest_run_id')
                )
            else:
                logger.warning(
                    f"Backtest failed for params {task_result.params}: {task_result.error}"
                )
        
        # Update progress
        best = self.optimizer.get_best()
        self._update_progress(
            db=db,
            run=run,
            completed=self.optimizer.iteration,
            best_params=best.params if best else None,
            best_score=best.score if best else None
        )
        
        logger.info(
            f"Progress: {run.completed_combinations}/{run.total_combinations or '?'} "
            f"({self.optimizer.get_progress() * 100:.1f}%)"
        )
        if best:
            logger.info(f"Current best: {best.score:.4f} with params {best.params}")

