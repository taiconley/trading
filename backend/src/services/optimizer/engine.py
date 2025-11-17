"""Core optimization engine that coordinates the optimization process."""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import pandas as pd

from common.db import get_db_session
from common.models import OptimizationRun, OptimizationResult
from .algorithms import (
    BaseOptimizer,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    BayesianOptimizer,
    GeneticAlgorithmOptimizer,
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
        config: Dict[str, Any] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None
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
            objective: Objective function ('sharpe_ratio', 'sortino_ratio', 'total_return', 'profit_factor',
                                          'win_rate', 'volatility', 'value_at_risk', 'avg_holding_time')
            constraints: List of constraint expressions
            num_workers: Number of parallel workers (None for CPU count)
            max_iterations: Maximum iterations for random search
            random_seed: Random seed for reproducibility
            config: Additional config (commission, slippage, etc.)
            start_date: Optional start date for backtest data filtering
            end_date: Optional end date for backtest data filtering
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
        self.start_date = start_date
        self.end_date = end_date
        
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
        elif self.algorithm == 'bayesian':
            if self.max_iterations is None:
                self.max_iterations = 50  # Default for Bayesian optimization
            return BayesianOptimizer(
                param_space=self.param_space,
                constraints=self.constraints,
                max_iterations=self.max_iterations,
                random_seed=self.random_seed,
                n_startup_trials=min(10, self.max_iterations // 5),  # 20% random trials
                multivariate=True,  # Consider parameter interactions
                patience=max(8, self.max_iterations // 3)  # Early stop if no improvement
            )
        elif self.algorithm == 'genetic':
            if self.max_iterations is None:
                self.max_iterations = 100  # Default for genetic algorithm
            
            # Extract GA-specific config
            population_size = self.config.get('population_size', 50)
            elite_size = self.config.get('elite_size', 5)
            mutation_rate = self.config.get('mutation_rate', 0.1)
            crossover_rate = self.config.get('crossover_rate', 0.8)
            tournament_size = self.config.get('tournament_size', 3)
            selection_method = self.config.get('selection_method', 'tournament')
            crossover_method = self.config.get('crossover_method', 'uniform')
            
            return GeneticAlgorithmOptimizer(
                param_space=self.param_space,
                constraints=self.constraints,
                max_iterations=self.max_iterations,
                random_seed=self.random_seed,
                population_size=population_size,
                elite_size=elite_size,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                tournament_size=tournament_size,
                selection_method=selection_method,
                crossover_method=crossover_method
            )
        else:
            raise ValueError(
                f"Unknown algorithm: {self.algorithm}. "
                "Supported: grid_search, random_search, bayesian, genetic"
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
            
            # Core performance metrics
            sharpe_ratio=metrics.get('sharpe_ratio'),
            sortino_ratio=metrics.get('sortino_ratio'),
            total_return_pct=metrics.get('total_return_pct'),
            annualized_volatility_pct=metrics.get('annualized_volatility_pct'),
            value_at_risk_pct=metrics.get('value_at_risk_pct'),
            max_drawdown_pct=metrics.get('max_drawdown_pct'),
            max_drawdown_duration_days=metrics.get('max_drawdown_duration_days'),
            
            # Trade statistics
            total_trades=metrics.get('total_trades'),
            winning_trades=metrics.get('winning_trades'),
            losing_trades=metrics.get('losing_trades'),
            win_rate=metrics.get('win_rate'),
            profit_factor=metrics.get('profit_factor'),
            
            # Trade performance
            avg_win=metrics.get('avg_win'),
            avg_loss=metrics.get('avg_loss'),
            largest_win=metrics.get('largest_win'),
            largest_loss=metrics.get('largest_loss'),
            
            # Trade timing
            avg_trade_duration_days=metrics.get('avg_trade_duration_days'),
            avg_holding_period_hours=metrics.get('avg_holding_period_hours'),
            
            # Costs
            total_commission=metrics.get('total_commission'),
            total_slippage=metrics.get('total_slippage')
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
                logger.info("Starting to generate parameter candidates...")
                batch = []
                candidate_count = 0
                
                for params in self.optimizer.generate_candidates():
                    candidate_count += 1
                    if candidate_count <= 3:
                        logger.info(f"Generated candidate {candidate_count}: {list(params.keys())[:3]}...")
                    batch.append(params)
                    
                    # Check if we should stop adding to batch (respecting max_iterations)
                    # Don't let batch grow beyond what we need
                    if self.max_iterations is not None:
                        remaining = self.max_iterations - self.optimizer.iteration
                        if len(batch) >= min(batch_size, remaining):
                            self._process_batch(db, run, batch)
                            batch = []
                            # Stop if we've reached the limit
                            if self.optimizer.iteration >= self.max_iterations:
                                break
                    elif len(batch) >= batch_size:
                        # No iteration limit, just use batch_size
                        self._process_batch(db, run, batch)
                        batch = []
                
                # Process remaining batch (only if we haven't exceeded limit)
                if batch and (self.max_iterations is None or self.optimizer.iteration < self.max_iterations):
                    # Trim batch if it would exceed max_iterations
                    if self.max_iterations is not None:
                        remaining = self.max_iterations - self.optimizer.iteration
                        batch = batch[:remaining]
                    if batch:  # Only process if there's anything left
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
        logger.info(f"Batch parameters: {[list(p.keys())[:3] for p in batch[:3]]}...")  # Show first 3 param keys
        
        # Execute backtests in parallel
        logger.info(f"Starting parallel execution with {self.executor.num_workers} workers...")
        results = self.executor.execute_batch(
            param_combinations=batch,
            strategy_name=self.strategy_name,
            symbols=self.symbols,
            timeframe=self.timeframe,
            lookback=self.lookback,
            objective=self.objective,
            config=self.config,
            start_date=self.start_date,
            end_date=self.end_date
        )
        logger.info(f"Batch execution returned {len(results)} results")
        
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
    
    async def _run_optimization_direct(
        self,
        db: Session,
        bars_data: Dict[str, pd.DataFrame],
        window_id: str = None
    ) -> Optional[AlgoResult]:
        """
        Run optimization directly on provided data without full database storage.
        
        This is used internally by validation methods like walk-forward analysis.
        Results are not stored in optimization_runs table.
        
        Args:
            db: Database session
            bars_data: Historical bars data
            window_id: Optional window identifier for logging
            
        Returns:
            Best optimization result or None
        """
        logger.info(f"Running direct optimization{f' for {window_id}' if window_id else ''}")
        
        # Generate and evaluate all candidates
        batch_size = 50
        batch = []
        
        for params in self.optimizer.generate_candidates():
            batch.append(params)
            
            if len(batch) >= batch_size:
                self._eval_batch_direct(batch, bars_data)
                batch = []
        
        # Process remaining batch
        if batch:
            self._eval_batch_direct(batch, bars_data)
        
        # Return best result
        best = self.optimizer.get_best()
        if best:
            logger.info(
                f"Direct optimization complete{f' for {window_id}' if window_id else ''}: "
                f"Best score={best.score:.4f}, params={best.params}"
            )
        else:
            logger.warning(f"Direct optimization{f' for {window_id}' if window_id else ''}: No valid results")
        
        return best
    
    def _eval_batch_direct(
        self,
        batch: List[Dict[str, Any]],
        bars_data: Dict[str, pd.DataFrame]
    ):
        """Evaluate a batch of parameters directly without database storage."""
        results = self.executor.execute_batch(
            param_combinations=batch,
            strategy_name=self.strategy_name,
            symbols=self.symbols,
            timeframe=self.timeframe,
            lookback=self.lookback,
            objective=self.objective,
            config=self.config,
            bars_data=bars_data,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Update optimizer with results
        for task_result in results:
            if task_result.success:
                algo_result = AlgoResult(
                    params=task_result.params,
                    score=task_result.result['score'],
                    metrics=task_result.result['metrics'],
                    backtest_run_id=task_result.result.get('backtest_run_id')
                )
                self.optimizer.update(algo_result)
