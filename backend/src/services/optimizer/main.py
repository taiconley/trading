"""
Optimizer Service - Parameter Optimization for Trading Strategies

Provides both CLI and REST API interfaces for strategy parameter optimization.

CLI Usage:
    python main.py optimize --strategy SMA_Crossover --params '{"short_period": [5,10,15], "long_period": [20,30,40]}'

REST API:
    Run: uvicorn main:app --host 0.0.0.0 --port 8006
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy import desc

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.config import settings
from common.db import get_db_session
from common.models import OptimizationRun, OptimizationResult
from common.logging import setup_logging
from services.optimizer.engine import OptimizationEngine

# Setup logging
setup_logging(service_name="optimizer")
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Strategy Parameter Optimizer",
    description="Optimize trading strategy parameters using various algorithms",
    version="1.0.0"
)

# Track running optimizations
running_optimizations: Dict[int, OptimizationEngine] = {}


# =============================================================================
# Pydantic Models
# =============================================================================

class OptimizationRequest(BaseModel):
    """Request to start a new optimization."""
    strategy_name: str = Field(..., description="Strategy to optimize")
    symbols: List[str] = Field(..., description="Symbols for backtest")
    timeframe: str = Field(default="1 day", description="Timeframe for backtest")
    lookback: int = Field(default=365, description="Lookback period in days")
    param_ranges: Dict[str, Any] = Field(..., description="Parameter ranges to search")
    algorithm: str = Field(default="grid_search", description="Optimization algorithm")
    objective: str = Field(default="sharpe_ratio", description="Objective function")
    constraints: Optional[List[str]] = Field(default=None, description="Constraint expressions")
    num_workers: Optional[int] = Field(default=None, description="Number of parallel workers")
    max_iterations: Optional[int] = Field(default=None, description="Max iterations (random search)")
    random_seed: Optional[int] = Field(default=None, description="Random seed")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Additional config")
    
    class Config:
        schema_extra = {
            "example": {
                "strategy_name": "SMA_Crossover",
                "symbols": ["AAPL"],
                "timeframe": "1 day",
                "lookback": 365,
                "param_ranges": {
                    "short_period": [5, 10, 15, 20],
                    "long_period": [30, 40, 50, 60]
                },
                "algorithm": "grid_search",
                "objective": "sharpe_ratio",
                "constraints": ["short_period < long_period"]
            }
        }


class OptimizationResponse(BaseModel):
    """Response with optimization details."""
    run_id: int
    strategy_name: str
    algorithm: str
    status: str
    completed_combinations: int
    total_combinations: Optional[int]
    best_params: Optional[Dict[str, Any]]
    best_score: Optional[float]
    created_at: str


class OptimizationResultResponse(BaseModel):
    """Response with individual optimization result."""
    id: int
    params: Dict[str, Any]
    score: float
    sharpe_ratio: Optional[float]
    total_return: Optional[float]
    max_drawdown: Optional[float]
    win_rate: Optional[float]
    profit_factor: Optional[float]
    total_trades: Optional[int]


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/healthz")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "optimizer",
        "running_optimizations": len(running_optimizations)
    }


@app.post("/optimizations", response_model=OptimizationResponse)
async def start_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """
    Start a new parameter optimization.
    
    The optimization will run in the background.
    Use GET /optimizations/{id} to check status and results.
    """
    try:
        # Create optimization engine
        engine = OptimizationEngine(
            strategy_name=request.strategy_name,
            symbols=request.symbols,
            timeframe=request.timeframe,
            lookback=request.lookback,
            param_ranges=request.param_ranges,
            algorithm=request.algorithm,
            objective=request.objective,
            constraints=request.constraints,
            num_workers=request.num_workers,
            max_iterations=request.max_iterations,
            random_seed=request.random_seed,
            config=request.config
        )
        
        # Run optimization in background
        background_tasks.add_task(run_optimization_task, engine)
        
        # Return initial response (run record will be created in background)
        # For now, return a placeholder response
        return OptimizationResponse(
            run_id=0,  # Will be updated
            strategy_name=request.strategy_name,
            algorithm=request.algorithm,
            status="pending",
            completed_combinations=0,
            total_combinations=engine.param_space.count_combinations() if request.algorithm == 'grid_search' else request.max_iterations,
            best_params=None,
            best_score=None,
            created_at=str(datetime.now(timezone.utc))
        )
        
    except Exception as e:
        logger.error(f"Failed to start optimization: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/optimizations/{run_id}", response_model=OptimizationResponse)
def get_optimization(run_id: int):
    """Get optimization run status and results."""
    with get_db_session() as db:
        run = db.query(OptimizationRun).filter(OptimizationRun.id == run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail="Optimization run not found")
        
        return OptimizationResponse(
            run_id=run.id,
            strategy_name=run.strategy_name,
            algorithm=run.algorithm,
            status=run.status,
            completed_combinations=run.completed_combinations,
            total_combinations=run.total_combinations,
            best_params=run.best_params,
            best_score=float(run.best_score) if run.best_score else None,
            created_at=str(run.created_at)
        )


@app.get("/optimizations", response_model=List[OptimizationResponse])
def list_optimizations(limit: int = 20, strategy: Optional[str] = None):
    """List optimization runs."""
    with get_db_session() as db:
        query = db.query(OptimizationRun)
        
        if strategy:
            query = query.filter(OptimizationRun.strategy_name == strategy)
        
        runs = query.order_by(desc(OptimizationRun.created_at)).limit(limit).all()
        
        return [
            OptimizationResponse(
                run_id=run.id,
                strategy_name=run.strategy_name,
                algorithm=run.algorithm,
                status=run.status,
                completed_combinations=run.completed_combinations,
                total_combinations=run.total_combinations,
                best_params=run.best_params,
                best_score=float(run.best_score) if run.best_score else None,
                created_at=str(run.created_at)
            )
            for run in runs
        ]


@app.get("/optimizations/{run_id}/results", response_model=List[OptimizationResultResponse])
def get_optimization_results(run_id: int, top_n: int = 20):
    """Get detailed results from an optimization run (top N by score)."""
    with get_db_session() as db:
        # Check if run exists
        run = db.query(OptimizationRun).filter(OptimizationRun.id == run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail="Optimization run not found")
        
        # Get top N results
        results = (
            db.query(OptimizationResult)
            .filter(OptimizationResult.run_id == run_id)
            .order_by(desc(OptimizationResult.score))
            .limit(top_n)
            .all()
        )
        
        return [
            OptimizationResultResponse(
                id=result.id,
                params=result.params_json,
                score=float(result.score),
                sharpe_ratio=float(result.sharpe_ratio) if result.sharpe_ratio else None,
                total_return=float(result.total_return) if result.total_return else None,
                max_drawdown=float(result.max_drawdown) if result.max_drawdown else None,
                win_rate=float(result.win_rate) if result.win_rate else None,
                profit_factor=float(result.profit_factor) if result.profit_factor else None,
                total_trades=result.total_trades
            )
            for result in results
        ]


# =============================================================================
# Background Tasks
# =============================================================================

async def run_optimization_task(engine: OptimizationEngine):
    """Background task to run optimization."""
    try:
        result = engine.run()
        logger.info(f"Optimization {result['run_id']} completed successfully")
    except Exception as e:
        logger.error(f"Optimization task failed: {e}", exc_info=True)


# =============================================================================
# CLI Interface
# =============================================================================

def cli_optimize(args):
    """Run optimization from CLI."""
    # Parse param ranges
    try:
        param_ranges = json.loads(args.params)
    except json.JSONDecodeError:
        logger.error("Invalid JSON for --params")
        return 1
    
    # Parse constraints
    constraints = None
    if args.constraints:
        constraints = args.constraints.split(',')
    
    # Parse symbols
    symbols = args.symbols.split(',') if ',' in args.symbols else [args.symbols]
    
    # Parse config
    config = {}
    if args.config:
        try:
            config = json.loads(args.config)
        except json.JSONDecodeError:
            logger.error("Invalid JSON for --config")
            return 1
    
    logger.info("=" * 80)
    logger.info("Strategy Parameter Optimization")
    logger.info("=" * 80)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Objective: {args.objective}")
    logger.info(f"Parameter ranges: {param_ranges}")
    if constraints:
        logger.info(f"Constraints: {constraints}")
    logger.info("=" * 80)
    
    try:
        # Create optimization engine
        engine = OptimizationEngine(
            strategy_name=args.strategy,
            symbols=symbols,
            timeframe=args.timeframe,
            lookback=args.lookback,
            param_ranges=param_ranges,
            algorithm=args.algorithm,
            objective=args.objective,
            constraints=constraints,
            num_workers=args.workers,
            max_iterations=args.max_iterations,
            random_seed=args.seed,
            config=config
        )
        
        # Run optimization
        result = engine.run(batch_size=args.batch_size)
        
        logger.info("=" * 80)
        logger.info("Optimization Complete!")
        logger.info("=" * 80)
        logger.info(f"Run ID: {result['run_id']}")
        logger.info(f"Status: {result['status']}")
        logger.info(f"Combinations tested: {result['completed_combinations']}")
        logger.info(f"Duration: {result['duration_seconds']:.1f} seconds")
        if result['best_score'] is not None:
            logger.info(f"Best {args.objective}: {result['best_score']:.4f}")
            logger.info(f"Best parameters: {json.dumps(result['best_params'], indent=2)}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        return 1


def cli_walk_forward(args):
    """Run walk-forward analysis from CLI."""
    from services.optimizer.validation import WalkForwardAnalysis, WalkForwardConfig
    from services.optimizer.algorithms import ParameterSpace
    from common.models import Candle
    from common.db import get_db_session
    import asyncio
    import pandas as pd
    
    # Parse parameter ranges
    try:
        param_ranges = json.loads(args.params)
    except json.JSONDecodeError:
        logger.error("Invalid JSON for --params")
        return 1
    
    # Parse constraints
    constraints = args.constraints.split(',') if args.constraints else None
    
    # Parse symbols
    symbols = args.symbols.split(',') if ',' in args.symbols else [args.symbols]
    
    logger.info("=" * 80)
    logger.info("Walk-Forward Analysis")
    logger.info("=" * 80)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"In-sample: {args.in_sample_days} days, Out-sample: {args.out_sample_days} days")
    logger.info(f"Step: {args.step_days} days, Anchored: {args.anchored}")
    logger.info("=" * 80)
    
    try:
        # Load historical data (same pattern as backtester)
        logger.info("Loading historical data...")
        bars_data = {}
        with get_db_session() as db:
            for symbol in symbols:
                candles = db.query(Candle).filter(
                    Candle.symbol == symbol,
                    Candle.tf == args.timeframe
                ).order_by(Candle.ts).all()
                
                if not candles:
                    raise ValueError(f"No data found for {symbol} with timeframe '{args.timeframe}'")
                
                df = pd.DataFrame([{
                    'timestamp': c.ts,
                    'open': float(c.open),
                    'high': float(c.high),
                    'low': float(c.low),
                    'close': float(c.close),
                    'volume': c.volume
                } for c in candles])
                
                bars_data[symbol] = df.sort_values('timestamp').reset_index(drop=True)
                logger.info(f"Loaded {len(df)} bars for {symbol}")
        
        # Create walk-forward config
        wf_config = WalkForwardConfig(
            in_sample_days=args.in_sample_days,
            out_sample_days=args.out_sample_days,
            step_days=args.step_days,
            anchored=args.anchored
        )
        
        # Create walk-forward analysis
        wf = WalkForwardAnalysis(
            config=wf_config,
            strategy_name=args.strategy,
            symbols=symbols,
            timeframe=args.timeframe,
            param_space=ParameterSpace(ranges=param_ranges),
            algorithm=args.algorithm,
            objective=args.objective,
            constraints=constraints,
            lookback=args.lookback,
            num_workers=args.workers or 4,
            random_seed=args.seed
        )
        
        # Get date range from data
        start_date = min([df['timestamp'].min() for df in bars_data.values()])
        end_date = max([df['timestamp'].max() for df in bars_data.values()])
        
        # Run walk-forward analysis
        results = asyncio.run(wf.run(start_date, end_date, bars_data))
        
        logger.info("=" * 80)
        logger.info("Walk-Forward Analysis Complete!")
        logger.info("=" * 80)
        logger.info(f"Total windows: {results['total_windows']}")
        logger.info(f"Valid windows: {results['valid_windows']}")
        if results['valid_windows'] > 0:
            logger.info(f"Avg in-sample score: {results['avg_in_sample_score']:.4f}")
            logger.info(f"Avg out-sample score: {results['avg_out_sample_score']:.4f}")
            logger.info(f"Score degradation: {results['score_degradation']:.4f}")
            logger.info(f"Stability: {results['score_stability']:.4f}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Walk-forward analysis failed: {e}", exc_info=True)
        return 1


def cli_out_of_sample(args):
    """Run out-of-sample testing from CLI."""
    from services.optimizer.validation import OutOfSampleTesting
    from services.optimizer.algorithms import ParameterSpace
    from common.models import Candle
    from common.db import get_db_session
    import asyncio
    import pandas as pd
    
    # Parse parameter ranges
    try:
        param_ranges = json.loads(args.params)
    except json.JSONDecodeError:
        logger.error("Invalid JSON for --params")
        return 1
    
    # Parse constraints
    constraints = args.constraints.split(',') if args.constraints else None
    
    # Parse symbols
    symbols = args.symbols.split(',') if ',' in args.symbols else [args.symbols]
    
    logger.info("=" * 80)
    logger.info("Out-of-Sample Testing")
    logger.info("=" * 80)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Train ratio: {args.train_ratio:.1%}")
    logger.info("=" * 80)
    
    try:
        # Load historical data (same pattern as backtester)
        logger.info("Loading historical data...")
        bars_data = {}
        with get_db_session() as db:
            for symbol in symbols:
                candles = db.query(Candle).filter(
                    Candle.symbol == symbol,
                    Candle.tf == args.timeframe
                ).order_by(Candle.ts).all()
                
                if not candles:
                    raise ValueError(f"No data found for {symbol} with timeframe '{args.timeframe}'")
                
                df = pd.DataFrame([{
                    'timestamp': c.ts,
                    'open': float(c.open),
                    'high': float(c.high),
                    'low': float(c.low),
                    'close': float(c.close),
                    'volume': c.volume
                } for c in candles])
                
                bars_data[symbol] = df.sort_values('timestamp').reset_index(drop=True)
                logger.info(f"Loaded {len(df)} bars for {symbol}")
        
        # Create out-of-sample testing
        oos = OutOfSampleTesting(
            train_ratio=args.train_ratio,
            strategy_name=args.strategy,
            symbols=symbols,
            timeframe=args.timeframe,
            param_space=ParameterSpace(ranges=param_ranges),
            algorithm=args.algorithm,
            objective=args.objective,
            constraints=constraints,
            lookback=args.lookback,
            num_workers=args.workers or 4,
            random_seed=args.seed
        )
        
        # Run out-of-sample testing
        results = asyncio.run(oos.run(bars_data))
        
        logger.info("=" * 80)
        logger.info("Out-of-Sample Testing Complete!")
        logger.info("=" * 80)
        logger.info(f"Best parameters: {json.dumps(results['best_params'], indent=2)}")
        logger.info(f"Train score: {results['train_score']:.4f}")
        logger.info(f"Test score: {results['test_score']:.4f}")
        logger.info(f"Score degradation: {results['score_degradation']:.4f} ({results['score_degradation_pct']:.1f}%)")
        logger.info(f"Overfitting detected: {results['overfitting_detected']}")
        logger.info("=" * 80)
        
        print("\n" + "=" * 80)
        print("OUT-OF-SAMPLE TESTING COMPLETE!")
        print("=" * 80)
        print(f"Best parameters: {json.dumps(results['best_params'], indent=2)}")
        print(f"Train score: {results['train_score']:.4f}")
        print(f"Test score: {results['test_score']:.4f}")
        print(f"Score degradation: {results['score_degradation']:.4f} ({results['score_degradation_pct']:.1f}%)")
        print(f"Overfitting detected: {results['overfitting_detected']}")
        print("=" * 80 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Out-of-sample testing failed: {e}", exc_info=True)
        return 1


def cli_cross_validate(args):
    """Run cross-validation from CLI."""
    from services.optimizer.validation import TimeSeriesCrossValidation
    from services.optimizer.algorithms import ParameterSpace
    from common.models import Candle
    from common.db import get_db_session
    import asyncio
    import pandas as pd
    
    # Parse parameter ranges
    try:
        param_ranges = json.loads(args.params)
    except json.JSONDecodeError:
        logger.error("Invalid JSON for --params")
        return 1
    
    # Parse constraints
    constraints = args.constraints.split(',') if args.constraints else None
    
    # Parse symbols
    symbols = args.symbols.split(',') if ',' in args.symbols else [args.symbols]
    
    logger.info("=" * 80)
    logger.info("Cross-Validation")
    logger.info("=" * 80)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Folds: {args.n_splits}, Test ratio: {args.test_ratio:.1%}")
    logger.info(f"Purge: {args.purge_days} days, Embargo: {args.embargo_days} days")
    logger.info("=" * 80)
    
    try:
        # Load historical data (same pattern as backtester)
        logger.info("Loading historical data...")
        bars_data = {}
        with get_db_session() as db:
            for symbol in symbols:
                candles = db.query(Candle).filter(
                    Candle.symbol == symbol,
                    Candle.tf == args.timeframe
                ).order_by(Candle.ts).all()
                
                if not candles:
                    raise ValueError(f"No data found for {symbol} with timeframe '{args.timeframe}'")
                
                df = pd.DataFrame([{
                    'timestamp': c.ts,
                    'open': float(c.open),
                    'high': float(c.high),
                    'low': float(c.low),
                    'close': float(c.close),
                    'volume': c.volume
                } for c in candles])
                
                bars_data[symbol] = df.sort_values('timestamp').reset_index(drop=True)
                logger.info(f"Loaded {len(df)} bars for {symbol}")
        
        # Create cross-validation
        cv = TimeSeriesCrossValidation(
            n_splits=args.n_splits,
            test_size_ratio=args.test_ratio,
            purge_days=args.purge_days,
            embargo_days=args.embargo_days,
            strategy_name=args.strategy,
            symbols=symbols,
            timeframe=args.timeframe,
            param_space=ParameterSpace(ranges=param_ranges),
            algorithm=args.algorithm,
            objective=args.objective,
            constraints=constraints,
            lookback=args.lookback,
            num_workers=args.workers or 4,
            random_seed=args.seed
        )
        
        # Run cross-validation
        results = asyncio.run(cv.run(bars_data))
        
        logger.info("=" * 80)
        logger.info("Cross-Validation Complete!")
        logger.info("=" * 80)
        logger.info(f"Valid folds: {results['n_folds']}")
        logger.info(f"Avg train score: {results['avg_train_score']:.4f}")
        logger.info(f"Avg test score: {results['avg_test_score']:.4f}")
        logger.info(f"Test score std: {results['std_test_score']:.4f}")
        logger.info(f"Score degradation: {results['score_degradation']:.4f}")
        logger.info(f"Stability: {results['stability']:.4f}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Cross-validation failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Strategy Parameter Optimizer")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Run parameter optimization')
    optimize_parser.add_argument('--strategy', required=True, help='Strategy name')
    optimize_parser.add_argument('--symbols', required=True, help='Comma-separated symbols')
    optimize_parser.add_argument('--timeframe', default='1 day', help='Timeframe')
    optimize_parser.add_argument('--lookback', type=int, default=100, help='Lookback days')
    optimize_parser.add_argument('--params', required=True, help='Parameter ranges (JSON)')
    optimize_parser.add_argument('--algorithm', default='grid_search', 
                                help='Algorithm (grid_search, random_search, bayesian)')
    optimize_parser.add_argument('--objective', default='sharpe_ratio',
                                help='Objective function (sharpe_ratio, total_return, profit_factor, win_rate)')
    optimize_parser.add_argument('--constraints', help='Comma-separated constraints')
    optimize_parser.add_argument('--workers', type=int, default=None, help='Number of workers')
    optimize_parser.add_argument('--max-iterations', type=int, default=None, help='Max iterations')
    optimize_parser.add_argument('--seed', type=int, default=None, help='Random seed')
    optimize_parser.add_argument('--batch-size', type=int, default=50, help='Batch size')
    optimize_parser.add_argument('--config', help='Additional config (JSON)')
    
    # Walk-forward command
    wf_parser = subparsers.add_parser('walk-forward', help='Run walk-forward analysis')
    wf_parser.add_argument('--strategy', required=True, help='Strategy name')
    wf_parser.add_argument('--symbols', required=True, help='Comma-separated symbols')
    wf_parser.add_argument('--timeframe', default='1 day', help='Timeframe')
    wf_parser.add_argument('--lookback', type=int, default=100, help='Lookback periods')
    wf_parser.add_argument('--params', required=True, help='Parameter ranges (JSON)')
    wf_parser.add_argument('--algorithm', default='grid_search', help='Algorithm')
    wf_parser.add_argument('--objective', default='sharpe_ratio', help='Objective function')
    wf_parser.add_argument('--constraints', help='Comma-separated constraints')
    wf_parser.add_argument('--in-sample-days', type=int, default=180, help='In-sample period (days)')
    wf_parser.add_argument('--out-sample-days', type=int, default=60, help='Out-of-sample period (days)')
    wf_parser.add_argument('--step-days', type=int, default=60, help='Step size (days)')
    wf_parser.add_argument('--anchored', action='store_true', help='Use anchored windows')
    wf_parser.add_argument('--workers', type=int, default=None, help='Number of workers')
    wf_parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    # Out-of-sample command
    oos_parser = subparsers.add_parser('out-of-sample', help='Run out-of-sample testing')
    oos_parser.add_argument('--strategy', required=True, help='Strategy name')
    oos_parser.add_argument('--symbols', required=True, help='Comma-separated symbols')
    oos_parser.add_argument('--timeframe', default='1 day', help='Timeframe')
    oos_parser.add_argument('--lookback', type=int, default=100, help='Lookback periods')
    oos_parser.add_argument('--params', required=True, help='Parameter ranges (JSON)')
    oos_parser.add_argument('--algorithm', default='grid_search', help='Algorithm')
    oos_parser.add_argument('--objective', default='sharpe_ratio', help='Objective function')
    oos_parser.add_argument('--constraints', help='Comma-separated constraints')
    oos_parser.add_argument('--train-ratio', type=float, default=0.7, help='Training set ratio (0-1)')
    oos_parser.add_argument('--workers', type=int, default=None, help='Number of workers')
    oos_parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    # Cross-validation command
    cv_parser = subparsers.add_parser('cross-validate', help='Run cross-validation')
    cv_parser.add_argument('--strategy', required=True, help='Strategy name')
    cv_parser.add_argument('--symbols', required=True, help='Comma-separated symbols')
    cv_parser.add_argument('--timeframe', default='1 day', help='Timeframe')
    cv_parser.add_argument('--lookback', type=int, default=100, help='Lookback periods')
    cv_parser.add_argument('--params', required=True, help='Parameter ranges (JSON)')
    cv_parser.add_argument('--algorithm', default='grid_search', help='Algorithm')
    cv_parser.add_argument('--objective', default='sharpe_ratio', help='Objective function')
    cv_parser.add_argument('--constraints', help='Comma-separated constraints')
    cv_parser.add_argument('--n-splits', type=int, default=5, help='Number of folds')
    cv_parser.add_argument('--test-ratio', type=float, default=0.2, help='Test set ratio (0-1)')
    cv_parser.add_argument('--purge-days', type=int, default=0, help='Purge days')
    cv_parser.add_argument('--embargo-days', type=int, default=0, help='Embargo days')
    cv_parser.add_argument('--workers', type=int, default=None, help='Number of workers')
    cv_parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    # API command
    api_parser = subparsers.add_parser('api', help='Start REST API server')
    api_parser.add_argument('--host', default='0.0.0.0', help='Host to bind')
    api_parser.add_argument('--port', type=int, default=8006, help='Port to bind')
    
    args = parser.parse_args()
    
    if args.command == 'optimize':
        return cli_optimize(args)
    elif args.command == 'walk-forward':
        return cli_walk_forward(args)
    elif args.command == 'out-of-sample':
        return cli_out_of_sample(args)
    elif args.command == 'cross-validate':
        return cli_cross_validate(args)
    elif args.command == 'api':
        import uvicorn
        logger.info(f"Starting Optimizer API on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    from datetime import datetime, timezone
    sys.exit(main())

