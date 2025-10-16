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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Strategy Parameter Optimizer")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Run parameter optimization')
    optimize_parser.add_argument('--strategy', required=True, help='Strategy name')
    optimize_parser.add_argument('--symbols', required=True, help='Comma-separated symbols')
    optimize_parser.add_argument('--timeframe', default='1 day', help='Timeframe')
    optimize_parser.add_argument('--lookback', type=int, default=365, help='Lookback days')
    optimize_parser.add_argument('--params', required=True, help='Parameter ranges (JSON)')
    optimize_parser.add_argument('--algorithm', default='grid_search', 
                                help='Algorithm (grid_search, random_search)')
    optimize_parser.add_argument('--objective', default='sharpe_ratio',
                                help='Objective function (sharpe_ratio, total_return, profit_factor, win_rate)')
    optimize_parser.add_argument('--constraints', help='Comma-separated constraints')
    optimize_parser.add_argument('--workers', type=int, default=None, help='Number of workers')
    optimize_parser.add_argument('--max-iterations', type=int, default=None, help='Max iterations')
    optimize_parser.add_argument('--seed', type=int, default=None, help='Random seed')
    optimize_parser.add_argument('--batch-size', type=int, default=50, help='Batch size')
    optimize_parser.add_argument('--config', help='Additional config (JSON)')
    
    # API command
    api_parser = subparsers.add_parser('api', help='Start REST API server')
    api_parser.add_argument('--host', default='0.0.0.0', help='Host to bind')
    api_parser.add_argument('--port', type=int, default=8006, help='Port to bind')
    
    args = parser.parse_args()
    
    if args.command == 'optimize':
        return cli_optimize(args)
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

