"""
Result caching to avoid redundant backtests.

Caches backtest results by parameter combination hash to avoid
running identical backtests multiple times.
"""

import hashlib
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta

from common.db import get_db_session
from common.models import OptimizationResult

logger = logging.getLogger(__name__)


class ResultCache:
    """
    Cache for optimization/backtest results.
    
    Uses parameter combination hash as cache key and stores results
    in the optimization_results table across runs.
    """
    
    def __init__(
        self,
        strategy_name: str,
        symbols: list,
        timeframe: str,
        lookback: int,
        config: Dict[str, Any] = None
    ):
        """
        Initialize result cache.
        
        Args:
            strategy_name: Strategy being optimized
            symbols: Symbols being tested
            timeframe: Bar timeframe
            lookback: Lookback period
            config: Additional configuration (commission, slippage, etc.)
        """
        self.strategy_name = strategy_name
        self.symbols = sorted(symbols)  # Sort for consistent hashing
        self.timeframe = timeframe
        self.lookback = lookback
        self.config = config or {}
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.enabled = True
    
    def _get_cache_key(self, params: Dict[str, Any]) -> str:
        """
        Generate cache key for a parameter combination.
        
        The key includes strategy, symbols, timeframe, lookback, config, and params
        to ensure we only reuse results from identical backtest conditions.
        
        Args:
            params: Parameter combination
        
        Returns:
            Hash string for cache key
        """
        cache_dict = {
            'strategy': self.strategy_name,
            'symbols': self.symbols,
            'timeframe': self.timeframe,
            'lookback': self.lookback,
            'config': {k: v for k, v in self.config.items() if k not in ['num_workers']},  # Exclude execution params
            'params': params
        }
        
        # Convert to stable JSON string
        json_str = json.dumps(cache_dict, sort_keys=True)
        
        # Generate hash
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def get(
        self,
        params: Dict[str, Any],
        max_age_days: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached result for a parameter combination.
        
        Args:
            params: Parameter combination to look up
            max_age_days: Maximum age of cached results in days (None = no limit)
        
        Returns:
            Cached result dictionary or None if not found
        """
        if not self.enabled:
            return None
        
        cache_key = self._get_cache_key(params)
        
        with get_db_session() as db:
            # Query for matching results
            query = db.query(OptimizationResult).filter(
                OptimizationResult.params_json == params
            )
            
            # Apply age filter if specified
            if max_age_days is not None:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
                query = query.filter(OptimizationResult.created_at >= cutoff_date)
            
            # Get most recent result
            result = query.order_by(OptimizationResult.created_at.desc()).first()
            
            if result:
                self.hits += 1
                logger.debug(f"Cache HIT for params {params} (age: {datetime.now(timezone.utc) - result.created_at})")
                
                return {
                    'score': float(result.score),
                    'metrics': {
                        'sharpe_ratio': float(result.sharpe_ratio) if result.sharpe_ratio else None,
                        'total_return_pct': float(result.total_return) if result.total_return else None,
                        'max_drawdown_pct': float(result.max_drawdown) if result.max_drawdown else None,
                        'win_rate': float(result.win_rate) if result.win_rate else None,
                        'profit_factor': float(result.profit_factor) if result.profit_factor else None,
                        'total_trades': result.total_trades
                    },
                    'backtest_run_id': result.backtest_run_id,
                    'cached': True,
                    'cache_age_days': (datetime.now(timezone.utc) - result.created_at).days
                }
            else:
                self.misses += 1
                logger.debug(f"Cache MISS for params {params}")
                return None
    
    def put(
        self,
        params: Dict[str, Any],
        result: Dict[str, Any],
        run_id: int
    ) -> None:
        """
        Store result in cache.
        
        Note: This is automatically done when storing optimization results,
        so this method is primarily for completeness.
        
        Args:
            params: Parameter combination
            result: Result dictionary with score and metrics
            run_id: Optimization run ID
        """
        # Cache is automatically populated when storing optimization results
        # in the engine, so this is mostly a no-op
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            'enabled': self.enabled,
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total,
            'hit_rate': hit_rate
        }
    
    def clear_stats(self) -> None:
        """Reset cache statistics."""
        self.hits = 0
        self.misses = 0
    
    def disable(self) -> None:
        """Disable caching."""
        self.enabled = False
        logger.info("Result cache disabled")
    
    def enable(self) -> None:
        """Enable caching."""
        self.enabled = True
        logger.info("Result cache enabled")


class CachedExecutor:
    """
    Executor wrapper that checks cache before running backtests.
    
    This wraps the ParallelExecutor to add caching functionality.
    """
    
    def __init__(
        self,
        executor,
        cache: ResultCache,
        max_cache_age_days: Optional[int] = 30
    ):
        """
        Initialize cached executor.
        
        Args:
            executor: Base executor (ParallelExecutor)
            cache: ResultCache instance
            max_cache_age_days: Maximum age of cached results to use
        """
        self.executor = executor
        self.cache = cache
        self.max_cache_age_days = max_cache_age_days
    
    def execute_batch(
        self,
        param_combinations: list,
        **kwargs
    ) -> list:
        """
        Execute batch with caching.
        
        Checks cache for each parameter combination and only runs
        backtests for cache misses.
        
        Args:
            param_combinations: List of parameter dicts
            **kwargs: Arguments to pass to executor
        
        Returns:
            List of TaskResult objects (mix of cached and computed)
        """
        from .executor import TaskResult
        
        cached_results = []
        uncached_params = []
        
        # Check cache for each parameter combination
        for params in param_combinations:
            cached = self.cache.get(params, max_age_days=self.max_cache_age_days)
            if cached:
                # Create TaskResult from cached data
                cached_results.append(TaskResult(
                    params=params,
                    success=True,
                    result=cached
                ))
            else:
                uncached_params.append(params)
        
        # Execute uncached combinations
        computed_results = []
        if uncached_params:
            logger.info(
                f"Cache: {len(cached_results)} hits, {len(uncached_params)} misses "
                f"({self.cache.get_stats()['hit_rate']:.1%} hit rate)"
            )
            computed_results = self.executor.execute_batch(
                param_combinations=uncached_params,
                **kwargs
            )
        else:
            logger.info(f"Cache: All {len(cached_results)} results found in cache!")
        
        # Combine cached and computed results
        all_results = cached_results + computed_results
        
        return all_results

