"""
Basic tests for Phase 3 optimizer features.

Run these tests with:
    pytest test_phase3.py -v
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Phase 3 modules
try:
    from analytics import (
        ParameterSensitivityAnalyzer,
        ParetoFrontierAnalyzer,
        analyze_parameter_sensitivity,
        analyze_pareto_frontier
    )
    from cache import ResultCache
    from export import ResultExporter, export_results_csv, export_results_json
    from monitoring import ResourceMonitor, OptimizationMonitor, get_system_info
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the optimizer directory or have proper PYTHONPATH")
    raise


class TestParameterSensitivity:
    """Tests for parameter sensitivity analysis."""
    
    def test_sensitivity_analyzer_initialization(self):
        """Test analyzer can be initialized."""
        analyzer = ParameterSensitivityAnalyzer(run_id=1)
        assert analyzer.run_id == 1
        assert analyzer.results_df is None
        assert analyzer.sensitivity_metrics == {}
    
    def test_mock_sensitivity_analysis(self):
        """Test sensitivity analysis with mock data."""
        # Create mock results DataFrame
        analyzer = ParameterSensitivityAnalyzer(run_id=1)
        
        # Simulate some optimization results
        analyzer.results_df = pd.DataFrame({
            'short_period': [5, 10, 15, 20, 5, 10, 15, 20],
            'long_period': [30, 30, 30, 30, 50, 50, 50, 50],
            'score': [0.5, 0.8, 1.2, 0.9, 0.6, 1.0, 1.5, 1.1]
        })
        
        # Analyze parameters
        param_names = ['short_period', 'long_period']
        for param_name in param_names:
            metrics = analyzer._analyze_parameter(param_name, param_names)
            assert metrics.parameter_name == param_name
            assert isinstance(metrics.sensitivity_score, float)
            assert isinstance(metrics.correlation, float)
            assert isinstance(metrics.mean_score, float)
    
    def test_parameter_ranking(self):
        """Test parameter importance ranking."""
        analyzer = ParameterSensitivityAnalyzer(run_id=1)
        
        # Create mock metrics
        from analytics import SensitivityMetrics
        analyzer.sensitivity_metrics = {
            'param_a': SensitivityMetrics(
                parameter_name='param_a',
                sensitivity_score=0.8,
                correlation=0.5,
                importance_rank=0,
                mean_score=1.0,
                std_score=0.2,
                min_score=0.5,
                max_score=1.5,
                interactions={},
                analysis_data={}
            ),
            'param_b': SensitivityMetrics(
                parameter_name='param_b',
                sensitivity_score=0.3,
                correlation=0.2,
                importance_rank=0,
                mean_score=1.0,
                std_score=0.1,
                min_score=0.8,
                max_score=1.2,
                interactions={},
                analysis_data={}
            )
        }
        
        # Rank parameters
        analyzer._rank_parameters()
        
        # Check ranking
        assert analyzer.sensitivity_metrics['param_a'].importance_rank == 1
        assert analyzer.sensitivity_metrics['param_b'].importance_rank == 2


class TestParetoFrontier:
    """Tests for Pareto frontier analysis."""
    
    def test_pareto_analyzer_initialization(self):
        """Test Pareto analyzer can be initialized."""
        analyzer = ParetoFrontierAnalyzer(run_id=1)
        assert analyzer.run_id == 1
        assert analyzer.results_df is None
        assert analyzer.pareto_solutions == []
    
    def test_pareto_efficiency_detection(self):
        """Test Pareto efficient point detection."""
        analyzer = ParetoFrontierAnalyzer(run_id=1)
        
        # Test points (maximize both objectives)
        # Points: (1,1), (2,2), (1.5, 1), (3,3)
        # Only (3,3) is Pareto efficient - it dominates all others
        costs = np.array([
            [1, 1],
            [2, 2],
            [1.5, 1],
            [3, 3]
        ])
        
        pareto_mask = analyzer._is_pareto_efficient(costs)
        
        # Point (3,3) should be Pareto efficient (dominates all)
        assert pareto_mask[3] == True  # (3,3)
        # All others are dominated
        assert pareto_mask[0] == False  # (1,1) dominated
        assert pareto_mask[1] == False  # (2,2) dominated
        assert pareto_mask[2] == False  # (1.5,1) dominated
        
        # Test with non-dominated points
        costs2 = np.array([
            [1, 3],  # High on objective 2, low on 1
            [3, 1],  # High on objective 1, low on 2
            [2, 2]   # Middle - also efficient (balanced trade-off)
        ])
        pareto_mask2 = analyzer._is_pareto_efficient(costs2)
        
        # All three points are Pareto efficient (different trade-offs)
        # (1,3) is best on obj2, (3,1) is best on obj1, (2,2) balances both
        assert pareto_mask2[0] == True   # (1,3) efficient
        assert pareto_mask2[1] == True   # (3,1) efficient
        assert pareto_mask2[2] == True   # (2,2) also efficient (no domination)
        
        # Test with a clearly dominated point
        costs3 = np.array([
            [1, 3],  # Efficient
            [3, 1],  # Efficient
            [1, 1]   # Dominated by both (lower on all objectives)
        ])
        pareto_mask3 = analyzer._is_pareto_efficient(costs3)
        
        assert pareto_mask3[0] == True   # (1,3) efficient
        assert pareto_mask3[1] == True   # (3,1) efficient
        assert pareto_mask3[2] == False  # (1,1) dominated
    
    def test_frontier_plot_data_structure(self):
        """Test plot data structure for frontier."""
        analyzer = ParetoFrontierAnalyzer(run_id=1)
        
        # Mock results
        analyzer.results_df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'sharpe_ratio': [1.0, 1.5, 2.0, 1.2],
            'max_drawdown': [0.15, 0.10, 0.08, 0.12]
        })
        
        analyzer.pareto_solutions = [{'id': 3}]  # Mock Pareto solution
        
        plot_data = analyzer.get_frontier_plot_data('sharpe_ratio', 'max_drawdown')
        
        assert 'all_points' in plot_data
        assert 'pareto_points' in plot_data
        assert 'objectives' in plot_data
        assert plot_data['objectives'] == ['sharpe_ratio', 'max_drawdown']


class TestResultCache:
    """Tests for result caching."""
    
    def test_cache_initialization(self):
        """Test cache can be initialized."""
        cache = ResultCache(
            strategy_name='SMA_Crossover',
            symbols=['AAPL'],
            timeframe='1 day',
            lookback=365
        )
        assert cache.strategy_name == 'SMA_Crossover'
        assert cache.symbols == ['AAPL']
        assert cache.enabled == True
        assert cache.hits == 0
        assert cache.misses == 0
    
    def test_cache_key_generation(self):
        """Test cache key generation is consistent."""
        cache = ResultCache(
            strategy_name='SMA_Crossover',
            symbols=['AAPL'],
            timeframe='1 day',
            lookback=365
        )
        
        params1 = {'short_period': 10, 'long_period': 50}
        params2 = {'long_period': 50, 'short_period': 10}  # Different order
        
        # Same parameters (different order) should produce same key
        key1 = cache._get_cache_key(params1)
        key2 = cache._get_cache_key(params2)
        
        assert key1 == key2
        assert len(key1) == 64  # SHA256 hash length
    
    def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = ResultCache(
            strategy_name='SMA_Crossover',
            symbols=['AAPL'],
            timeframe='1 day',
            lookback=365
        )
        
        cache.hits = 8
        cache.misses = 2
        
        stats = cache.get_stats()
        
        assert stats['hits'] == 8
        assert stats['misses'] == 2
        assert stats['total_requests'] == 10
        assert stats['hit_rate'] == 0.8
        assert stats['enabled'] == True


class TestExport:
    """Tests for export functionality."""
    
    def test_exporter_initialization(self):
        """Test exporter can be initialized."""
        exporter = ResultExporter(run_id=1)
        assert exporter.run_id == 1
        assert exporter.run_info is None
        assert exporter.results == []
    
    def test_csv_export_structure(self):
        """Test CSV export produces valid structure."""
        exporter = ResultExporter(run_id=1)
        
        # Mock data
        from common.models import OptimizationResult
        mock_result = Mock(spec=OptimizationResult)
        mock_result.params_json = {'short_period': 10, 'long_period': 50}
        mock_result.score = 1.5
        mock_result.sharpe_ratio = 1.5
        mock_result.total_return = 25.0
        mock_result.max_drawdown = 0.10
        mock_result.win_rate = 0.6
        mock_result.profit_factor = 2.0
        mock_result.total_trades = 50
        
        exporter.results = [mock_result]
        
        csv_output = exporter.to_csv(top_n=1)
        
        # Check CSV structure
        assert 'rank,score' in csv_output
        assert 'short_period' in csv_output or 'long_period' in csv_output
        assert 'sharpe_ratio' in csv_output
    
    def test_json_export_structure(self):
        """Test JSON export produces valid structure."""
        exporter = ResultExporter(run_id=1)
        
        # Mock run info
        from common.models import OptimizationRun
        mock_run = Mock(spec=OptimizationRun)
        mock_run.id = 1
        mock_run.strategy_name = 'SMA_Crossover'
        mock_run.algorithm = 'grid_search'
        mock_run.symbols = ['AAPL']
        mock_run.timeframe = '1 day'
        mock_run.param_ranges = {'short_period': [5, 10], 'long_period': [30, 50]}
        mock_run.objective = 'sharpe_ratio'
        mock_run.status = 'completed'
        mock_run.total_combinations = 4
        mock_run.completed_combinations = 4
        mock_run.best_params = {'short_period': 10, 'long_period': 50}
        mock_run.best_score = 1.5
        mock_run.start_time = None
        mock_run.end_time = None
        mock_run.created_at = None
        
        exporter.run_info = mock_run
        exporter.results = []
        
        json_output = exporter.to_json(include_metadata=True)
        
        # Check JSON structure
        assert 'metadata' in json_output
        assert 'results' in json_output
        assert json_output['metadata']['strategy_name'] == 'SMA_Crossover'


class TestResourceMonitoring:
    """Tests for resource monitoring."""
    
    def test_monitor_initialization(self):
        """Test resource monitor can be initialized."""
        monitor = ResourceMonitor()
        assert monitor.enabled == True
        assert monitor.start_time is None
        assert monitor.snapshots == []
    
    def test_capture_snapshot(self):
        """Test snapshot capture."""
        monitor = ResourceMonitor()
        monitor.start()
        
        snapshot = monitor.capture_snapshot()
        
        assert snapshot.timestamp is not None
        assert isinstance(snapshot.cpu_percent, float)
        assert isinstance(snapshot.memory_mb, float)
        assert isinstance(snapshot.num_threads, int)
    
    def test_optimization_monitor(self):
        """Test optimization monitor."""
        monitor = OptimizationMonitor(run_id=1, total_combinations=100)
        assert monitor.run_id == 1
        assert monitor.total_combinations == 100
        assert monitor.completed_combinations == 0
    
    def test_system_info(self):
        """Test system info retrieval."""
        info = get_system_info()
        
        assert 'cpu' in info
        assert 'memory' in info
        assert 'disk' in info
        assert 'count' in info['cpu']
        assert 'total_mb' in info['memory']


# Integration test markers
@pytest.mark.integration
class TestPhase3Integration:
    """Integration tests requiring database."""
    
    def test_full_sensitivity_workflow(self):
        """Test full sensitivity analysis workflow (requires DB)."""
        pytest.skip("Integration test - requires database and real optimization run")
    
    def test_full_export_workflow(self):
        """Test full export workflow (requires DB)."""
        pytest.skip("Integration test - requires database and real optimization run")
    
    def test_cache_with_real_backtests(self):
        """Test caching with real backtests (requires DB)."""
        pytest.skip("Integration test - requires database and backtester")


if __name__ == '__main__':
    print("Running Phase 3 optimizer tests...")
    print("=" * 80)
    print("Run with: pytest test_phase3.py -v")
    print("=" * 80)

