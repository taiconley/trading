"""
Export functionality for optimization results.

Supports exporting results to CSV, JSON, and generating summary reports.
"""

import csv
import json
import logging
from io import StringIO
from typing import Dict, Any, List, Optional
from datetime import datetime

from common.db import get_db_session
from common.models import OptimizationRun, OptimizationResult, ParameterSensitivity

logger = logging.getLogger(__name__)


class ResultExporter:
    """Export optimization results in various formats."""
    
    def __init__(self, run_id: int):
        """
        Initialize exporter for a specific optimization run.
        
        Args:
            run_id: Optimization run ID to export
        """
        self.run_id = run_id
        self.run_info: Optional[OptimizationRun] = None
        self.results: List[OptimizationResult] = []
    
    def load_data(self) -> None:
        """Load optimization data from database."""
        with get_db_session() as db:
            # Load run info
            self.run_info = db.query(OptimizationRun).filter(
                OptimizationRun.id == self.run_id
            ).first()
            
            if not self.run_info:
                raise ValueError(f"Optimization run {self.run_id} not found")
            
            # Load all results
            self.results = db.query(OptimizationResult).filter(
                OptimizationResult.run_id == self.run_id
            ).order_by(OptimizationResult.score.desc()).all()
            
            logger.info(f"Loaded {len(self.results)} results for export")
    
    def to_csv(self, top_n: Optional[int] = None) -> str:
        """
        Export results to CSV format.
        
        Args:
            top_n: If specified, only export top N results
        
        Returns:
            CSV string
        """
        if not self.results:
            self.load_data()
        
        output = StringIO()
        
        # Determine parameter names from first result
        if not self.results:
            return "No results to export\n"
        
        param_names = sorted(self.results[0].params_json.keys())
        
        # Write header
        fieldnames = (
            ['rank', 'score'] +
            param_names +
            ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate', 'profit_factor', 'total_trades']
        )
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write results
        results_to_export = self.results[:top_n] if top_n else self.results
        for rank, result in enumerate(results_to_export, start=1):
            row = {
                'rank': rank,
                'score': float(result.score) if result.score else None
            }
            # Add parameters
            for param in param_names:
                row[param] = result.params_json.get(param)
            # Add metrics
            row['sharpe_ratio'] = float(result.sharpe_ratio) if result.sharpe_ratio else None
            row['total_return'] = float(result.total_return) if result.total_return else None
            row['max_drawdown'] = float(result.max_drawdown) if result.max_drawdown else None
            row['win_rate'] = float(result.win_rate) if result.win_rate else None
            row['profit_factor'] = float(result.profit_factor) if result.profit_factor else None
            row['total_trades'] = result.total_trades
            
            writer.writerow(row)
        
        return output.getvalue()
    
    def to_json(self, top_n: Optional[int] = None, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Export results to JSON format.
        
        Args:
            top_n: If specified, only export top N results
            include_metadata: Whether to include run metadata
        
        Returns:
            Dictionary suitable for JSON serialization
        """
        if not self.results:
            self.load_data()
        
        output = {}
        
        # Add metadata if requested
        if include_metadata and self.run_info:
            output['metadata'] = {
                'run_id': self.run_info.id,
                'strategy_name': self.run_info.strategy_name,
                'algorithm': self.run_info.algorithm,
                'symbols': self.run_info.symbols,
                'timeframe': self.run_info.timeframe,
                'param_ranges': self.run_info.param_ranges,
                'objective': self.run_info.objective,
                'status': self.run_info.status,
                'total_combinations': self.run_info.total_combinations,
                'completed_combinations': self.run_info.completed_combinations,
                'best_params': self.run_info.best_params,
                'best_score': float(self.run_info.best_score) if self.run_info.best_score else None,
                'start_time': self.run_info.start_time.isoformat() if self.run_info.start_time else None,
                'end_time': self.run_info.end_time.isoformat() if self.run_info.end_time else None,
                'duration_seconds': (
                    (self.run_info.end_time - self.run_info.start_time).total_seconds()
                    if self.run_info.start_time and self.run_info.end_time else None
                ),
                'created_at': self.run_info.created_at.isoformat()
            }
        
        # Add results
        results_to_export = self.results[:top_n] if top_n else self.results
        output['results'] = []
        
        for rank, result in enumerate(results_to_export, start=1):
            result_dict = {
                'rank': rank,
                'params': result.params_json,
                'score': float(result.score) if result.score else None,
                'metrics': {
                    'sharpe_ratio': float(result.sharpe_ratio) if result.sharpe_ratio else None,
                    'total_return': float(result.total_return) if result.total_return else None,
                    'max_drawdown': float(result.max_drawdown) if result.max_drawdown else None,
                    'win_rate': float(result.win_rate) if result.win_rate else None,
                    'profit_factor': float(result.profit_factor) if result.profit_factor else None,
                    'total_trades': result.total_trades
                },
                'backtest_run_id': result.backtest_run_id
            }
            output['results'].append(result_dict)
        
        output['total_results'] = len(self.results)
        output['exported_results'] = len(results_to_export)
        
        return output
    
    def to_summary_report(self) -> str:
        """
        Generate a human-readable summary report.
        
        Returns:
            Formatted text report
        """
        if not self.results:
            self.load_data()
        
        if not self.run_info:
            return "No run information available"
        
        lines = []
        lines.append("=" * 80)
        lines.append("OPTIMIZATION SUMMARY REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Run info
        lines.append(f"Run ID: {self.run_info.id}")
        lines.append(f"Strategy: {self.run_info.strategy_name}")
        lines.append(f"Algorithm: {self.run_info.algorithm}")
        lines.append(f"Symbols: {', '.join(self.run_info.symbols)}")
        lines.append(f"Timeframe: {self.run_info.timeframe}")
        lines.append(f"Objective: {self.run_info.objective}")
        lines.append(f"Status: {self.run_info.status}")
        lines.append("")
        
        # Progress
        lines.append(f"Combinations Tested: {self.run_info.completed_combinations}")
        if self.run_info.total_combinations:
            pct = (self.run_info.completed_combinations / self.run_info.total_combinations) * 100
            lines.append(f"Progress: {pct:.1f}% ({self.run_info.completed_combinations}/{self.run_info.total_combinations})")
        lines.append("")
        
        # Timing
        if self.run_info.start_time and self.run_info.end_time:
            duration = (self.run_info.end_time - self.run_info.start_time).total_seconds()
            lines.append(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            if self.run_info.completed_combinations > 0:
                time_per_combo = duration / self.run_info.completed_combinations
                lines.append(f"Time per combination: {time_per_combo:.2f} seconds")
        lines.append("")
        
        # Best result
        if self.run_info.best_params and self.run_info.best_score:
            lines.append("BEST PARAMETERS FOUND")
            lines.append("-" * 80)
            lines.append(f"Best {self.run_info.objective}: {float(self.run_info.best_score):.4f}")
            lines.append("")
            lines.append("Parameters:")
            for key, value in sorted(self.run_info.best_params.items()):
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        # Top 10 results
        if self.results:
            lines.append("TOP 10 RESULTS")
            lines.append("-" * 80)
            lines.append(f"{'Rank':<6} {'Score':<12} {'Params'}")
            lines.append("-" * 80)
            
            for rank, result in enumerate(self.results[:10], start=1):
                params_str = ", ".join(f"{k}={v}" for k, v in sorted(result.params_json.items()))
                lines.append(f"{rank:<6} {float(result.score):<12.4f} {params_str}")
            
            lines.append("")
        
        # Parameter sensitivity (if available)
        with get_db_session() as db:
            sensitivity = db.query(ParameterSensitivity).filter(
                ParameterSensitivity.run_id == self.run_id
            ).order_by(ParameterSensitivity.importance_rank).all()
            
            if sensitivity:
                lines.append("PARAMETER IMPORTANCE")
                lines.append("-" * 80)
                lines.append(f"{'Rank':<6} {'Parameter':<30} {'Sensitivity':<15} {'Correlation'}")
                lines.append("-" * 80)
                
                for s in sensitivity:
                    lines.append(
                        f"{s.importance_rank:<6} {s.parameter_name:<30} "
                        f"{float(s.sensitivity_score):<15.4f} {float(s.correlation_with_objective):.4f}"
                    )
                lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


def export_results_csv(run_id: int, top_n: Optional[int] = None) -> str:
    """
    Convenience function to export results as CSV.
    
    Args:
        run_id: Optimization run ID
        top_n: Number of top results to export
    
    Returns:
        CSV string
    """
    exporter = ResultExporter(run_id)
    return exporter.to_csv(top_n=top_n)


def export_results_json(
    run_id: int,
    top_n: Optional[int] = None,
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to export results as JSON.
    
    Args:
        run_id: Optimization run ID
        top_n: Number of top results to export
        include_metadata: Whether to include run metadata
    
    Returns:
        Dictionary suitable for JSON serialization
    """
    exporter = ResultExporter(run_id)
    return exporter.to_json(top_n=top_n, include_metadata=include_metadata)


def generate_summary_report(run_id: int) -> str:
    """
    Convenience function to generate summary report.
    
    Args:
        run_id: Optimization run ID
    
    Returns:
        Formatted text report
    """
    exporter = ResultExporter(run_id)
    return exporter.to_summary_report()

