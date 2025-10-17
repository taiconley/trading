"""
Advanced analytics for optimization results.

Provides parameter sensitivity analysis, Pareto frontier analysis,
and other advanced analytics for understanding optimization results.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from scipy.spatial import distance
from sqlalchemy.orm import Session

from common.db import get_db_session
from common.models import OptimizationRun, OptimizationResult, ParameterSensitivity

logger = logging.getLogger(__name__)


@dataclass
class SensitivityMetrics:
    """Metrics for a single parameter's sensitivity."""
    parameter_name: str
    sensitivity_score: float  # Normalized variance or effect size
    correlation: float  # Correlation with objective
    importance_rank: int  # 1 = most important
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    interactions: Dict[str, float]  # Interactions with other parameters
    analysis_data: Dict[str, Any]  # Raw data for visualization


class ParameterSensitivityAnalyzer:
    """
    Analyzes parameter sensitivity from optimization results.
    
    Uses multiple techniques:
    1. Variance-based sensitivity (how much variance each parameter explains)
    2. Correlation analysis (how parameter values correlate with objective)
    3. Interaction effects (how parameters interact with each other)
    """
    
    def __init__(self, run_id: int):
        """
        Initialize analyzer for a specific optimization run.
        
        Args:
            run_id: Optimization run ID to analyze
        """
        self.run_id = run_id
        self.results_df: Optional[pd.DataFrame] = None
        self.sensitivity_metrics: Dict[str, SensitivityMetrics] = {}
    
    def load_results(self, db: Session) -> None:
        """Load optimization results into DataFrame."""
        # Get run info
        run = db.query(OptimizationRun).filter(OptimizationRun.id == self.run_id).first()
        if not run:
            raise ValueError(f"Optimization run {self.run_id} not found")
        
        # Load all results
        results = db.query(OptimizationResult).filter(
            OptimizationResult.run_id == self.run_id
        ).all()
        
        if not results:
            raise ValueError(f"No results found for optimization run {self.run_id}")
        
        # Convert to DataFrame
        data = []
        for result in results:
            row = {'score': float(result.score)}
            row.update(result.params_json)
            data.append(row)
        
        self.results_df = pd.DataFrame(data)
        logger.info(f"Loaded {len(self.results_df)} results for sensitivity analysis")
    
    def analyze(self, db: Session) -> Dict[str, SensitivityMetrics]:
        """
        Perform complete sensitivity analysis.
        
        Args:
            db: Database session
            
        Returns:
            Dictionary mapping parameter name to SensitivityMetrics
        """
        logger.info(f"Starting sensitivity analysis for run {self.run_id}")
        
        # Load results if not already loaded
        if self.results_df is None:
            self.load_results(db)
        
        # Get parameter names (exclude 'score')
        param_names = [col for col in self.results_df.columns if col != 'score']
        
        if not param_names:
            logger.warning("No parameters found in results")
            return {}
        
        # Calculate metrics for each parameter
        for param_name in param_names:
            metrics = self._analyze_parameter(param_name, param_names)
            self.sensitivity_metrics[param_name] = metrics
        
        # Rank parameters by importance
        self._rank_parameters()
        
        # Store results in database
        self._store_results(db)
        
        logger.info(f"Sensitivity analysis complete for {len(param_names)} parameters")
        return self.sensitivity_metrics
    
    def _analyze_parameter(
        self,
        param_name: str,
        all_param_names: List[str]
    ) -> SensitivityMetrics:
        """Analyze sensitivity for a single parameter."""
        param_values = self.results_df[param_name].values
        scores = self.results_df['score'].values
        
        # Basic statistics
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        
        # Correlation with objective
        correlation = float(stats.pearsonr(param_values, scores)[0])
        if np.isnan(correlation):
            correlation = 0.0
        
        # Variance-based sensitivity (how much variance this parameter explains)
        # Use coefficient of determination (R²) from simple linear regression
        if len(np.unique(param_values)) > 1:
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(param_values, scores)
                r_squared = r_value ** 2
                sensitivity_score = float(r_squared)
            except:
                sensitivity_score = abs(correlation)
        else:
            # Parameter is constant
            sensitivity_score = 0.0
        
        # Interaction effects with other parameters
        interactions = {}
        for other_param in all_param_names:
            if other_param != param_name:
                other_values = self.results_df[other_param].values
                if len(np.unique(other_values)) > 1 and len(np.unique(param_values)) > 1:
                    try:
                        corr = float(stats.pearsonr(param_values, other_values)[0])
                        if not np.isnan(corr) and abs(corr) > 0.3:  # Only store significant interactions
                            interactions[other_param] = corr
                    except:
                        pass
        
        # Collect raw data for visualization
        analysis_data = {
            'values': param_values.tolist(),
            'scores': scores.tolist(),
            'unique_values': sorted([float(x) for x in np.unique(param_values)]),
            'score_by_value': self._get_score_by_value(param_name)
        }
        
        return SensitivityMetrics(
            parameter_name=param_name,
            sensitivity_score=sensitivity_score,
            correlation=correlation,
            importance_rank=0,  # Will be set during ranking
            mean_score=mean_score,
            std_score=std_score,
            min_score=min_score,
            max_score=max_score,
            interactions=interactions,
            analysis_data=analysis_data
        )
    
    def _get_score_by_value(self, param_name: str) -> Dict[str, Any]:
        """Calculate mean/std score for each unique parameter value."""
        grouped = self.results_df.groupby(param_name)['score'].agg(['mean', 'std', 'count'])
        return {
            'values': [float(x) for x in grouped.index.tolist()],
            'mean_scores': [float(x) for x in grouped['mean'].tolist()],
            'std_scores': [float(x) if not np.isnan(x) else 0.0 for x in grouped['std'].tolist()],
            'counts': [int(x) for x in grouped['count'].tolist()]
        }
    
    def _rank_parameters(self) -> None:
        """Rank parameters by importance."""
        # Sort by sensitivity score (descending)
        sorted_params = sorted(
            self.sensitivity_metrics.values(),
            key=lambda m: m.sensitivity_score,
            reverse=True
        )
        
        # Assign ranks
        for rank, metrics in enumerate(sorted_params, start=1):
            metrics.importance_rank = rank
    
    def _store_results(self, db: Session) -> None:
        """Store sensitivity analysis results in database."""
        # Delete existing sensitivity results for this run
        db.query(ParameterSensitivity).filter(
            ParameterSensitivity.run_id == self.run_id
        ).delete()
        
        # Insert new results
        for metrics in self.sensitivity_metrics.values():
            record = ParameterSensitivity(
                run_id=self.run_id,
                parameter_name=metrics.parameter_name,
                sensitivity_score=metrics.sensitivity_score,
                correlation_with_objective=metrics.correlation,
                importance_rank=metrics.importance_rank,
                mean_score=metrics.mean_score,
                std_score=metrics.std_score,
                min_score=metrics.min_score,
                max_score=metrics.max_score,
                interactions=metrics.interactions,
                analysis_data=metrics.analysis_data
            )
            db.add(record)
        
        db.commit()
        logger.info(f"Stored sensitivity analysis for {len(self.sensitivity_metrics)} parameters")
    
    def get_top_parameters(self, n: int = 5) -> List[SensitivityMetrics]:
        """Get top N most important parameters."""
        sorted_params = sorted(
            self.sensitivity_metrics.values(),
            key=lambda m: m.importance_rank
        )
        return sorted_params[:n]
    
    def plot_summary(self) -> Dict[str, Any]:
        """
        Generate plot data for sensitivity summary.
        
        Returns dict suitable for JSON serialization with plot data.
        """
        if not self.sensitivity_metrics:
            return {}
        
        sorted_params = sorted(
            self.sensitivity_metrics.values(),
            key=lambda m: m.importance_rank
        )
        
        return {
            'parameter_names': [m.parameter_name for m in sorted_params],
            'sensitivity_scores': [m.sensitivity_score for m in sorted_params],
            'correlations': [m.correlation for m in sorted_params],
            'importance_ranks': [m.importance_rank for m in sorted_params]
        }


class ParetoFrontierAnalyzer:
    """
    Analyzes Pareto frontier for multi-objective optimization.
    
    Identifies non-dominated solutions where improving one objective
    requires sacrificing another.
    """
    
    def __init__(self, run_id: int):
        """
        Initialize analyzer for a specific optimization run.
        
        Args:
            run_id: Optimization run ID to analyze
        """
        self.run_id = run_id
        self.results_df: Optional[pd.DataFrame] = None
        self.pareto_solutions: List[Dict[str, Any]] = []
    
    def load_results(self, db: Session, objectives: List[str]) -> None:
        """
        Load optimization results with multiple objectives.
        
        Args:
            db: Database session
            objectives: List of objective names (e.g., ['sharpe_ratio', 'max_drawdown'])
        """
        # Load all results
        results = db.query(OptimizationResult).filter(
            OptimizationResult.run_id == self.run_id
        ).all()
        
        if not results:
            raise ValueError(f"No results found for optimization run {self.run_id}")
        
        # Convert to DataFrame
        data = []
        for result in results:
            row = {
                'id': result.id,
                'params': result.params_json,
                'score': float(result.score)
            }
            # Add requested objectives
            for obj in objectives:
                if hasattr(result, obj):
                    value = getattr(result, obj)
                    row[obj] = float(value) if value is not None else None
            data.append(row)
        
        self.results_df = pd.DataFrame(data)
        logger.info(f"Loaded {len(self.results_df)} results for Pareto analysis")
    
    def find_pareto_frontier(
        self,
        objectives: List[str],
        maximize: List[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Find Pareto-optimal solutions.
        
        Args:
            objectives: List of objective columns to consider
            maximize: List of booleans indicating whether to maximize (True) or minimize (False)
                     each objective. Default is True for all.
        
        Returns:
            List of Pareto-optimal solution dictionaries
        """
        if maximize is None:
            maximize = [True] * len(objectives)
        
        if len(maximize) != len(objectives):
            raise ValueError("maximize must have same length as objectives")
        
        # Extract objective values
        obj_values = self.results_df[objectives].values
        
        # Handle missing values
        valid_mask = ~np.isnan(obj_values).any(axis=1)
        obj_values = obj_values[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        if len(obj_values) == 0:
            logger.warning("No valid objective values for Pareto analysis")
            return []
        
        # Convert minimization to maximization by negating
        for i, is_max in enumerate(maximize):
            if not is_max:
                obj_values[:, i] = -obj_values[:, i]
        
        # Find Pareto frontier
        pareto_mask = self._is_pareto_efficient(obj_values)
        pareto_indices = valid_indices[pareto_mask]
        
        # Collect Pareto solutions
        self.pareto_solutions = []
        for idx in pareto_indices:
            row = self.results_df.iloc[idx]
            solution = {
                'id': int(row['id']),
                'params': row['params'],
                'score': float(row['score']),
                'objectives': {obj: float(row[obj]) for obj in objectives if obj in row and not pd.isna(row[obj])}
            }
            self.pareto_solutions.append(solution)
        
        logger.info(f"Found {len(self.pareto_solutions)} Pareto-optimal solutions")
        return self.pareto_solutions
    
    def _is_pareto_efficient(self, costs: np.ndarray) -> np.ndarray:
        """
        Find the Pareto-efficient points (maximization).
        
        Args:
            costs: An (n_points, n_costs) array
        
        Returns:
            Boolean array of which points are Pareto-efficient
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # Remove dominated points
                is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)
                is_efficient[i] = True  # Keep self
        return is_efficient
    
    def get_frontier_plot_data(
        self,
        obj1: str,
        obj2: str
    ) -> Dict[str, Any]:
        """
        Get plot data for 2D Pareto frontier visualization.
        
        Args:
            obj1: First objective name (x-axis)
            obj2: Second objective name (y-axis)
        
        Returns:
            Dictionary with plot data
        """
        if self.results_df is None:
            return {}
        
        # All points
        all_points = self.results_df[[obj1, obj2]].dropna()
        
        # Pareto points
        pareto_ids = [s['id'] for s in self.pareto_solutions]
        pareto_mask = self.results_df['id'].isin(pareto_ids)
        pareto_points = self.results_df[pareto_mask][[obj1, obj2]].dropna()
        
        return {
            'all_points': {
                obj1: all_points[obj1].tolist(),
                obj2: all_points[obj2].tolist()
            },
            'pareto_points': {
                obj1: pareto_points[obj1].tolist(),
                obj2: pareto_points[obj2].tolist()
            },
            'objectives': [obj1, obj2],
            'n_total': len(all_points),
            'n_pareto': len(pareto_points)
        }


def analyze_parameter_sensitivity(run_id: int) -> Dict[str, Any]:
    """
    Convenience function to analyze parameter sensitivity.
    
    Args:
        run_id: Optimization run ID
    
    Returns:
        Dictionary with sensitivity metrics
    """
    with get_db_session() as db:
        analyzer = ParameterSensitivityAnalyzer(run_id)
        analyzer.load_results(db)
        metrics = analyzer.analyze(db)
        
        return {
            'run_id': run_id,
            'parameters': {
                name: {
                    'sensitivity_score': m.sensitivity_score,
                    'correlation': m.correlation,
                    'importance_rank': m.importance_rank,
                    'mean_score': m.mean_score,
                    'std_score': m.std_score,
                    'interactions': m.interactions
                }
                for name, m in metrics.items()
            },
            'top_parameters': [
                {
                    'name': m.parameter_name,
                    'sensitivity_score': m.sensitivity_score,
                    'correlation': m.correlation,
                    'rank': m.importance_rank
                }
                for m in analyzer.get_top_parameters(5)
            ],
            'plot_data': analyzer.plot_summary()
        }


def analyze_pareto_frontier(
    run_id: int,
    objectives: List[str],
    maximize: List[bool] = None
) -> Dict[str, Any]:
    """
    Convenience function to analyze Pareto frontier.
    
    Args:
        run_id: Optimization run ID
        objectives: List of objective names
        maximize: List of booleans for each objective
    
    Returns:
        Dictionary with Pareto frontier analysis
    """
    with get_db_session() as db:
        analyzer = ParetoFrontierAnalyzer(run_id)
        analyzer.load_results(db, objectives)
        solutions = analyzer.find_pareto_frontier(objectives, maximize)
        
        # Get plot data for first two objectives if available
        plot_data = {}
        if len(objectives) >= 2:
            plot_data = analyzer.get_frontier_plot_data(objectives[0], objectives[1])
        
        return {
            'run_id': run_id,
            'objectives': objectives,
            'n_solutions': len(solutions),
            'pareto_solutions': solutions,
            'plot_data': plot_data
        }

