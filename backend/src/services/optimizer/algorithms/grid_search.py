"""Grid search optimizer - exhaustive search over parameter grid."""

import itertools
from typing import Dict, Any, Iterator

from .base import BaseOptimizer, ParameterSpace


class GridSearchOptimizer(BaseOptimizer):
    """
    Grid Search optimization algorithm.
    
    Exhaustively tests all combinations in the parameter grid.
    Best for:
    - Small parameter spaces
    - Ensuring comprehensive coverage
    - When computational cost is not a concern
    """
    
    def __init__(
        self,
        param_space: ParameterSpace,
        constraints: list = None,
        max_iterations: int = None,
        random_seed: int = None
    ):
        """
        Initialize grid search optimizer.
        
        Args:
            param_space: Parameter space definition
            constraints: List of constraint expressions
            max_iterations: Maximum number of iterations (limits the grid search)
            random_seed: Not used for grid search (included for interface consistency)
        """
        super().__init__(param_space, constraints, max_iterations, random_seed)
        
        # Pre-expand parameter ranges
        self.expanded_ranges = self.param_space.expand_ranges()
        
        # Calculate total combinations
        self.total_combinations = self.param_space.count_combinations()
    
    def generate_candidates(self) -> Iterator[Dict[str, Any]]:
        """
        Generate all parameter combinations in the grid.
        
        Yields:
            Parameter dictionaries in grid order
        """
        # Get parameter names and values
        param_names = list(self.expanded_ranges.keys())
        param_values = [self.expanded_ranges[name] for name in param_names]
        
        # Generate all combinations using Cartesian product
        for combination in itertools.product(*param_values):
            # Create parameter dictionary
            params = dict(zip(param_names, combination))
            
            # Check constraints
            if self.param_space.validate_constraints(params, self.constraints):
                yield params
            
            # Check if we should stop
            if self.should_stop():
                break
    
    def get_progress(self) -> float:
        """
        Get optimization progress as percentage.
        
        Returns:
            Progress from 0.0 to 1.0
        """
        if self.total_combinations == 0:
            return 1.0
        return min(1.0, self.iteration / self.total_combinations)

