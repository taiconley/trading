"""Random search optimizer - random sampling of parameter space."""

import random
from typing import Dict, Any, Iterator

from .base import BaseOptimizer, ParameterSpace


class RandomSearchOptimizer(BaseOptimizer):
    """
    Random Search optimization algorithm.
    
    Randomly samples parameter combinations from the search space.
    Best for:
    - Large parameter spaces
    - Quick exploration
    - When grid search is too expensive
    
    Often finds good solutions faster than grid search in high-dimensional spaces.
    """
    
    def __init__(
        self,
        param_space: ParameterSpace,
        constraints: list = None,
        max_iterations: int = 100,
        random_seed: int = None
    ):
        """
        Initialize random search optimizer.
        
        Args:
            param_space: Parameter space definition
            constraints: List of constraint expressions
            max_iterations: Number of random samples to try
            random_seed: Random seed for reproducibility
        """
        super().__init__(param_space, constraints, max_iterations, random_seed)
        
        # Set random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)
        
        # Pre-expand parameter ranges
        self.expanded_ranges = self.param_space.expand_ranges()
        
        # Track tested combinations to avoid duplicates
        self.tested_combinations = set()
    
    def generate_candidates(self) -> Iterator[Dict[str, Any]]:
        """
        Generate random parameter combinations.
        
        Yields:
            Random parameter dictionaries
        """
        max_attempts = self.max_iterations * 10  # Prevent infinite loops
        attempts = 0
        
        while not self.should_stop() and attempts < max_attempts:
            attempts += 1
            
            # Generate random combination
            params = {}
            for param_name, param_values in self.expanded_ranges.items():
                params[param_name] = random.choice(param_values)
            
            # Convert to hashable tuple for duplicate checking
            # Handle dict values (like pair_selection) by converting to sorted tuple
            def make_hashable(value):
                if isinstance(value, dict):
                    return tuple(sorted(value.items()))
                return value
            
            params_tuple = tuple(sorted(
                (k, make_hashable(v)) for k, v in params.items()
            ))
            
            # Skip if already tested
            if params_tuple in self.tested_combinations:
                continue
            
            # Check constraints
            if not self.param_space.validate_constraints(params, self.constraints):
                continue
            
            # Mark as tested
            self.tested_combinations.add(params_tuple)
            
            yield params
    
    def get_progress(self) -> float:
        """
        Get optimization progress as percentage.
        
        Returns:
            Progress from 0.0 to 1.0
        """
        if self.max_iterations is None or self.max_iterations == 0:
            return 0.0
        return min(1.0, self.iteration / self.max_iterations)

