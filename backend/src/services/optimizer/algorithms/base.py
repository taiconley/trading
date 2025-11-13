"""Base optimizer class and parameter space definitions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Iterator, Union
import itertools


@dataclass
class ParameterSpace:
    """
    Definition of parameter search space.
    
    Each parameter can be:
    - List of values: [1, 2, 3, 4, 5]
    - Range tuple: (start, stop, step) for numeric ranges
    """
    ranges: Dict[str, Union[List[Any], tuple]]
    
    def validate(self) -> None:
        """Validate parameter space definition."""
        if not self.ranges:
            raise ValueError("Parameter space cannot be empty")
        
        for param_name, param_range in self.ranges.items():
            if isinstance(param_range, list):
                if not param_range:
                    raise ValueError(f"Parameter {param_name} has empty range")
            elif isinstance(param_range, tuple):
                if len(param_range) != 3:
                    raise ValueError(
                        f"Parameter {param_name} range tuple must be (start, stop, step)"
                    )
            elif isinstance(param_range, dict):
                # Special handling for pair_selection: dict of pair_key -> [True, False]
                if param_name == "pair_selection":
                    if not param_range:
                        raise ValueError(f"Parameter {param_name} has empty selection dict")
                    for pair_key, choices in param_range.items():
                        if not isinstance(choices, list) or not choices:
                            raise ValueError(
                                f"Parameter {param_name}[{pair_key}] must be a non-empty list"
                            )
                        if not all(isinstance(c, bool) for c in choices):
                            raise ValueError(
                                f"Parameter {param_name}[{pair_key}] must contain only booleans"
                            )
                else:
                    raise ValueError(
                        f"Parameter {param_name} dict type only supported for 'pair_selection'"
                    )
            else:
                raise ValueError(
                    f"Parameter {param_name} must be list, (start, stop, step) tuple, or dict (for pair_selection)"
                )
    
    def expand_ranges(self) -> Dict[str, List[Any]]:
        """
        Expand all ranges to lists of values.
        
        Converts (start, stop, step) tuples to explicit lists.
        For pair_selection dict, expands to list of selection dicts.
        """
        expanded = {}
        for param_name, param_range in self.ranges.items():
            if isinstance(param_range, list):
                expanded[param_name] = param_range
            elif isinstance(param_range, tuple):
                start, stop, step = param_range
                # Generate range based on type
                if isinstance(start, float) or isinstance(step, float):
                    # Float range
                    values = []
                    val = start
                    while val <= stop:
                        values.append(val)
                        val += step
                    expanded[param_name] = values
                else:
                    # Integer range
                    expanded[param_name] = list(range(start, stop + 1, step))
            elif isinstance(param_range, dict) and param_name == "pair_selection":
                # For pair_selection, we need to generate all combinations of True/False for each pair
                # This creates a list of selection dicts, one for each combination
                import itertools
                pair_keys = list(param_range.keys())
                choice_lists = [param_range[key] for key in pair_keys]
                combinations = list(itertools.product(*choice_lists))
                expanded[param_name] = [
                    {pair_keys[i]: choice for i, choice in enumerate(combo)}
                    for combo in combinations
                ]
            else:
                raise ValueError(f"Invalid parameter range for {param_name}")
        return expanded
    
    def count_combinations(self) -> int:
        """Count total number of parameter combinations."""
        expanded = self.expand_ranges()
        total = 1
        for values in expanded.values():
            total *= len(values)
        return total
    
    def validate_constraints(self, params: Dict[str, Any], constraints: List[str]) -> bool:
        """
        Validate parameter combination against constraints.
        
        Args:
            params: Parameter combination to validate
            constraints: List of constraint expressions (e.g., "short_period < long_period")
        
        Returns:
            True if all constraints are satisfied
        """
        if not constraints:
            return True
        
        for constraint in constraints:
            try:
                # Create a safe evaluation environment with only params
                if not eval(constraint, {"__builtins__": {}}, params):
                    return False
            except Exception:
                # If constraint can't be evaluated, skip it
                continue
        
        return True


@dataclass
class OptimizationResult:
    """Result from evaluating a parameter combination."""
    params: Dict[str, Any]
    score: float
    metrics: Dict[str, Any]
    backtest_run_id: int = None
    
    def __lt__(self, other):
        """Compare by score for sorting."""
        return self.score < other.score
    
    def __le__(self, other):
        return self.score <= other.score
    
    def __gt__(self, other):
        return self.score > other.score
    
    def __ge__(self, other):
        return self.score >= other.score


class BaseOptimizer(ABC):
    """
    Base class for optimization algorithms.
    
    All optimizers must implement:
    - generate_candidates(): Yield parameter combinations to test
    """
    
    def __init__(
        self,
        param_space: ParameterSpace,
        constraints: List[str] = None,
        max_iterations: int = None,
        random_seed: int = None
    ):
        """
        Initialize optimizer.
        
        Args:
            param_space: Parameter space definition
            constraints: List of constraint expressions
            max_iterations: Maximum number of iterations (None for unlimited)
            random_seed: Random seed for reproducibility
        """
        self.param_space = param_space
        self.constraints = constraints or []
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        
        # Validate parameter space
        self.param_space.validate()
        
        # State
        self.iteration = 0
        self.best_result: OptimizationResult = None
        self.all_results: List[OptimizationResult] = []
    
    @abstractmethod
    def generate_candidates(self) -> Iterator[Dict[str, Any]]:
        """
        Generate parameter combinations to test.
        
        Must be implemented by subclasses.
        
        Yields:
            Parameter dictionaries to evaluate
        """
        pass
    
    def should_stop(self) -> bool:
        """
        Check if optimization should stop.
        
        Returns:
            True if stopping criteria met
        """
        if self.max_iterations is not None and self.iteration >= self.max_iterations:
            return True
        return False
    
    def update(self, result: OptimizationResult) -> None:
        """
        Update optimizer state with new result.
        
        Args:
            result: Result from evaluating a parameter combination
        """
        self.iteration += 1
        self.all_results.append(result)
        
        # Update best result
        if self.best_result is None or result.score > self.best_result.score:
            self.best_result = result
    
    def get_best(self) -> OptimizationResult:
        """Get best result found so far."""
        return self.best_result
    
    def get_top_n(self, n: int = 10) -> List[OptimizationResult]:
        """
        Get top N results.
        
        Args:
            n: Number of top results to return
        
        Returns:
            List of top N results sorted by score (descending)
        """
        sorted_results = sorted(self.all_results, key=lambda r: r.score, reverse=True)
        return sorted_results[:n]

