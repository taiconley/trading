"""Bayesian optimization using Optuna framework."""

import optuna
from typing import Dict, Any, Iterator, Optional
import logging

from .base import BaseOptimizer, ParameterSpace

logger = logging.getLogger(__name__)


class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian Optimization using Optuna.
    
    Uses Tree-structured Parzen Estimator (TPE) to intelligently search
    the parameter space by learning from previous evaluations.
    
    Best for:
    - Expensive evaluation functions (long backtests)
    - Finding good parameters quickly
    - Continuous and discrete parameter spaces
    - When you want intelligent exploration vs exploitation
    
    Typically finds good solutions in 20-50 iterations vs hundreds for grid search.
    """
    
    def __init__(
        self,
        param_space: ParameterSpace,
        constraints: list = None,
        max_iterations: int = 50,
        random_seed: int = None,
        n_startup_trials: int = 10,
        multivariate: bool = True
    ):
        """
        Initialize Bayesian optimizer with Optuna.
        
        Args:
            param_space: Parameter space definition
            constraints: List of constraint expressions
            max_iterations: Maximum number of trials
            random_seed: Random seed for reproducibility
            n_startup_trials: Number of random trials before TPE starts (default: 10)
            multivariate: Whether to use multivariate TPE (considers parameter interactions)
        """
        super().__init__(param_space, constraints, max_iterations, random_seed)
        
        self.n_startup_trials = n_startup_trials
        self.multivariate = multivariate
        
        # Create Optuna study
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=n_startup_trials,
            multivariate=multivariate,
            seed=random_seed
        )
        
        # Suppress Optuna's logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        self.study = optuna.create_study(
            direction='maximize',  # We always maximize the objective
            sampler=sampler
        )
        
        # Expand parameter ranges for Optuna
        self.expanded_ranges = self.param_space.expand_ranges()
        
        # Track generated parameters to avoid duplicates
        self.generated_params = []
    
    def _suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest parameters for a trial using Optuna's suggestion API.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Dictionary of suggested parameters
        """
        params = {}
        
        for param_name, param_values in self.expanded_ranges.items():
            # Special handling for pair_selection
            if param_name == "pair_selection":
                # param_values is a list of selection dicts
                # We need to suggest one of them
                selected_dict = trial.suggest_categorical(param_name, param_values)
                params[param_name] = selected_dict
            # Check if parameter is numeric
            elif all(isinstance(v, (int, float)) for v in param_values):
                # Determine if integer or float
                if all(isinstance(v, int) for v in param_values):
                    # Integer parameter
                    params[param_name] = trial.suggest_int(
                        param_name,
                        min(param_values),
                        max(param_values)
                    )
                else:
                    # Float parameter
                    params[param_name] = trial.suggest_float(
                        param_name,
                        min(param_values),
                        max(param_values)
                    )
            else:
                # Categorical parameter
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_values
                )
        
        return params
    
    def _objective_wrapper(self, trial: optuna.Trial) -> float:
        """
        Wrapper for Optuna's objective function.
        
        This is called by Optuna during optimization.
        We don't actually evaluate here - we just generate parameters.
        The actual evaluation happens in the main optimization loop.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Dummy score (will be updated later)
        """
        # Suggest parameters
        params = self._suggest_parameters(trial)
        
        # Check constraints
        if not self.param_space.validate_constraints(params, self.constraints):
            # Prune invalid trial
            raise optuna.TrialPruned()
        
        # Store parameters for later retrieval
        self.generated_params.append(params)
        
        # Return dummy value (will be updated with tell())
        return 0.0
    
    def generate_candidates(self) -> Iterator[Dict[str, Any]]:
        """
        Generate parameter combinations using Bayesian optimization.
        
        Yields:
            Parameter dictionaries suggested by Optuna
        """
        # Track trials to prevent infinite loops if all trials are pruned
        # We allow generous overhead for pruned trials (10x max_iterations)
        # This is necessary when constraints eliminate many candidates
        max_attempts = self.max_iterations * 10 if self.max_iterations else None
        attempts = 0
        
        while not self.should_stop():
            # Safety check to prevent infinite loops
            if max_attempts and attempts >= max_attempts:
                logger.warning(f"Reached maximum attempts ({max_attempts}), stopping")
                break
            
            # Ask Optuna for next trial
            trial = self.study.ask()
            attempts += 1
            
            try:
                # Generate parameters for this trial
                params = self._suggest_parameters(trial)
                
                # Check constraints
                if not self.param_space.validate_constraints(params, self.constraints):
                    # Mark trial as pruned and continue searching
                    self.study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                    continue
                
                # Yield parameters for evaluation
                # The score will be updated via update() method
                self._current_trial = trial
                yield params
                
                # After yielding, update() will be called which increments self.iteration
                # Check if we should stop after this successful evaluation
                if self.should_stop():
                    break
                
            except optuna.TrialPruned:
                # Skip pruned trials
                continue
    
    def update(self, result) -> None:
        """
        Update optimizer state with new result.
        
        Args:
            result: OptimizationResult with params, score, and metrics
        """
        # Update base class state
        super().update(result)
        
        # Tell Optuna about the result
        if hasattr(self, '_current_trial'):
            self.study.tell(self._current_trial, result.score)
            delattr(self, '_current_trial')
    
    def get_progress(self) -> float:
        """
        Get optimization progress as percentage.
        
        Returns:
            Progress from 0.0 to 1.0
        """
        if self.max_iterations is None or self.max_iterations == 0:
            return 0.0
        return min(1.0, self.iteration / self.max_iterations)
    
    def get_best_trials(self, n: int = 10) -> list:
        """
        Get top N trials from Optuna study.
        
        Args:
            n: Number of top trials to return
        
        Returns:
            List of trial objects
        """
        return self.study.best_trials[:n]
    
    def get_optimization_history(self) -> Dict[str, Any]:
        """
        Get optimization history for visualization.
        
        Returns:
            Dictionary with trial history and best values
        """
        trials = self.study.trials
        
        return {
            'trial_numbers': [t.number for t in trials],
            'values': [t.value if t.value is not None else 0.0 for t in trials],
            'best_values': [
                max([t.value for t in trials[:i+1] if t.value is not None] or [0.0])
                for i in range(len(trials))
            ],
            'states': [t.state.name for t in trials]
        }

