"""
Genetic Algorithm optimizer for parameter optimization.

Implements a classic genetic algorithm with:
- Tournament and roulette wheel selection
- Single-point, two-point, and uniform crossover
- Gaussian and uniform mutation
- Elitism to preserve best solutions
- Convergence tracking
"""

import random
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from .base import BaseOptimizer, OptimizationResult, ParameterSpace

logger = logging.getLogger(__name__)


class GeneticAlgorithmOptimizer(BaseOptimizer):
    """
    Genetic Algorithm optimizer.
    
    Uses evolutionary principles to search parameter space:
    1. Initialize random population
    2. Evaluate fitness (objective function)
    3. Select parents based on fitness
    4. Create offspring via crossover and mutation
    5. Replace population with new generation
    6. Repeat until convergence or max generations
    """
    
    def __init__(
        self,
        param_space: ParameterSpace,
        constraints: List[str] = None,
        max_iterations: int = 100,
        random_seed: Optional[int] = None,
        population_size: int = 50,
        elite_size: int = 5,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        tournament_size: int = 3,
        selection_method: str = 'tournament',
        crossover_method: str = 'uniform',
        convergence_threshold: float = 0.001,
        convergence_generations: int = 10
    ):
        """
        Initialize Genetic Algorithm optimizer.
        
        Args:
            param_space: Parameter space to search
            constraints: List of constraint expressions
            max_iterations: Maximum number of generations
            random_seed: Random seed for reproducibility
            population_size: Number of individuals in population
            elite_size: Number of top individuals to preserve each generation
            mutation_rate: Probability of mutation (0-1)
            crossover_rate: Probability of crossover (0-1)
            tournament_size: Size of tournament for selection
            selection_method: 'tournament' or 'roulette'
            crossover_method: 'single_point', 'two_point', or 'uniform'
            convergence_threshold: Change in best fitness to consider converged
            convergence_generations: Generations without improvement for convergence
        """
        super().__init__(param_space, constraints, max_iterations, random_seed)
        
        self.population_size = population_size
        self.elite_size = min(elite_size, population_size)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = min(tournament_size, population_size)
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.convergence_threshold = convergence_threshold
        self.convergence_generations = convergence_generations
        
        # Population state
        self.population: List[Dict[str, Any]] = []
        self.fitness_scores: List[float] = []
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self.generations_without_improvement = 0
        
        logger.info(
            f"Initialized GA optimizer: pop={population_size}, "
            f"elite={elite_size}, mutation={mutation_rate}, "
            f"crossover={crossover_rate}, method={selection_method}"
        )
    
    def _initialize_population(self) -> List[Dict[str, Any]]:
        """Initialize random population."""
        population = []
        attempts = 0
        max_attempts = self.population_size * 100
        
        while len(population) < self.population_size and attempts < max_attempts:
            individual = self._random_individual()
            if self._satisfies_constraints(individual):
                population.append(individual)
            attempts += 1
        
        if len(population) < self.population_size:
            logger.warning(
                f"Could only generate {len(population)}/{self.population_size} "
                "valid individuals due to constraints"
            )
        
        logger.info(f"Initialized population with {len(population)} individuals")
        return population
    
    def _random_individual(self) -> Dict[str, Any]:
        """Generate a random individual."""
        expanded = self.param_space.expand_ranges()
        individual = {}
        for param_name, values in expanded.items():
            individual[param_name] = random.choice(values)
        return individual
    
    def _satisfies_constraints(self, params: Dict[str, Any]) -> bool:
        """Check if parameter combination satisfies all constraints."""
        return self.param_space.validate_constraints(params, self.constraints)
    
    def _encode_individual(self, individual: Dict[str, Any]) -> List[Any]:
        """Encode individual as a list (gene sequence)."""
        expanded = self.param_space.expand_ranges()
        return [individual[param] for param in sorted(expanded.keys())]
    
    def _decode_individual(self, genes: List[Any]) -> Dict[str, Any]:
        """Decode gene sequence back to parameter dict."""
        expanded = self.param_space.expand_ranges()
        param_names = sorted(expanded.keys())
        return {param_names[i]: genes[i] for i in range(len(genes))}
    
    def _select_parents(self, population: List[Dict[str, Any]], fitness_scores: List[float]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Select two parents for reproduction."""
        if not fitness_scores or len(fitness_scores) != len(population):
            # If no fitness scores yet, do random selection
            return random.choice(population).copy(), random.choice(population).copy()
        
        if self.selection_method == 'tournament':
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
        elif self.selection_method == 'roulette':
            parent1 = self._roulette_selection(population, fitness_scores)
            parent2 = self._roulette_selection(population, fitness_scores)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
        
        return parent1, parent2
    
    def _tournament_selection(self, population: List[Dict[str, Any]], fitness_scores: List[float]) -> Dict[str, Any]:
        """Select individual via tournament selection."""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx].copy()
    
    def _roulette_selection(self, population: List[Dict[str, Any]], fitness_scores: List[float]) -> Dict[str, Any]:
        """Select individual via roulette wheel selection."""
        # Handle negative fitness by shifting
        min_fitness = min(fitness_scores)
        if min_fitness < 0:
            adjusted_fitness = [f - min_fitness + 1 for f in fitness_scores]
        else:
            adjusted_fitness = fitness_scores
        
        total_fitness = sum(adjusted_fitness)
        if total_fitness == 0:
            return random.choice(population).copy()
        
        # Roulette wheel
        spin = random.uniform(0, total_fitness)
        cumulative = 0
        for i, fitness in enumerate(adjusted_fitness):
            cumulative += fitness
            if cumulative >= spin:
                return population[i].copy()
        
        return population[-1].copy()
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform crossover between two parents."""
        if random.random() > self.crossover_rate:
            # No crossover, return copies of parents
            return parent1.copy(), parent2.copy()
        
        genes1 = self._encode_individual(parent1)
        genes2 = self._encode_individual(parent2)
        
        if self.crossover_method == 'single_point':
            child1_genes, child2_genes = self._single_point_crossover(genes1, genes2)
        elif self.crossover_method == 'two_point':
            child1_genes, child2_genes = self._two_point_crossover(genes1, genes2)
        elif self.crossover_method == 'uniform':
            child1_genes, child2_genes = self._uniform_crossover(genes1, genes2)
        else:
            raise ValueError(f"Unknown crossover method: {self.crossover_method}")
        
        child1 = self._decode_individual(child1_genes)
        child2 = self._decode_individual(child2_genes)
        
        return child1, child2
    
    def _single_point_crossover(self, genes1: List[Any], genes2: List[Any]) -> Tuple[List[Any], List[Any]]:
        """Single-point crossover."""
        point = random.randint(1, len(genes1) - 1)
        child1 = genes1[:point] + genes2[point:]
        child2 = genes2[:point] + genes1[point:]
        return child1, child2
    
    def _two_point_crossover(self, genes1: List[Any], genes2: List[Any]) -> Tuple[List[Any], List[Any]]:
        """Two-point crossover."""
        point1 = random.randint(1, len(genes1) - 2)
        point2 = random.randint(point1 + 1, len(genes1) - 1)
        
        child1 = genes1[:point1] + genes2[point1:point2] + genes1[point2:]
        child2 = genes2[:point1] + genes1[point1:point2] + genes2[point2:]
        return child1, child2
    
    def _uniform_crossover(self, genes1: List[Any], genes2: List[Any]) -> Tuple[List[Any], List[Any]]:
        """Uniform crossover."""
        child1 = []
        child2 = []
        for g1, g2 in zip(genes1, genes2):
            if random.random() < 0.5:
                child1.append(g1)
                child2.append(g2)
            else:
                child1.append(g2)
                child2.append(g1)
        return child1, child2
    
    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate an individual."""
        if random.random() > self.mutation_rate:
            return individual
        
        mutated = individual.copy()
        expanded = self.param_space.expand_ranges()
        
        # Mutate one random parameter
        param_name = random.choice(list(expanded.keys()))
        possible_values = expanded[param_name]
        mutated[param_name] = random.choice(possible_values)
        
        return mutated
    
    def _get_elite(self, population: List[Dict[str, Any]], fitness_scores: List[float]) -> List[Dict[str, Any]]:
        """Get elite individuals (top performers)."""
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
        elite_indices = sorted_indices[:self.elite_size]
        return [population[i].copy() for i in elite_indices]
    
    def _check_convergence(self) -> bool:
        """Check if algorithm has converged."""
        if len(self.best_fitness_history) < self.convergence_generations:
            return False
        
        recent_best = self.best_fitness_history[-self.convergence_generations:]
        improvement = max(recent_best) - min(recent_best)
        
        if improvement < self.convergence_threshold:
            logger.info(
                f"Converged: improvement {improvement:.6f} < threshold {self.convergence_threshold} "
                f"over {self.convergence_generations} generations"
            )
            return True
        
        return False
    
    def generate_candidates(self):
        """Generate candidate parameter combinations using genetic algorithm."""
        # Initialize population
        self.population = self._initialize_population()
        
        # Yield initial population for evaluation
        for individual in self.population:
            yield individual
            self.iteration += 1
        
        # Evolution loop
        generation = 1
        while self.iteration < self.max_iterations:
            # Wait for fitness scores to be updated via update() method
            # This is called by the engine after evaluating the population
            
            # Check convergence
            if self._check_convergence():
                logger.info(f"Converged after {generation} generations")
                break
            
            # Create new population
            new_population = []
            
            # Elitism: preserve best individuals
            elite = self._get_elite(self.population, self.fitness_scores)
            new_population.extend(elite)
            
            # Generate offspring to fill rest of population
            while len(new_population) < self.population_size and self.iteration < self.max_iterations:
                # Select parents
                parent1, parent2 = self._select_parents(self.population, self.fitness_scores)
                
                # Crossover
                child1, child2 = self._crossover(parent1, parent2)
                
                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                # Add valid children
                for child in [child1, child2]:
                    if len(new_population) < self.population_size and self._satisfies_constraints(child):
                        new_population.append(child)
                        yield child
                        self.iteration += 1
                        
                        if self.iteration >= self.max_iterations:
                            break
            
            # Update population for next generation
            self.population = new_population
            generation += 1
            
            # Log generation stats
            best_score = self.best.score if hasattr(self, 'best') and self.best else 0.0
            avg_score = np.mean(self.fitness_scores) if self.fitness_scores else 0.0
            logger.info(
                f"Generation {generation}: best={best_score:.4f}, "
                f"avg={avg_score:.4f}, "
                f"pop_size={len(self.population)}"
            )
    
    def update(self, result: OptimizationResult) -> None:
        """Update optimizer state with a new result."""
        super().update(result)
        
        # Track fitness for current population
        self.fitness_scores.append(result.score)
        
        # Check if we've completed a generation
        if len(self.fitness_scores) == len(self.population):
            # Record statistics
            best_fitness = max(self.fitness_scores)
            avg_fitness = np.mean(self.fitness_scores)
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            # Track generations without improvement
            if len(self.best_fitness_history) > 1:
                if best_fitness <= self.best_fitness_history[-2]:
                    self.generations_without_improvement += 1
                else:
                    self.generations_without_improvement = 0
            
            # Reset fitness scores for next generation
            self.fitness_scores = []
    
    def get_progress(self) -> float:
        """Get optimization progress (0.0 to 1.0)."""
        if self.max_iterations is None:
            return 0.0
        return min(1.0, self.iteration / self.max_iterations)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get algorithm statistics."""
        stats = {
            'algorithm': 'genetic',
            'generation': len(self.best_fitness_history),
            'population_size': len(self.population),
            'elite_size': self.elite_size,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'selection_method': self.selection_method,
            'crossover_method': self.crossover_method,
            'generations_without_improvement': self.generations_without_improvement
        }
        
        if self.best_fitness_history:
            stats['best_fitness'] = max(self.best_fitness_history)
            stats['avg_fitness'] = self.avg_fitness_history[-1] if self.avg_fitness_history else 0
            stats['fitness_improvement'] = (
                self.best_fitness_history[-1] - self.best_fitness_history[0]
                if len(self.best_fitness_history) > 1 else 0
            )
        
        return stats

