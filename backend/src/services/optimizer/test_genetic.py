"""
Tests for Genetic Algorithm optimizer.

Run with:
    pytest test_genetic.py -v
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from algorithms.genetic import GeneticAlgorithmOptimizer
from algorithms.base import ParameterSpace, OptimizationResult


class TestGeneticAlgorithm:
    """Tests for Genetic Algorithm optimizer."""
    
    def test_initialization(self):
        """Test GA optimizer can be initialized."""
        param_space = ParameterSpace(ranges={
            'short_period': [5, 10, 15, 20],
            'long_period': [30, 40, 50, 60]
        })
        
        ga = GeneticAlgorithmOptimizer(
            param_space=param_space,
            max_iterations=100,
            population_size=20,
            random_seed=42
        )
        
        assert ga.population_size == 20
        assert ga.mutation_rate == 0.1
        assert ga.crossover_rate == 0.8
        assert ga.elite_size == 5
        assert len(ga.population) == 0  # Not initialized yet
    
    def test_population_initialization(self):
        """Test random population initialization."""
        param_space = ParameterSpace(ranges={
            'short_period': [5, 10, 15, 20],
            'long_period': [30, 40, 50, 60]
        })
        
        ga = GeneticAlgorithmOptimizer(
            param_space=param_space,
            population_size=10,
            random_seed=42
        )
        
        population = ga._initialize_population()
        
        assert len(population) == 10
        # Check all individuals are dicts with correct keys
        for individual in population:
            assert isinstance(individual, dict)
            assert 'short_period' in individual
            assert 'long_period' in individual
            # Check values are from allowed ranges
            assert individual['short_period'] in [5, 10, 15, 20]
            assert individual['long_period'] in [30, 40, 50, 60]
    
    def test_constraint_handling(self):
        """Test population initialization respects constraints."""
        param_space = ParameterSpace(ranges={
            'short_period': [5, 10, 15, 20],
            'long_period': [30, 40, 50, 60]
        })
        
        ga = GeneticAlgorithmOptimizer(
            param_space=param_space,
            constraints=['short_period < long_period'],
            population_size=10,
            random_seed=42
        )
        
        population = ga._initialize_population()
        
        # All individuals should satisfy constraint
        for individual in population:
            assert individual['short_period'] < individual['long_period']
    
    def test_encode_decode(self):
        """Test individual encoding and decoding."""
        param_space = ParameterSpace(ranges={
            'short_period': [5, 10, 15],
            'long_period': [30, 40, 50]
        })
        
        ga = GeneticAlgorithmOptimizer(param_space=param_space)
        
        individual = {'short_period': 10, 'long_period': 40}
        
        # Encode to gene sequence
        genes = ga._encode_individual(individual)
        assert isinstance(genes, list)
        assert len(genes) == 2
        
        # Decode back to individual
        decoded = ga._decode_individual(genes)
        assert decoded == individual
    
    def test_single_point_crossover(self):
        """Test single-point crossover."""
        param_space = ParameterSpace(ranges={
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        
        ga = GeneticAlgorithmOptimizer(
            param_space=param_space,
            crossover_method='single_point',
            crossover_rate=1.0,  # Always crossover for testing
            random_seed=42
        )
        
        parent1 = {'a': 1, 'b': 4, 'c': 7}
        parent2 = {'a': 3, 'b': 6, 'c': 9}
        
        child1, child2 = ga._crossover(parent1, parent2)
        
        # Children should be different from parents (with high probability)
        # and contain values from both parents
        assert isinstance(child1, dict)
        assert isinstance(child2, dict)
        assert set(child1.keys()) == {'a', 'b', 'c'}
        assert set(child2.keys()) == {'a', 'b', 'c'}
    
    def test_two_point_crossover(self):
        """Test two-point crossover."""
        param_space = ParameterSpace(ranges={
            'a': [1, 2],
            'b': [3, 4],
            'c': [5, 6],
            'd': [7, 8]
        })
        
        ga = GeneticAlgorithmOptimizer(
            param_space=param_space,
            crossover_method='two_point',
            crossover_rate=1.0,
            random_seed=42
        )
        
        parent1 = {'a': 1, 'b': 3, 'c': 5, 'd': 7}
        parent2 = {'a': 2, 'b': 4, 'c': 6, 'd': 8}
        
        child1, child2 = ga._crossover(parent1, parent2)
        
        assert isinstance(child1, dict)
        assert isinstance(child2, dict)
    
    def test_uniform_crossover(self):
        """Test uniform crossover."""
        param_space = ParameterSpace(ranges={
            'a': [1, 2],
            'b': [3, 4],
            'c': [5, 6]
        })
        
        ga = GeneticAlgorithmOptimizer(
            param_space=param_space,
            crossover_method='uniform',
            crossover_rate=1.0,
            random_seed=42
        )
        
        parent1 = {'a': 1, 'b': 3, 'c': 5}
        parent2 = {'a': 2, 'b': 4, 'c': 6}
        
        child1, child2 = ga._crossover(parent1, parent2)
        
        assert isinstance(child1, dict)
        assert isinstance(child2, dict)
        # Each gene should come from one of the parents
        assert child1['a'] in [1, 2]
        assert child1['b'] in [3, 4]
        assert child1['c'] in [5, 6]
    
    def test_mutation(self):
        """Test mutation operator."""
        param_space = ParameterSpace(ranges={
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        
        ga = GeneticAlgorithmOptimizer(
            param_space=param_space,
            mutation_rate=1.0,  # Always mutate for testing
            random_seed=42
        )
        
        original = {'a': 1, 'b': 10}
        mutated = ga._mutate(original)
        
        assert isinstance(mutated, dict)
        # At least one parameter should potentially be different
        # (though with small parameter spaces it might be same by chance)
        assert mutated['a'] in [1, 2, 3, 4, 5]
        assert mutated['b'] in [10, 20, 30, 40, 50]
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        param_space = ParameterSpace(ranges={
            'x': [1, 2, 3]
        })
        
        ga = GeneticAlgorithmOptimizer(
            param_space=param_space,
            tournament_size=2,
            random_seed=42
        )
        
        population = [
            {'x': 1},
            {'x': 2},
            {'x': 3}
        ]
        fitness_scores = [0.5, 0.8, 0.3]
        
        selected = ga._tournament_selection(population, fitness_scores)
        
        assert isinstance(selected, dict)
        assert 'x' in selected
        # Should tend to select higher fitness individuals
    
    def test_roulette_selection(self):
        """Test roulette wheel selection."""
        param_space = ParameterSpace(ranges={
            'x': [1, 2, 3]
        })
        
        ga = GeneticAlgorithmOptimizer(
            param_space=param_space,
            selection_method='roulette',
            random_seed=42
        )
        
        population = [
            {'x': 1},
            {'x': 2},
            {'x': 3}
        ]
        fitness_scores = [0.1, 0.9, 0.3]
        
        selected = ga._roulette_selection(population, fitness_scores)
        
        assert isinstance(selected, dict)
        assert 'x' in selected
    
    def test_elite_preservation(self):
        """Test elite individuals are preserved."""
        param_space = ParameterSpace(ranges={
            'x': [1, 2, 3, 4, 5]
        })
        
        ga = GeneticAlgorithmOptimizer(
            param_space=param_space,
            elite_size=2
        )
        
        population = [
            {'x': 1},
            {'x': 2},
            {'x': 3},
            {'x': 4},
            {'x': 5}
        ]
        fitness_scores = [0.1, 0.9, 0.3, 0.7, 0.2]
        
        elite = ga._get_elite(population, fitness_scores)
        
        assert len(elite) == 2
        # Should get the two best individuals (fitness 0.9 and 0.7)
        assert elite[0] in population
        assert elite[1] in population
    
    def test_generation_loop(self):
        """Test generation loop generates candidates."""
        param_space = ParameterSpace(ranges={
            'short_period': [5, 10, 15],
            'long_period': [30, 40, 50]
        })
        
        ga = GeneticAlgorithmOptimizer(
            param_space=param_space,
            population_size=5,
            max_iterations=20,  # 5 initial + 15 more = 20 total
            random_seed=42
        )
        
        candidates = []
        for candidate in ga.generate_candidates():
            candidates.append(candidate)
            
            # Simulate evaluation by updating with a result
            result = OptimizationResult(
                params=candidate,
                score=0.5 + (len(candidates) * 0.01),  # Slightly increasing score
                metrics={'sharpe_ratio': 0.5}
            )
            ga.update(result)
            
            if len(candidates) >= 20:
                break
        
        assert len(candidates) >= 5  # At least the initial population
        # All candidates should be valid
        for candidate in candidates:
            assert 'short_period' in candidate
            assert 'long_period' in candidate
    
    def test_convergence_detection(self):
        """Test convergence detection."""
        param_space = ParameterSpace(ranges={'x': [1, 2, 3]})
        
        ga = GeneticAlgorithmOptimizer(
            param_space=param_space,
            convergence_generations=3,
            convergence_threshold=0.001
        )
        
        # Simulate no improvement
        ga.best_fitness_history = [1.0, 1.0001, 1.0002, 1.0001, 1.0]
        
        assert ga._check_convergence() == True
        
        # Simulate improvement
        ga.best_fitness_history = [1.0, 1.1, 1.2, 1.3, 1.4]
        
        assert ga._check_convergence() == False
    
    def test_stats(self):
        """Test statistics reporting."""
        param_space = ParameterSpace(ranges={'x': [1, 2, 3]})
        
        ga = GeneticAlgorithmOptimizer(
            param_space=param_space,
            population_size=10,
            elite_size=2,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
        ga.best_fitness_history = [0.5, 0.6, 0.7]
        ga.avg_fitness_history = [0.4, 0.5, 0.6]
        ga.population = [{'x': 1}] * 10
        
        stats = ga.get_stats()
        
        assert stats['algorithm'] == 'genetic'
        assert stats['generation'] == 3
        assert stats['population_size'] == 10
        assert stats['elite_size'] == 2
        assert stats['mutation_rate'] == 0.1
        assert stats['best_fitness'] == 0.7
        assert stats['avg_fitness'] == 0.6


if __name__ == '__main__':
    print("Running Genetic Algorithm tests...")
    print("=" * 80)
    print("Run with: pytest test_genetic.py -v")
    print("=" * 80)

