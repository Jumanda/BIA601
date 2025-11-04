"""
Tests for Genetic Algorithm implementation
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.genetic_algorithm import GeneticAlgorithm, Chromosome
from src.genetic_algorithm.operators import Selection, Crossover, Mutation


@pytest.fixture
def sample_data():
    """Generate sample dataset for testing."""
    X, y = make_classification(n_samples=100, n_features=20, n_informative=10,
                               n_redundant=5, n_classes=2, random_state=42)
    return X, y


@pytest.fixture
def small_ga(sample_data):
    """Create a small GA instance for testing."""
    X, y = sample_data
    return GeneticAlgorithm(X, y, population_size=10, n_generations=5, random_state=42)


def test_chromosome_initialization():
    """Test chromosome initialization."""
    chrom = Chromosome(n_features=10)
    assert len(chrom) == 10
    assert chrom.count_selected() >= 0
    assert chrom.count_selected() <= 10


def test_chromosome_copy():
    """Test chromosome copying."""
    chrom1 = Chromosome(n_features=10)
    chrom2 = chrom1.copy()
    assert np.array_equal(chrom1.genes, chrom2.genes)
    chrom1.genes[0] = not chrom1.genes[0]
    assert not np.array_equal(chrom1.genes, chrom2.genes)


def test_chromosome_fitness():
    """Test fitness calculation."""
    chrom = Chromosome(n_features=10)
    fitness = chrom.fitness(0.9, 5, alpha=0.7, beta=0.3)
    assert fitness > 0
    assert fitness <= 1.0


def test_selection_tournament(sample_data):
    """Test tournament selection."""
    X, y = sample_data
    ga = GeneticAlgorithm(X, y, population_size=10, n_generations=1, random_state=42)
    fitness_scores = [0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.3, 0.2, 0.1, 0.0]
    selected = Selection.tournament_selection(ga.population, fitness_scores, tournament_size=3)
    assert selected is not None
    assert isinstance(selected, Chromosome)


def test_crossover_single_point():
    """Test single-point crossover."""
    parent1 = Chromosome(genes=np.array([True, True, True, False, False]))
    parent2 = Chromosome(genes=np.array([False, False, False, True, True]))
    child1, child2 = Crossover.single_point(parent1, parent2)
    assert len(child1) == 5
    assert len(child2) == 5
    assert not np.array_equal(child1.genes, parent1.genes) or not np.array_equal(child1.genes, parent2.genes)


def test_crossover_uniform():
    """Test uniform crossover."""
    parent1 = Chromosome(genes=np.array([True, True, True, False, False]))
    parent2 = Chromosome(genes=np.array([False, False, False, True, True]))
    child1, child2 = Crossover.uniform(parent1, parent2)
    assert len(child1) == 5
    assert len(child2) == 5


def test_mutation_bit_flip():
    """Test bit-flip mutation."""
    chrom = Chromosome(genes=np.array([True, True, True, False, False]))
    original = chrom.genes.copy()
    mutated = Mutation.bit_flip(chrom, mutation_rate=1.0)  # 100% mutation
    # With 100% mutation, all bits should flip
    assert np.array_equal(mutated.genes, ~original)


def test_ga_initialization(sample_data):
    """Test GA initialization."""
    X, y = sample_data
    ga = GeneticAlgorithm(X, y, population_size=20, n_generations=10, random_state=42)
    assert len(ga.population) == 20
    assert ga.n_features == X.shape[1]
    assert ga.n_samples == X.shape[0]


def test_ga_run_small(small_ga):
    """Test running GA with small parameters."""
    result = small_ga.run(verbose=False)
    assert 'best_chromosome' in result
    assert 'best_fitness' in result
    assert 'selected_features' in result
    assert len(result['selected_features']) > 0
    assert result['best_fitness'] > -np.inf


def test_ga_get_selected_features(small_ga):
    """Test getting selected features."""
    small_ga.run(verbose=False)
    features = small_ga.get_selected_features()
    assert len(features) > 0
    assert all(isinstance(f, (int, np.integer)) for f in features)

