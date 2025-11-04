"""
Main Genetic Algorithm implementation for feature selection
"""

import numpy as np
from typing import List, Tuple, Callable, Dict, Any
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import time

from .chromosome import Chromosome
from .operators import Selection, Crossover, Mutation


class GeneticAlgorithm:
    """
    Genetic Algorithm for feature selection.
    """
    
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 population_size: int = 50,
                 n_generations: int = 50,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.01,
                 tournament_size: int = 3,
                 selection_method: str = 'tournament',
                 crossover_method: str = 'uniform',
                 elite_size: int = 2,
                 min_features: int = 1,
                 max_features: int = None,
                 fitness_alpha: float = 0.7,
                 fitness_beta: float = 0.3,
                 cv_folds: int = 5,
                 random_state: int = None,
                 progress_callback: Callable = None):
        """
        Initialize Genetic Algorithm for feature selection.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            population_size: Size of population
            n_generations: Number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation per gene
            tournament_size: Size of tournament for selection
            selection_method: 'tournament' or 'roulette'
            crossover_method: 'single_point', 'two_point', or 'uniform'
            elite_size: Number of best individuals to preserve
            min_features: Minimum number of features to select
            max_features: Maximum number of features to select (None = no limit)
            fitness_alpha: Weight for model performance in fitness
            fitness_beta: Weight for feature reduction in fitness
            cv_folds: Number of cross-validation folds
            random_state: Random seed
        """
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.elite_size = elite_size
        self.min_features = min_features
        self.max_features = max_features if max_features else self.n_features
        self.fitness_alpha = fitness_alpha
        self.fitness_beta = fitness_beta
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self.progress_callback = progress_callback
        
        # Initialize population
        self.population = self._initialize_population()
        
        # History tracking
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_n_features': [],
            'generation_time': []
        }
        
        # Best solution
        self.best_chromosome = None
        self.best_fitness = -np.inf
        self.best_model_score = -np.inf
    
    def _initialize_population(self) -> List[Chromosome]:
        """Initialize random population."""
        population = []
        for _ in range(self.population_size):
            # Random initialization with constraints
            while True:
                chromosome = Chromosome(n_features=self.n_features)
                n_selected = chromosome.count_selected()
                if self.min_features <= n_selected <= self.max_features:
                    break
            population.append(chromosome)
        return population
    
    def _evaluate_fitness(self, chromosome: Chromosome) -> Tuple[float, float]:
        """
        Evaluate fitness of a chromosome.
        
        Returns:
            Tuple of (fitness_score, model_score)
        """
        selected_features = chromosome.get_selected_features()
        n_selected = len(selected_features)
        
        # Ensure constraints
        if n_selected < self.min_features or n_selected > self.max_features:
            return -np.inf, 0.0
        
        if n_selected == 0:
            return -np.inf, 0.0
        
        # Select features
        X_selected = self.X[:, selected_features]
        
        # Train and evaluate model using cross-validation
        try:
            model = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            scorer = make_scorer(f1_score, average='weighted')
            cv_scores = cross_val_score(model, X_selected, self.y, 
                                       cv=self.cv_folds, scoring=scorer)
            model_score = cv_scores.mean()
        except Exception as e:
            # If model fails, return low fitness
            print(f"Model evaluation failed: {e}")
            return -np.inf, 0.0
        
        # Calculate fitness
        fitness = chromosome.fitness(model_score, n_selected,
                                   alpha=self.fitness_alpha,
                                   beta=self.fitness_beta)
        
        return fitness, model_score
    
    def _select_parents(self, fitness_scores: List[float]) -> Tuple[Chromosome, Chromosome]:
        """Select two parents using selection method."""
        if self.selection_method == 'tournament':
            parent1 = Selection.tournament_selection(
                self.population, fitness_scores, self.tournament_size)
            parent2 = Selection.tournament_selection(
                self.population, fitness_scores, self.tournament_size)
        else:  # roulette
            parent1 = Selection.roulette_wheel_selection(
                self.population, fitness_scores)
            parent2 = Selection.roulette_wheel_selection(
                self.population, fitness_scores)
        
        return parent1, parent2
    
    def _create_offspring(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Create offspring from two parents."""
        # Crossover
        if np.random.random() < self.crossover_rate:
            if self.crossover_method == 'single_point':
                child1, child2 = Crossover.single_point(parent1, parent2)
            elif self.crossover_method == 'two_point':
                child1, child2 = Crossover.two_point(parent1, parent2)
            else:  # uniform
                child1, child2 = Crossover.uniform(parent1, parent2)
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()
        
        # Mutation
        child1 = Mutation.adaptive_mutation(child1, self.mutation_rate)
        child2 = Mutation.adaptive_mutation(child2, self.mutation_rate)
        
        return child1, child2
    
    def _enforce_constraints(self, chromosome: Chromosome) -> Chromosome:
        """Enforce min/max feature constraints."""
        n_selected = chromosome.count_selected()
        
        if n_selected < self.min_features:
            # Add random features
            unselected = np.where(~chromosome.genes)[0]
            n_to_add = self.min_features - n_selected
            if len(unselected) > 0:
                features_to_add = np.random.choice(unselected, 
                                                  min(n_to_add, len(unselected)),
                                                  replace=False)
                chromosome.genes[features_to_add] = True
        elif n_selected > self.max_features:
            # Remove random features
            selected = np.where(chromosome.genes)[0]
            n_to_remove = n_selected - self.max_features
            features_to_remove = np.random.choice(selected, n_to_remove, replace=False)
            chromosome.genes[features_to_remove] = False
        
        return chromosome
    
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the genetic algorithm.
        
        Args:
            verbose: Print progress information
        
        Returns:
            Dictionary with results and history
        """
        if verbose:
            print(f"Starting Genetic Algorithm for Feature Selection")
            print(f"Population size: {self.population_size}")
            print(f"Generations: {self.n_generations}")
            print(f"Total features: {self.n_features}")
            print("-" * 60)
        
        # Evaluate initial population
        fitness_scores = []
        model_scores = []
        
        for chromosome in self.population:
            fitness, model_score = self._evaluate_fitness(chromosome)
            fitness_scores.append(fitness)
            model_scores.append(model_score)
        
        # Track best
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = fitness_scores[best_idx]
            self.best_chromosome = self.population[best_idx].copy()
            self.best_model_score = model_scores[best_idx]
        
        # Main evolution loop
        for generation in range(self.n_generations):
            start_time = time.time()
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:][::-1]
            for idx in elite_indices:
                new_population.append(self.population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1, parent2 = self._select_parents(fitness_scores)
                child1, child2 = self._create_offspring(parent1, parent2)
                
                # Enforce constraints
                child1 = self._enforce_constraints(child1)
                child2 = self._enforce_constraints(child2)
                
                new_population.extend([child1, child2])
            
            # Trim to population size
            new_population = new_population[:self.population_size]
            
            # Evaluate new population
            fitness_scores = []
            model_scores = []
            
            for chromosome in new_population:
                fitness, model_score = self._evaluate_fitness(chromosome)
                fitness_scores.append(fitness)
                model_scores.append(model_score)
            
            # Update population
            self.population = new_population
            
            # Track best
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_chromosome = self.population[best_idx].copy()
                self.best_model_score = model_scores[best_idx]
            
            # Record history
            self.history['best_fitness'].append(self.best_fitness)
            self.history['avg_fitness'].append(np.mean(fitness_scores))
            self.history['best_n_features'].append(self.best_chromosome.count_selected())
            self.history['generation_time'].append(time.time() - start_time)
            
            # Call progress callback if provided
            if self.progress_callback:
                progress_data = {
                    'generation': generation + 1,
                    'total_generations': self.n_generations,
                    'best_fitness': self.best_fitness,
                    'avg_fitness': np.mean(fitness_scores),
                    'best_n_features': self.best_chromosome.count_selected(),
                    'model_score': self.best_model_score,
                    'progress': (generation + 1) / self.n_generations * 100
                }
                self.progress_callback(progress_data)
            
            if verbose and (generation + 1) % 10 == 0:
                print(f"Generation {generation + 1}/{self.n_generations}: "
                      f"Best Fitness = {self.best_fitness:.4f}, "
                      f"Best Features = {self.best_chromosome.count_selected()}, "
                      f"Model Score = {self.best_model_score:.4f}")
        
        if verbose:
            print("-" * 60)
            print(f"Evolution complete!")
            print(f"Best fitness: {self.best_fitness:.4f}")
            print(f"Best model score: {self.best_model_score:.4f}")
            print(f"Selected features: {self.best_chromosome.count_selected()}/{self.n_features}")
            print(f"Feature indices: {self.best_chromosome.get_selected_features()[:10]}..." 
                  if self.best_chromosome.count_selected() > 10 
                  else f"Feature indices: {self.best_chromosome.get_selected_features()}")
        
        return {
            'best_chromosome': self.best_chromosome,
            'best_fitness': self.best_fitness,
            'best_model_score': self.best_model_score,
            'history': self.history,
            'selected_features': self.best_chromosome.get_selected_features()
        }
    
    def get_selected_features(self) -> np.ndarray:
        """Get indices of selected features from best solution."""
        if self.best_chromosome is None:
            raise ValueError("Algorithm has not been run yet. Call run() first.")
        return self.best_chromosome.get_selected_features()

