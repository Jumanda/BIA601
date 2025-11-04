"""
Genetic operators: Selection, Crossover, and Mutation
"""

import numpy as np
from typing import List
from .chromosome import Chromosome


class Selection:
    """Selection operators for genetic algorithm."""
    
    @staticmethod
    def tournament_selection(population: List[Chromosome], 
                           fitness_scores: List[float],
                           tournament_size: int = 3) -> Chromosome:
        """
        Tournament selection.
        
        Args:
            population: List of chromosomes
            fitness_scores: List of fitness scores for each chromosome
            tournament_size: Number of individuals in tournament
        
        Returns:
            Selected chromosome
        """
        indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in indices]
        winner_idx = indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    @staticmethod
    def roulette_wheel_selection(population: List[Chromosome],
                                fitness_scores: List[float]) -> Chromosome:
        """
        Roulette wheel (fitness proportionate) selection.
        
        Args:
            population: List of chromosomes
            fitness_scores: List of fitness scores for each chromosome
        
        Returns:
            Selected chromosome
        """
        # Convert to probabilities (handle negative fitness)
        min_fitness = min(fitness_scores)
        adjusted_fitness = [f - min_fitness + 1e-10 for f in fitness_scores]
        total_fitness = sum(adjusted_fitness)
        probabilities = [f / total_fitness for f in adjusted_fitness]
        
        selected_idx = np.random.choice(len(population), p=probabilities)
        return population[selected_idx].copy()


class Crossover:
    """Crossover operators for genetic algorithm."""
    
    @staticmethod
    def single_point(parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Single-point crossover.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
        
        Returns:
            Two offspring chromosomes
        """
        if parent1.n_features != parent2.n_features:
            raise ValueError("Parents must have the same number of features")
        
        point = np.random.randint(1, parent1.n_features)
        
        child1_genes = np.concatenate([parent1.genes[:point], parent2.genes[point:]])
        child2_genes = np.concatenate([parent2.genes[:point], parent1.genes[point:]])
        
        child1 = Chromosome(genes=child1_genes)
        child2 = Chromosome(genes=child2_genes)
        
        return child1, child2
    
    @staticmethod
    def uniform(parent1: Chromosome, parent2: Chromosome, 
               crossover_prob: float = 0.5) -> Tuple[Chromosome, Chromosome]:
        """
        Uniform crossover.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            crossover_prob: Probability of taking gene from parent1 (default: 0.5)
        
        Returns:
            Two offspring chromosomes
        """
        if parent1.n_features != parent2.n_features:
            raise ValueError("Parents must have the same number of features")
        
        mask = np.random.random(parent1.n_features) < crossover_prob
        
        child1_genes = np.where(mask, parent1.genes, parent2.genes)
        child2_genes = np.where(mask, parent2.genes, parent1.genes)
        
        child1 = Chromosome(genes=child1_genes)
        child2 = Chromosome(genes=child2_genes)
        
        return child1, child2
    
    @staticmethod
    def two_point(parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Two-point crossover.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
        
        Returns:
            Two offspring chromosomes
        """
        if parent1.n_features != parent2.n_features:
            raise ValueError("Parents must have the same number of features")
        
        points = sorted(np.random.choice(parent1.n_features, 2, replace=False))
        
        child1_genes = np.concatenate([
            parent1.genes[:points[0]],
            parent2.genes[points[0]:points[1]],
            parent1.genes[points[1]:]
        ])
        child2_genes = np.concatenate([
            parent2.genes[:points[0]],
            parent1.genes[points[0]:points[1]],
            parent2.genes[points[1]:]
        ])
        
        child1 = Chromosome(genes=child1_genes)
        child2 = Chromosome(genes=child2_genes)
        
        return child1, child2


class Mutation:
    """Mutation operators for genetic algorithm."""
    
    @staticmethod
    def bit_flip(chromosome: Chromosome, mutation_rate: float = 0.01) -> Chromosome:
        """
        Bit-flip mutation.
        
        Args:
            chromosome: Chromosome to mutate
            mutation_rate: Probability of flipping each bit
        
        Returns:
            Mutated chromosome
        """
        mutated = chromosome.copy()
        mutation_mask = np.random.random(chromosome.n_features) < mutation_rate
        mutated.genes[mutation_mask] = ~mutated.genes[mutation_mask]
        return mutated
    
    @staticmethod
    def adaptive_mutation(chromosome: Chromosome, 
                         base_rate: float = 0.01,
                         n_selected: int = None) -> Chromosome:
        """
        Adaptive mutation: higher rate if too few/many features selected.
        
        Args:
            chromosome: Chromosome to mutate
            base_rate: Base mutation rate
            n_selected: Number of currently selected features
        
        Returns:
            Mutated chromosome
        """
        if n_selected is None:
            n_selected = chromosome.count_selected()
        
        # Adjust mutation rate based on feature count
        # If too few features (< 10%), increase mutation
        # If too many features (> 80%), increase mutation
        feature_ratio = n_selected / chromosome.n_features
        
        if feature_ratio < 0.1 or feature_ratio > 0.8:
            mutation_rate = base_rate * 2.0
        else:
            mutation_rate = base_rate
        
        return Mutation.bit_flip(chromosome, mutation_rate)

