"""
Chromosome representation for feature selection
"""

import numpy as np
from typing import List, Tuple


class Chromosome:
    """
    Represents a chromosome in the genetic algorithm.
    Each chromosome is a binary vector where 1 indicates a feature is selected.
    """
    
    def __init__(self, genes: np.ndarray = None, n_features: int = None):
        """
        Initialize a chromosome.
        
        Args:
            genes: Binary array representing selected features (1 = selected, 0 = not selected)
            n_features: Number of features (used if genes is None)
        """
        if genes is not None:
            self.genes = genes.astype(bool)
            self.n_features = len(genes)
        elif n_features is not None:
            # Initialize with random features
            self.n_features = n_features
            # Start with ~50% features selected
            self.genes = np.random.random(n_features) > 0.5
        else:
            raise ValueError("Either genes or n_features must be provided")
    
    def __len__(self):
        """Return the number of features."""
        return self.n_features
    
    def __str__(self):
        """String representation."""
        selected = np.sum(self.genes)
        return f"Chromosome(selected={selected}/{self.n_features})"
    
    def __repr__(self):
        """Representation."""
        return self.__str__()
    
    def copy(self):
        """Create a deep copy of the chromosome."""
        new_chromosome = Chromosome(genes=self.genes.copy())
        return new_chromosome
    
    def get_selected_features(self) -> np.ndarray:
        """
        Get indices of selected features.
        
        Returns:
            Array of feature indices that are selected
        """
        return np.where(self.genes)[0]
    
    def count_selected(self) -> int:
        """Return the number of selected features."""
        return np.sum(self.genes)
    
    def to_array(self) -> np.ndarray:
        """Convert to binary array."""
        return self.genes.astype(int)
    
    def fitness(self, model_score: float, n_selected: int, 
                alpha: float = 0.7, beta: float = 0.3) -> float:
        """
        Calculate fitness score.
        Balance between model performance and number of selected features.
        
        Args:
            model_score: Model performance score (e.g., accuracy, F1-score)
            n_selected: Number of selected features
            alpha: Weight for model performance (default: 0.7)
            beta: Weight for feature reduction (default: 0.3)
        
        Returns:
            Fitness score (higher is better)
        """
        # Normalize feature reduction: fewer features = better
        # Assume max_features as total features
        max_features = self.n_features
        feature_reduction_score = 1.0 - (n_selected / max_features) if max_features > 0 else 0
        
        # Combined fitness: balance performance and feature reduction
        fitness = alpha * model_score + beta * feature_reduction_score
        
        return fitness

