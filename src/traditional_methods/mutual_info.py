"""
Mutual Information based feature selection
"""

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from typing import Optional


class MutualInfoSelector:
    """Feature selection based on mutual information."""
    
    def __init__(self, k: Optional[int] = None, threshold: float = 0.0):
        """
        Initialize mutual information selector.
        
        Args:
            k: Number of features to select (if None, use threshold)
            threshold: Minimum mutual information threshold
        """
        self.k = k
        self.threshold = threshold
        self.selected_features_ = None
        self.feature_scores_ = None
        self.is_classification = None
    
    def _detect_task_type(self, y: np.ndarray) -> bool:
        """Detect if classification or regression."""
        if self.is_classification is None:
            # Simple heuristic: if discrete with few unique values, treat as classification
            unique_values = len(np.unique(y))
            self.is_classification = unique_values < 20
        return self.is_classification
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the selector.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        is_classification = self._detect_task_type(y)
        
        # Calculate mutual information
        if is_classification:
            mi_scores = mutual_info_classif(X, y, random_state=42)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=42)
        
        self.feature_scores_ = np.array(mi_scores)
        
        # Select features
        if self.k is not None:
            # Select top k features
            top_k_indices = np.argsort(self.feature_scores_)[-self.k:][::-1]
            self.selected_features_ = np.sort(top_k_indices)
        else:
            # Select features above threshold
            self.selected_features_ = np.where(self.feature_scores_ >= self.threshold)[0]
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to selected features."""
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted first.")
        return X[:, self.selected_features_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)
    
    def get_selected_features(self) -> np.ndarray:
        """Get indices of selected features."""
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted first.")
        return self.selected_features_

