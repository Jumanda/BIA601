"""
Correlation-based feature selection
"""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from typing import Optional


class CorrelationSelector:
    """Feature selection based on correlation with target."""
    
    def __init__(self, k: Optional[int] = None, threshold: float = 0.1):
        """
        Initialize correlation-based selector.
        
        Args:
            k: Number of features to select (if None, use threshold)
            threshold: Minimum correlation threshold
        """
        self.k = k
        self.threshold = threshold
        self.selected_features_ = None
        self.feature_scores_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the selector.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        # Calculate correlation between each feature and target
        correlations = []
        for i in range(X.shape[1]):
            if np.std(X[:, i]) > 1e-10:  # Avoid division by zero
                corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
                correlations.append(corr if not np.isnan(corr) else 0.0)
            else:
                correlations.append(0.0)
        
        self.feature_scores_ = np.array(correlations)
        
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
    
    def get_support(self, n_features: int = None) -> np.ndarray:
        """Get boolean mask of selected features."""
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted first.")
        if n_features is None:
            n_features = len(self.feature_scores_) if self.feature_scores_ is not None else max(self.selected_features_) + 1
        mask = np.zeros(n_features, dtype=bool)
        mask[self.selected_features_] = True
        return mask
    
    def get_selected_features(self) -> np.ndarray:
        """Get indices of selected features."""
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted first.")
        return self.selected_features_

