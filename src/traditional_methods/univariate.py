"""
Univariate statistical tests for feature selection
"""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, chi2
from typing import Optional


class UnivariateSelector:
    """Feature selection using univariate statistical tests."""
    
    def __init__(self, k: Optional[int] = None, score_func: str = 'f_classif'):
        """
        Initialize univariate selector.
        
        Args:
            k: Number of features to select
            score_func: 'f_classif', 'f_regression', or 'chi2'
        """
        self.k = k
        self.score_func = score_func
        self.selected_features_ = None
        self.feature_scores_ = None
        self.selector_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the selector.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        # Choose score function
        if self.score_func == 'f_classif':
            score_func = f_classif
        elif self.score_func == 'f_regression':
            score_func = f_regression
        elif self.score_func == 'chi2':
            score_func = chi2
        else:
            raise ValueError(f"Unknown score function: {self.score_func}")
        
        # Determine k
        k = self.k if self.k is not None else min(X.shape[1], 10)
        
        # Create selector
        self.selector_ = SelectKBest(score_func=score_func, k=k)
        self.selector_.fit(X, y)
        
        self.selected_features_ = self.selector_.get_support(indices=True)
        self.feature_scores_ = self.selector_.scores_
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to selected features."""
        if self.selector_ is None:
            raise ValueError("Selector must be fitted first.")
        return self.selector_.transform(X)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)
    
    def get_selected_features(self) -> np.ndarray:
        """Get indices of selected features."""
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted first.")
        return self.selected_features_

