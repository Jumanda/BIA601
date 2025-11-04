"""
Recursive Feature Elimination (RFE)
"""

import numpy as np
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Optional


class RecursiveEliminationSelector:
    """Recursive Feature Elimination for feature selection."""
    
    def __init__(self, n_features_to_select: Optional[int] = None,
                 use_cv: bool = False, cv: int = 5):
        """
        Initialize RFE selector.
        
        Args:
            n_features_to_select: Number of features to select
            use_cv: Use cross-validation (RFECV) if True
            cv: Number of CV folds
        """
        self.n_features_to_select = n_features_to_select
        self.use_cv = use_cv
        self.cv = cv
        self.selector_ = None
        self.selected_features_ = None
        self.feature_scores_ = None
        self.is_classification = None
    
    def _detect_task_type(self, y: np.ndarray) -> bool:
        """Detect if classification or regression."""
        if self.is_classification is None:
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
        
        # Create base estimator
        if is_classification:
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # Create selector
        if self.use_cv:
            self.selector_ = RFECV(estimator, cv=self.cv, scoring='f1_weighted' if is_classification else 'r2')
        else:
            n_features = self.n_features_to_select if self.n_features_to_select else min(X.shape[1] // 2, 10)
            self.selector_ = RFE(estimator, n_features_to_select=n_features)
        
        self.selector_.fit(X, y)
        
        self.selected_features_ = self.selector_.get_support(indices=True)
        
        # Get feature rankings/scores
        if hasattr(self.selector_, 'grid_scores_'):
            self.feature_scores_ = self.selector_.grid_scores_
        elif hasattr(self.selector_, 'ranking_'):
            # Convert ranking to scores (lower rank = better)
            self.feature_scores_ = 1.0 / (self.selector_.ranking_ + 1e-10)
        else:
            self.feature_scores_ = np.ones(X.shape[1])
        
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

