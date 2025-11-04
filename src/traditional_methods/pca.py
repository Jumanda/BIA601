"""
Principal Component Analysis (PCA) for dimensionality reduction and feature selection
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, Union


class PCASelector:
    """
    Feature selection/dimensionality reduction using Principal Component Analysis.
    
    PCA transforms features into principal components that capture maximum variance.
    This is a dimensionality reduction technique rather than traditional feature selection,
    but we adapt it to return which original features contribute most to the selected components.
    """
    
    def __init__(self, n_components: Optional[Union[int, float]] = None, variance_threshold: float = 0.95):
        """
        Initialize PCA selector.
        
        Args:
            n_components: Number of components to keep
                - If int: exact number of components
                - If float (0 < n_components < 1): keep components that explain this variance ratio
                - If None: keep all components
            variance_threshold: Minimum variance to retain (used if n_components is float)
        """
        self.n_components = n_components if n_components is not None else variance_threshold
        self.variance_threshold = variance_threshold
        self.pca_ = None
        self.selected_features_ = None
        self.feature_scores_ = None  # Explained variance ratio
        self.components_ = None
        self.explained_variance_ratio_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fit PCA to the data.
        
        Args:
            X: Feature matrix (shape: n_samples, n_features)
            y: Target vector (optional, not used in PCA but kept for API consistency)
        """
        # Determine number of components
        n_components = self.n_components
        if isinstance(n_components, float) and 0 < n_components < 1:
            # Use variance threshold
            n_components = n_components
        elif n_components is None:
            # Use variance threshold
            n_components = self.variance_threshold
        
        # Create and fit PCA
        self.pca_ = PCA(n_components=n_components, random_state=42)
        self.pca_.fit(X)
        
        # Store results
        self.components_ = self.pca_.components_
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_
        
        # Calculate feature importance based on absolute loading values
        # Sum absolute loadings across all components, weighted by explained variance
        if self.components_.shape[0] > 0:
            # Weighted feature importance: sum of absolute loadings weighted by variance explained
            feature_importance = np.zeros(X.shape[1])
            for i, component in enumerate(self.components_):
                # Weight by explained variance ratio
                weight = self.explained_variance_ratio_[i]
                feature_importance += np.abs(component) * weight
            
            self.feature_scores_ = feature_importance
        else:
            self.feature_scores_ = np.ones(X.shape[1])
        
        # Select top features based on importance scores
        # Select features that contribute most to the principal components
        # We'll select top k features where k is based on cumulative explained variance
        if isinstance(self.n_components, int):
            # Select top n_components features
            n_select = min(self.n_components, X.shape[1])
        else:
            # Select features until we have enough variance explanation
            # Use a heuristic: select features until cumulative importance reaches threshold
            n_select = min(int(X.shape[1] * 0.5), 50)  # Select up to 50% of features or 50 max
        
        top_indices = np.argsort(self.feature_scores_)[-n_select:][::-1]
        self.selected_features_ = np.sort(top_indices)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using PCA (returns principal components).
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed data (principal components)
        """
        if self.pca_ is None:
            raise ValueError("PCA must be fitted first.")
        return self.pca_.transform(X)
    
    def transform_selected_features(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using only selected original features (not PCA components).
        
        Args:
            X: Feature matrix
            
        Returns:
            Data with only selected features
        """
        if self.selected_features_ is None:
            raise ValueError("PCA must be fitted first.")
        return X[:, self.selected_features_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Fit PCA and transform data to principal components.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            
        Returns:
            Principal components
        """
        self.fit(X, y)
        return self.transform(X)
    
    def get_selected_features(self) -> np.ndarray:
        """
        Get indices of selected original features.
        
        Returns:
            Array of feature indices
        """
        if self.selected_features_ is None:
            raise ValueError("PCA must be fitted first.")
        return self.selected_features_
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Get explained variance ratio for each principal component.
        
        Returns:
            Array of explained variance ratios
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("PCA must be fitted first.")
        return self.explained_variance_ratio_
    
    def get_total_explained_variance(self) -> float:
        """
        Get total explained variance ratio.
        
        Returns:
            Total explained variance (0-1)
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("PCA must be fitted first.")
        return float(np.sum(self.explained_variance_ratio_))



