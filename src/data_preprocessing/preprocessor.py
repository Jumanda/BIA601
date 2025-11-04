"""
Data preprocessing for feature selection
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Optional, Tuple


class DataPreprocessor:
    """Data preprocessing utilities."""
    
    def __init__(self,
                 handle_missing: str = 'mean',
                 scale_features: bool = True,
                 encode_categorical: bool = True):
        """
        Initialize preprocessor.
        
        Args:
            handle_missing: Strategy for handling missing values ('mean', 'median', 'most_frequent', 'drop')
            scale_features: Whether to standardize features
            encode_categorical: Whether to encode categorical variables
        """
        self.handle_missing = handle_missing
        self.scale_features = scale_features
        self.encode_categorical = encode_categorical
        
        self.imputer_ = None
        self.scaler_ = None
        self.label_encoders_ = {}
        self.feature_names_ = None
        self.target_name_ = None
    
    def fit(self, X, y=None):
        """
        Fit preprocessor.
        
        Args:
            X: Feature matrix or DataFrame
            y: Target vector (optional)
        """
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        self.feature_names_ = X.columns.tolist()
        
        # Handle missing values
        if self.handle_missing != 'drop':
            self.imputer_ = SimpleImputer(strategy=self.handle_missing)
            X_imputed = pd.DataFrame(
                self.imputer_.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_imputed = X.dropna()
        
        # Encode categorical variables
        if self.encode_categorical:
            for col in X_imputed.columns:
                if X_imputed[col].dtype == 'object' or X_imputed[col].dtype.name == 'category':
                    le = LabelEncoder()
                    X_imputed[col] = le.fit_transform(X_imputed[col].astype(str))
                    self.label_encoders_[col] = le
        
        # Scale features
        if self.scale_features:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X_imputed)
        
        # Handle target if provided
        if y is not None:
            if isinstance(y, pd.Series):
                self.target_name_ = y.name
            elif hasattr(y, 'name'):
                self.target_name_ = y.name
            else:
                self.target_name_ = 'target'
    
    def transform(self, X, y=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform data.
        
        Args:
            X: Feature matrix or DataFrame
            y: Target vector (optional)
        
        Returns:
            Tuple of (X_transformed, y_transformed)
        """
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            if self.feature_names_:
                X = pd.DataFrame(X, columns=self.feature_names_)
            else:
                X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # Handle missing values
        if self.imputer_ is not None:
            X_transformed = pd.DataFrame(
                self.imputer_.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_transformed = X.dropna()
        
        # Encode categorical variables
        if self.encode_categorical:
            for col in X_transformed.columns:
                if col in self.label_encoders_:
                    le = self.label_encoders_[col]
                    # Handle unseen categories
                    mask = X_transformed[col].isin(le.classes_)
                    X_transformed.loc[~mask, col] = le.classes_[0]  # Use most common
                    X_transformed[col] = le.transform(X_transformed[col].astype(str))
        
        # Scale features
        if self.scaler_ is not None:
            X_transformed = self.scaler_.transform(X_transformed)
            X_transformed = np.array(X_transformed)
        else:
            X_transformed = np.array(X_transformed)
        
        # Transform target if provided
        y_transformed = None
        if y is not None:
            if isinstance(y, pd.Series):
                y_transformed = y.values
            else:
                y_transformed = np.array(y)
            
            # Encode target if categorical
            if self.encode_categorical and 'target' in self.label_encoders_:
                le = self.label_encoders_['target']
                y_transformed = le.transform(y_transformed.astype(str))
        
        return X_transformed, y_transformed
    
    def fit_transform(self, X, y=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X, y)
    
    def get_feature_names(self):
        """Get feature names."""
        return self.feature_names_

