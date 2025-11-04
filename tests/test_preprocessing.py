"""
Tests for data preprocessing
"""

import pytest
import numpy as np
import pandas as pd
from src.data_preprocessing import DataPreprocessor


def test_preprocessor_fit_transform():
    """Test preprocessing fit and transform."""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    preprocessor = DataPreprocessor()
    X_transformed, y_transformed = preprocessor.fit_transform(X, y)
    
    assert X_transformed.shape[0] == X.shape[0]
    assert y_transformed is not None


def test_preprocessor_with_dataframe():
    """Test preprocessing with DataFrame."""
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    })
    
    X = df[['feature1', 'feature2']]
    y = df['target']
    
    preprocessor = DataPreprocessor()
    X_transformed, y_transformed = preprocessor.fit_transform(X, y)
    
    assert X_transformed.shape[0] == len(df)
    assert X_transformed.shape[1] == 2

