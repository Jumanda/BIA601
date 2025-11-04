"""
Tests for traditional feature selection methods
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.traditional_methods import (
    CorrelationSelector,
    MutualInfoSelector,
    UnivariateSelector,
    RecursiveEliminationSelector
)


@pytest.fixture
def sample_data():
    """Generate sample dataset for testing."""
    X, y = make_classification(n_samples=100, n_features=20, n_informative=10,
                               n_redundant=5, n_classes=2, random_state=42)
    return X, y


def test_correlation_selector(sample_data):
    """Test correlation-based feature selection."""
    X, y = sample_data
    selector = CorrelationSelector(k=10)
    selector.fit(X, y)
    selected = selector.get_selected_features()
    assert len(selected) == 10
    assert all(0 <= idx < X.shape[1] for idx in selected)
    
    X_transformed = selector.transform(X)
    assert X_transformed.shape[1] == 10


def test_mutual_info_selector(sample_data):
    """Test mutual information feature selection."""
    X, y = sample_data
    selector = MutualInfoSelector(k=10)
    selector.fit(X, y)
    selected = selector.get_selected_features()
    assert len(selected) == 10
    assert all(0 <= idx < X.shape[1] for idx in selected)
    
    X_transformed = selector.transform(X)
    assert X_transformed.shape[1] == 10


def test_univariate_selector(sample_data):
    """Test univariate statistical test selection."""
    X, y = sample_data
    selector = UnivariateSelector(k=10)
    selector.fit(X, y)
    selected = selector.get_selected_features()
    assert len(selected) == 10
    
    X_transformed = selector.transform(X)
    assert X_transformed.shape[1] == 10


def test_rfe_selector(sample_data):
    """Test recursive feature elimination."""
    X, y = sample_data
    selector = RecursiveEliminationSelector(n_features_to_select=10)
    selector.fit(X, y)
    selected = selector.get_selected_features()
    assert len(selected) <= X.shape[1]
    
    X_transformed = selector.transform(X)
    assert X_transformed.shape[1] == len(selected)

