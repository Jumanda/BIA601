#!/usr/bin/env python3
"""
Quick test script for PCA implementation
Use this to verify that PCA works correctly before integration
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.traditional_methods import PCASelector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score

def test_pca_classification():
    """Test PCA with classification task"""
    print("=" * 50)
    print("Testing PCA with Classification")
    print("=" * 50)
    
    # Load sample data
    data_path = Path(__file__).parent / 'data' / 'sample_dataset.csv'
    if not data_path.exists():
        print(f"Warning: {data_path} not found. Using synthetic data.")
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)
    else:
        df = pd.read_csv(data_path)
        # Assuming last column is target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test PCA with variance threshold
    print("\n1. Testing PCA with variance threshold (0.95)...")
    pca1 = PCASelector(n_components=0.95)
    pca1.fit(X_train, y_train)
    
    print(f"   Selected {len(pca1.get_selected_features())} features out of {X_train.shape[1]}")
    print(f"   Total explained variance: {pca1.get_total_explained_variance():.2%}")
    
    # Test PCA with fixed number of components
    print("\n2. Testing PCA with fixed number of components (10)...")
    pca2 = PCASelector(n_components=10)
    pca2.fit(X_train, y_train)
    
    print(f"   Selected {len(pca2.get_selected_features())} features")
    print(f"   Number of principal components: {pca2.pca_.n_components_}")
    
    # Evaluate performance
    print("\n3. Evaluating model performance...")
    # Use selected features
    X_train_selected = pca1.transform_selected_features(X_train)
    X_test_selected = pca1.transform_selected_features(X_test)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    
    return pca1

def test_pca_regression():
    """Test PCA with regression task"""
    print("\n" + "=" * 50)
    print("Testing PCA with Regression")
    print("=" * 50)
    
    # Generate synthetic regression data
    np.random.seed(42)
    X = np.random.randn(100, 20)
    y = np.random.randn(100)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test PCA
    print("\n1. Testing PCA with variance threshold...")
    pca = PCASelector(n_components=0.95)
    pca.fit(X_train, y_train)
    
    print(f"   Selected {len(pca.get_selected_features())} features")
    print(f"   Total explained variance: {pca.get_total_explained_variance():.2%}")
    
    # Evaluate performance
    print("\n2. Evaluating model performance...")
    X_train_selected = pca.transform_selected_features(X_train)
    X_test_selected = pca.transform_selected_features(X_test)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    
    r2 = r2_score(y_test, y_pred)
    print(f"   R2 Score: {r2:.4f}")
    
    return pca

if __name__ == '__main__':
    try:
        pca_classifier = test_pca_classification()
        pca_regressor = test_pca_regression()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        print("=" * 50)
        print("\nPCA is ready for integration with the team's code.")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



