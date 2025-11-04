"""
Comparison between Genetic Algorithm and Traditional Methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import time

from .genetic_algorithm import GeneticAlgorithm
from .traditional_methods import (
    CorrelationSelector,
    MutualInfoSelector,
    UnivariateSelector,
    RecursiveEliminationSelector,
    PCASelector
)


class MethodComparer:
    """Compare Genetic Algorithm with traditional feature selection methods."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 test_size: float = 0.2, random_state: int = 42):
        """
        Initialize comparer.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Test set size
            random_state: Random seed
        """
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        self.results = {}
    
    def evaluate_method(self, method_name: str, selected_features: np.ndarray,
                       fit_time: float) -> Dict:
        """
        Evaluate a feature selection method.
        
        Args:
            method_name: Name of the method
            selected_features: Indices of selected features
            fit_time: Time taken to fit the method
        
        Returns:
            Dictionary with evaluation metrics
        """
        if len(selected_features) == 0:
            return {
                'method': method_name,
                'n_features': 0,
                'fit_time': fit_time,
                'test_accuracy': 0.0,
                'test_f1': 0.0,
                'cv_score': 0.0,
                'selected_features': []
            }
        
        # Select features
        X_train_selected = self.X_train[:, selected_features]
        X_test_selected = self.X_test[:, selected_features]
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        model.fit(X_train_selected, self.y_train)
        
        # Test predictions
        y_pred = model.predict(X_test_selected)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        test_f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        # Cross-validation score
        scorer = make_scorer(f1_score, average='weighted')
        cv_scores = cross_val_score(model, X_train_selected, self.y_train,
                                   cv=5, scoring=scorer)
        cv_score = cv_scores.mean()
        
        return {
            'method': method_name,
            'n_features': len(selected_features),
            'fit_time': fit_time,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'cv_score': cv_score,
            'selected_features': selected_features.tolist()
        }
    
    def compare_all(self, ga_params: Dict = None) -> Dict:
        """
        Compare all methods.
        
        Args:
            ga_params: Parameters for genetic algorithm
        
        Returns:
            Dictionary with all comparison results
        """
        if ga_params is None:
            ga_params = {
                'population_size': 30,
                'n_generations': 20,
                'random_state': self.random_state
            }
        
        self.results = {}
        
        # 1. Genetic Algorithm
        print("Running Genetic Algorithm...")
        start_time = time.time()
        ga = GeneticAlgorithm(self.X_train, self.y_train, **ga_params)
        ga_result = ga.run(verbose=False)
        ga_time = time.time() - start_time
        
        self.results['genetic_algorithm'] = self.evaluate_method(
            'Genetic Algorithm',
            ga_result['selected_features'],
            ga_time
        )
        self.results['genetic_algorithm']['history'] = ga_result['history']
        
        # 2. Correlation-based
        print("Running Correlation-based selection...")
        start_time = time.time()
        corr_selector = CorrelationSelector(k=min(20, self.X_train.shape[1] // 2))
        corr_selector.fit(self.X_train, self.y_train)
        corr_time = time.time() - start_time
        
        self.results['correlation'] = self.evaluate_method(
            'Correlation',
            corr_selector.get_selected_features(),
            corr_time
        )
        
        # 3. Mutual Information
        print("Running Mutual Information selection...")
        start_time = time.time()
        mi_selector = MutualInfoSelector(k=min(20, self.X_train.shape[1] // 2))
        mi_selector.fit(self.X_train, self.y_train)
        mi_time = time.time() - start_time
        
        self.results['mutual_information'] = self.evaluate_method(
            'Mutual Information',
            mi_selector.get_selected_features(),
            mi_time
        )
        
        # 4. Univariate Statistical Tests
        print("Running Univariate selection...")
        start_time = time.time()
        univ_selector = UnivariateSelector(k=min(20, self.X_train.shape[1] // 2))
        univ_selector.fit(self.X_train, self.y_train)
        univ_time = time.time() - start_time
        
        self.results['univariate'] = self.evaluate_method(
            'Univariate',
            univ_selector.get_selected_features(),
            univ_time
        )
        
        # 5. Recursive Feature Elimination
        print("Running RFE...")
        start_time = time.time()
        rfe_selector = RecursiveEliminationSelector(
            n_features_to_select=min(20, self.X_train.shape[1] // 2)
        )
        rfe_selector.fit(self.X_train, self.y_train)
        rfe_time = time.time() - start_time
        
        self.results['rfe'] = self.evaluate_method(
            'RFE',
            rfe_selector.get_selected_features(),
            rfe_time
        )
        
        # 6. PCA (feature importance-based selection from loadings)
        print("Running PCA...")
        start_time = time.time()
        # Keep components to explain 95% variance
        pca_selector = PCASelector(n_components=0.95)
        pca_selector.fit(self.X_train, self.y_train)
        pca_time = time.time() - start_time
        
        self.results['pca'] = self.evaluate_method(
            'PCA',
            pca_selector.get_selected_features(),
            pca_time
        )
        
        return self.results
    
    def get_summary(self) -> pd.DataFrame:
        """Get comparison summary as DataFrame."""
        import pandas as pd
        
        summary_data = []
        for method_name, result in self.results.items():
            summary_data.append({
                'Method': result['method'],
                'N_Features': result['n_features'],
                'Fit_Time (s)': result['fit_time'],
                'Test_Accuracy': result['test_accuracy'],
                'Test_F1': result['test_f1'],
                'CV_Score': result['cv_score']
            })
        
        return pd.DataFrame(summary_data)

