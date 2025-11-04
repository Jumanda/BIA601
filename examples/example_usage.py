"""
Example usage of Genetic Algorithm for Feature Selection
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.data_preprocessing import DataPreprocessor
from src.genetic_algorithm import GeneticAlgorithm
from src.compare_methods import MethodComparer
from tqdm import tqdm

# Generate sample dataset
print("Generating sample dataset...")
X, y = make_classification(
    n_samples=500,
    n_features=50,
    n_informative=15,
    n_redundant=10,
    n_classes=2,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Dataset shape: {X.shape}")
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Method 1: Genetic Algorithm only
print("\n" + "="*60)
print("Running Genetic Algorithm...")
print("="*60)

# Create progress bar
pbar = tqdm(total=20, desc="Genetic Algorithm", unit="gen")

def update_progress(data):
    pbar.set_postfix({
        'Fitness': f"{data['best_fitness']:.3f}",
        'Features': data['best_n_features'],
        'Score': f"{data['model_score']:.3f}"
    })
    pbar.update(1)

ga = GeneticAlgorithm(
    X_train, y_train,
    population_size=30,
    n_generations=20,
    crossover_rate=0.8,
    mutation_rate=0.01,
    random_state=42,
    progress_callback=update_progress
)

ga_result = ga.run(verbose=False)
pbar.close()

print(f"\nBest solution:")
print(f"  Selected features: {len(ga_result['selected_features'])}/{X.shape[1]}")
print(f"  Best fitness: {ga_result['best_fitness']:.4f}")
print(f"  Model score: {ga_result['best_model_score']:.4f}")
print(f"  Feature indices: {ga_result['selected_features'][:10]}...")

# Method 2: Compare all methods
print("\n" + "="*60)
print("Comparing all methods...")
print("="*60)

# Progress bar for comparison
methods = ['Genetic Algorithm', 'Correlation', 'Mutual Information', 'Univariate', 'RFE']
method_pbar = tqdm(total=len(methods), desc="Comparing Methods", unit="method")

comparer = MethodComparer(X_train, y_train, random_state=42)

# Create progress callback for GA in comparison
def ga_progress_callback(data):
    method_pbar.set_postfix({'GA': f"Gen {data['generation']}/{data['total_generations']}"})

results = comparer.compare_all(ga_params={
    'population_size': 20,
    'n_generations': 15,
    'random_state': 42,
    'progress_callback': ga_progress_callback
})

method_pbar.update(1)  # Update for GA completion
for method in methods[1:]:
    method_pbar.update(1)
method_pbar.close()

# Print summary
print("\n" + "="*60)
print("Comparison Summary:")
print("="*60)

summary_df = comparer.get_summary()
print(summary_df.to_string(index=False))

print("\n" + "="*60)
print("Best method by metric:")
print("="*60)
print(f"  Highest Accuracy: {summary_df.loc[summary_df['Test_Accuracy'].idxmax(), 'Method']}")
print(f"  Highest F1 Score: {summary_df.loc[summary_df['Test_F1'].idxmax(), 'Method']}")
print(f"  Fewest Features: {summary_df.loc[summary_df['N_Features'].idxmin(), 'Method']}")
print(f"  Fastest: {summary_df.loc[summary_df['Fit_Time (s)'].idxmin(), 'Method']}")

