#!/usr/bin/env python3
"""
Script to create a sample dataset for testing
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_dataset(n_samples=200, n_features=30, output_file='sample_dataset.csv'):
    """
    Create a sample dataset for feature selection testing.
    
    Args:
        n_samples: Number of samples (rows)
        n_features: Number of features (columns)
        output_file: Output filename
    """
    print(f"Creating sample dataset with {n_samples} samples and {n_features} features...")
    
    # Create informative features (some are more important than others)
    np.random.seed(42)
    
    # Generate features
    data = {}
    for i in range(n_features):
        # Some features are more informative
        if i < 10:  # First 10 features are informative
            data[f'informative_feature_{i+1}'] = np.random.randn(n_samples) + (i % 3)
        else:  # Others are less informative
            data[f'noise_feature_{i+1}'] = np.random.randn(n_samples)
    
    # Create target based on informative features
    # Target depends on first 5 features
    target = np.zeros(n_samples)
    for i in range(n_samples):
        if data['informative_feature_1'][i] + data['informative_feature_2'][i] > 0:
            target[i] = 1
        elif data['informative_feature_3'][i] + data['informative_feature_4'][i] > 1:
            target[i] = 1
        else:
            target[i] = 0
    
    data['target'] = target.astype(int)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = Path('data') / output_file
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Dataset created successfully!")
    print(f"   File: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Target distribution:")
    print(f"      Class 0: {np.sum(target == 0)}")
    print(f"      Class 1: {np.sum(target == 1)}")
    print(f"\n📁 You can now upload this file to the web interface!")
    
    return output_path

if __name__ == '__main__':
    import sys
    
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_features = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    create_sample_dataset(n_samples=n_samples, n_features=n_features)

