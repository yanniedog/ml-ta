#!/usr/bin/env python3
"""
Simple test to isolate the feature engineering issue.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import load_config
from src.features import FeatureEngineer

def test_feature_engineering():
    """Test feature engineering."""
    print("Testing feature engineering...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='1min')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    print(f"Sample data shape: {df.shape}")
    
    try:
        config = load_config("config/settings.yaml")
        feature_engineer = FeatureEngineer(config)
        
        print("Building feature matrix...")
        feature_matrix = feature_engineer.build_feature_matrix(df, fit_scaler=True)
        print(f"Feature matrix shape: {feature_matrix.shape}")
        
        print("✅ Feature engineering test passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_feature_engineering() 