#!/usr/bin/env python3
"""Test script to check feature engineering functionality."""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, setup_logging, set_deterministic_seed
from src.data import DataLoader
from src.indicators import TechnicalIndicators
from src.features import FeatureEngineer

def main():
    print("Starting feature engineering test...")
    
    # Load data
    config = load_config("config/settings.yaml")
    loader = DataLoader(config)
    df = loader.load_bronze_data("SOLUSDT", "1m")
    if df is None or df.empty:
        print("No real data found, creating sample data for testing...")
        df = loader.create_sample_data("SOLUSDT", "1m", 5000)
    
    print(f"Data loaded: {df.shape}")
    
    # Calculate indicators
    print("Calculating indicators...")
    indicators = TechnicalIndicators(config)
    df_with_indicators = indicators.calculate_all_indicators(df)
    print(f"Indicators calculated: {df_with_indicators.shape}")
    
    # Test feature engineering
    print("\nTesting feature engineering...")
    start_time = time.time()
    try:
        feature_engineer = FeatureEngineer(config)
        feature_df = feature_engineer.build_feature_matrix(df_with_indicators)
        end_time = time.time()
        print(f"SUCCESS: Feature matrix built in {end_time - start_time:.2f} seconds")
        print(f"Result shape: {feature_df.shape}")
        print(f"Feature columns: {len(feature_engineer.feature_columns)}")
    except Exception as e:
        end_time = time.time()
        print(f"FAILED: Feature engineering failed after {end_time - start_time:.2f} seconds: {e}")
        import traceback
        traceback.print_exc()
    
    print("Feature engineering test completed.")

if __name__ == "__main__":
    main()
