#!/usr/bin/env python3
"""Simple test script to check individual feature engineering steps."""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, setup_logging, set_deterministic_seed
from src.data import DataLoader
from src.indicators import TechnicalIndicators
from src.features import FeatureEngineer

def main():
    print("Starting simple feature steps test...")
    
    # Load data
    config = load_config("config/settings.yaml")
    loader = DataLoader(config)
    df = loader.load_bronze_data("SOLUSDT", "1m")
    if df is None or df.empty:
        print("No real data found, creating sample data for testing...")
        df = loader.create_sample_data("SOLUSDT", "1m", 5000)
    
    print(f"1. Data loaded: {df.shape}")
    
    # Calculate indicators
    print("2. Calculating indicators...")
    start_time = time.time()
    indicators = TechnicalIndicators(config)
    df_with_indicators = indicators.calculate_all_indicators(df)
    end_time = time.time()
    print(f"   Indicators calculated in {end_time - start_time:.2f} seconds: {df_with_indicators.shape}")
    
    # Create feature engineer
    feature_engineer = FeatureEngineer(config)
    
    # Test individual feature steps
    result_df = df_with_indicators.copy()
    
    # Test regime flags
    print("3. Adding regime flags...")
    start_time = time.time()
    result_df = feature_engineer.add_regime_flags(result_df)
    end_time = time.time()
    print(f"   Regime flags added in {end_time - start_time:.2f} seconds: {result_df.shape}")
    
    # Test lags
    print("4. Adding lagged features...")
    start_time = time.time()
    result_df = feature_engineer.add_lags(result_df)
    end_time = time.time()
    print(f"   Lagged features added in {end_time - start_time:.2f} seconds: {result_df.shape}")
    
    # Test rolling z-scores
    print("5. Adding rolling z-scores...")
    start_time = time.time()
    result_df = feature_engineer.add_rolling_z_scores(result_df)
    end_time = time.time()
    print(f"   Rolling z-scores added in {end_time - start_time:.2f} seconds: {result_df.shape}")
    
    # Test feature interactions
    print("6. Adding feature interactions...")
    start_time = time.time()
    result_df = feature_engineer.add_interactions(result_df)
    end_time = time.time()
    print(f"   Feature interactions added in {end_time - start_time:.2f} seconds: {result_df.shape}")
    
    print(f"\nAll feature steps completed successfully. Final shape: {result_df.shape}")

if __name__ == "__main__":
    main()
