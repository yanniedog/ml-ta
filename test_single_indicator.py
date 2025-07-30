#!/usr/bin/env python3
"""Test script to check individual indicator functionality."""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, setup_logging, set_deterministic_seed
from src.data import DataLoader
from src.indicators import TechnicalIndicators

def test_data_loading():
    """Test data loading functionality."""
    print("=" * 60)
    print("TESTING: Data Loading")
    print("=" * 60)
    try:
        config = load_config("config/settings.yaml")
        loader = DataLoader(config)
        df = loader.load_bronze_data("SOLUSDT", "1m")
        if df is None or df.empty:
            print("No real data found, creating sample data for testing...")
            df = loader.create_sample_data("SOLUSDT", "1m", 5000)
        assert df is not None and not df.empty, "Failed to load or create data"
        print(f"SUCCESS: Data loaded: {df.shape}")
        return df, True
    except Exception as e:
        print(f"FAILED: Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False

def test_single_indicator(df, indicator_name, indicator_func):
    """Test a single indicator functionality."""
    print(f"=" * 60)
    print(f"TESTING: {indicator_name}")
    print(f"=" * 60)
    try:
        config = load_config("config/settings.yaml")
        indicators = TechnicalIndicators(config)
        
        start_time = time.time()
        result_df = indicator_func(indicators, df)
        end_time = time.time()
        
        print(f"SUCCESS: {indicator_name} calculated in {end_time - start_time:.2f} seconds")
        print(f"Result shape: {result_df.shape}")
        return True
    except Exception as e:
        print(f"FAILED: {indicator_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sma(df):
    """Test SMA indicator."""
    config = load_config("config/settings.yaml")
    indicators = TechnicalIndicators(config)
    result_df = df.copy()
    result_df = indicators._add_moving_averages(result_df)
    return result_df

def test_ichimoku(df):
    """Test Ichimoku indicator."""
    config = load_config("config/settings.yaml")
    indicators = TechnicalIndicators(config)
    result_df = df.copy()
    result_df = indicators._add_ichimoku(result_df)
    return result_df

def test_parabolic_sar(df):
    """Test Parabolic SAR indicator."""
    config = load_config("config/settings.yaml")
    indicators = TechnicalIndicators(config)
    result_df = df.copy()
    result_df = indicators._add_parabolic_sar(result_df)
    return result_df

def main():
    print("Starting single indicator test...")
    
    # Load data
    df, success = test_data_loading()
    if not success:
        print("Data loading failed, exiting...")
        return
    
    # Test individual indicators
    print(f"Data shape: {df.shape}")
    
    # Test SMA
    success = test_single_indicator(df, "SMA", test_sma)
    if not success:
        print("SMA test failed")
        return
    
    # Test Ichimoku
    success = test_single_indicator(df, "Ichimoku", test_ichimoku)
    if not success:
        print("Ichimoku test failed")
        return
    
    # Test Parabolic SAR
    success = test_single_indicator(df, "Parabolic SAR", test_parabolic_sar)
    if not success:
        print("Parabolic SAR test failed")
        return
    
    print("All individual indicator tests completed successfully")

if __name__ == "__main__":
    main()
