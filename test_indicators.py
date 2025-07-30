#!/usr/bin/env python3
"""Test script to check technical indicators functionality."""

import sys
import os
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

def test_technical_indicators(df):
    """Test technical indicators functionality."""
    print("=" * 60)
    print("TESTING: Technical Indicators")
    print("=" * 60)
    try:
        config = load_config("config/settings.yaml")
        print("Config loaded successfully")
        
        indicators = TechnicalIndicators(config)
        print("TechnicalIndicators created successfully")
        
        df_with_indicators = indicators.calculate_all_indicators(df)
        print(f"calculate_all_indicators returned: {type(df_with_indicators)}")
        print(f"Original columns: {len(df.columns)}")
        print(f"Columns with indicators: {len(df_with_indicators.columns)}")
        
        assert len(df_with_indicators.columns) > len(df.columns), "No indicators were added"
        print(f"OK: Indicators calculated: {len(df_with_indicators.columns) - len(df.columns)} new columns")
        return df_with_indicators, True
    except Exception as e:
        print(f"FAILED: Technical indicators failed: {e}")
        import traceback
        traceback.print_exc()
        return df, False

def main():
    print("Starting technical indicators test...")
    
    # Load data
    df, success = test_data_loading()
    if not success:
        print("Data loading failed, exiting...")
        return
    
    # Test technical indicators
    df_indicators, success = test_technical_indicators(df)
    print(f"Test completed. Success: {success}, Data shape: {df_indicators.shape if df_indicators is not None else 'None'}")

if __name__ == "__main__":
    main()
