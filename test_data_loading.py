#!/usr/bin/env python3
"""Test script to check data loading functionality."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, setup_logging, set_deterministic_seed
from src.data import DataLoader

def test_data_loading():
    """Test data loading functionality."""
    print("=" * 60)
    print("TESTING: Data Loading")
    print("=" * 60)
    try:
        config = load_config("config/settings.yaml")
        print("Config loaded successfully")
        
        loader = DataLoader(config)
        print("DataLoader created successfully")
        
        df = loader.load_bronze_data("SOLUSDT", "1m")
        print(f"load_bronze_data returned: {type(df)}")
        
        if df is None or df.empty:
            print("No real data found, creating sample data for testing...")
            df = loader.create_sample_data("SOLUSDT", "1m", 5000)
            print(f"create_sample_data returned: {type(df)}")
        
        print(f"Data check - df is None: {df is None}")
        print(f"Data check - df.empty: {df.empty if df is not None else 'N/A'}")
        
        assert df is not None and not df.empty, "Failed to load or create data"
        print(f"SUCCESS: Data loaded: {df.shape}")
        return df, True
    except Exception as e:
        print(f"FAILED: Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False

def main():
    print("Starting data loading test...")
    df, success = test_data_loading()
    print(f"Test completed. Success: {success}, Data shape: {df.shape if df is not None else 'None'}")

if __name__ == "__main__":
    main()
