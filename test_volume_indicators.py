#!/usr/bin/env python3
"""Test script to check volume indicators functionality."""

import sys
import os
import time
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, setup_logging, set_deterministic_seed
from src.data import DataLoader
from src.indicators import TechnicalIndicators

def main():
    print("Starting volume indicators test...")
    
    # Load data
    config = load_config("config/settings.yaml")
    loader = DataLoader(config)
    df = loader.load_bronze_data("SOLUSDT", "1m")
    if df is None or df.empty:
        print("No real data found, creating sample data for testing...")
        df = loader.create_sample_data("SOLUSDT", "1m", 5000)
    
    print(f"Data loaded: {df.shape}")
    print(f"Data index type: {type(df.index)}")
    print(f"Data index: {df.index[:5]}")
    
    # Check if index has date attribute
    print(f"Index has date attribute: {hasattr(df.index, 'date')}")
    if hasattr(df.index, 'date'):
        try:
            print(f"First few dates: {df.index.date[:5]}")
        except Exception as e:
            print(f"Error accessing date attribute: {e}")
    
    # Create indicators instance
    indicators = TechnicalIndicators(config)
    
    # Test volume indicators
    print("\nTesting volume indicators...")
    start_time = time.time()
    try:
        result_df = indicators._add_volume_indicators(df)
        end_time = time.time()
        print(f"SUCCESS: Volume indicators completed in {end_time - start_time:.2f} seconds")
        print(f"Result shape: {result_df.shape}")
    except Exception as e:
        end_time = time.time()
        print(f"FAILED: Volume indicators failed after {end_time - start_time:.2f} seconds: {e}")
        import traceback
        traceback.print_exc()
    
    print("Volume indicators test completed.")

if __name__ == "__main__":
    main()
