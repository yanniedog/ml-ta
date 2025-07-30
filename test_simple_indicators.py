#!/usr/bin/env python3
"""Simple test script to check indicator performance."""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, setup_logging, set_deterministic_seed
from src.data import DataLoader
from src.indicators import TechnicalIndicators

def main():
    print("Starting simple indicator test...")
    
    # Load data
    config = load_config("config/settings.yaml")
    loader = DataLoader(config)
    df = loader.load_bronze_data("SOLUSDT", "1m")
    if df is None or df.empty:
        print("No real data found, creating sample data for testing...")
        df = loader.create_sample_data("SOLUSDT", "1m", 5000)
    
    print(f"Data loaded: {df.shape}")
    
    # Create indicators instance
    indicators = TechnicalIndicators(config)
    
    # Test individual indicator groups
    indicator_tests = [
        ("Moving Averages", indicators._add_moving_averages),
        ("MACD", indicators._add_macd),
        ("RSI", indicators._add_rsi),
        ("Stochastic", indicators._add_stochastic),
        ("Bollinger Bands", indicators._add_bollinger_bands),
        ("Parabolic SAR", indicators._add_parabolic_sar),
        ("Ichimoku", indicators._add_ichimoku),
        ("ATR", indicators._add_atr),
        ("CCI", indicators._add_cci),
        ("ROC", indicators._add_roc),
        ("Williams %R", indicators._add_williams_r),
        ("Keltner Channels", indicators._add_keltner_channels),
        ("SuperTrend", indicators._add_supertrend),
        ("DPO", indicators._add_dpo),
        ("TRIX", indicators._add_trix),
        ("Momentum", indicators._add_momentum),
        ("ADX", indicators._add_adx),
        ("Volume Indicators", indicators._add_volume_indicators)
    ]
    
    result_df = df.copy()
    
    for name, func in indicator_tests:
        print(f"\nTesting {name}...")
        start_time = time.time()
        try:
            result_df = func(result_df)
            end_time = time.time()
            print(f"  SUCCESS: {name} completed in {end_time - start_time:.2f} seconds")
            print(f"  New shape: {result_df.shape}")
        except Exception as e:
            end_time = time.time()
            print(f"  FAILED: {name} failed after {end_time - start_time:.2f} seconds: {e}")
            break
    
    print(f"\nAll tests completed. Final data shape: {result_df.shape}")

if __name__ == "__main__":
    main()
