#!/usr/bin/env python3
"""
Simple test to isolate the Series comparison issue.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import load_config
from src.indicators import TechnicalIndicators

def test_indicators():
    """Test indicators calculation."""
    print("Testing indicators calculation...")
    
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
        indicators = TechnicalIndicators(config)
        
        # Test individual indicators
        print("Testing RSI...")
        rsi = indicators.rsi(df['close'])
        print(f"RSI shape: {rsi.shape}")
        
        print("Testing MACD...")
        macd = indicators.macd(df['close'])
        print(f"MACD shape: {macd.shape}")
        
        print("Testing OBV...")
        obv = indicators.obv(df['close'], df['volume'])
        print(f"OBV shape: {obv.shape}")
        
        print("Testing MFI...")
        mfi = indicators.mfi(df['high'], df['low'], df['close'], df['volume'])
        print(f"MFI shape: {mfi.shape}")
        
        print("Testing ADX...")
        adx = indicators.adx(df['high'], df['low'], df['close'])
        print(f"ADX shape: {adx.shape}")
        
        print("Testing all indicators...")
        result = indicators.calculate_all_indicators(df)
        print(f"All indicators shape: {result.shape}")
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_indicators() 