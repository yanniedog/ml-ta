#!/usr/bin/env python3
"""
Minimal test to isolate import issues.
"""

print("Starting minimal test...")

try:
    print("Testing basic imports...")
    import logging
    print("✓ logging imported")
    
    import time
    print("✓ time imported")
    
    import numpy as np
    print("✓ numpy imported")
    
    import pandas as pd
    print("✓ pandas imported")
    
    from pathlib import Path
    print("✓ pathlib imported")
    
    print("Testing src.utils import...")
    from src.utils import load_config
    print("✓ src.utils.load_config imported")
    
    print("Loading config...")
    config = load_config("config/settings.yaml")
    print("✓ config loaded")
    
    print("Testing src.data import...")
    from src.data import DataLoader
    print("✓ src.data.DataLoader imported")
    
    print("Creating DataLoader...")
    loader = DataLoader(config)
    print("✓ DataLoader created")
    
    print("Testing sample data creation...")
    df = loader.create_sample_data("SOLUSDT", "1m", 100)
    print(f"✓ Sample data created: {df.shape}")
    
    print("All tests passed!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
