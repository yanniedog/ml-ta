#!/usr/bin/env python3
"""
Test to check if logging setup is interfering with print statements.
"""

import sys

print("BEFORE logging setup - this should appear")
sys.stdout.flush()

try:
    from src.utils import load_config, setup_logging, set_deterministic_seed
    
    print("AFTER imports, BEFORE config load")
    sys.stdout.flush()
    
    config = load_config("config/settings.yaml")
    
    print("AFTER config load, BEFORE setup_logging")
    sys.stdout.flush()
    
    setup_logging(config)
    
    print("AFTER setup_logging - this might not appear if logging interferes")
    sys.stdout.flush()
    
    set_deterministic_seed(config.app["seed"])
    
    print("AFTER set_deterministic_seed")
    sys.stdout.flush()
    
    print("=" * 60)
    print("COMPREHENSIVE ML TRADING SYSTEM TEST")
    print("=" * 60)
    sys.stdout.flush()
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()

print("End of logging test")
sys.stdout.flush()
