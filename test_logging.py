#!/usr/bin/env python3
"""Test script to check logging functionality."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, setup_logging, set_deterministic_seed

def main():
    print("Starting logging test...")
    
    # Load configuration
    config = load_config("config/settings.yaml")
    print("Config loaded successfully")
    
    # Setup logging
    logger = setup_logging(config)
    print("Logging setup completed")
    
    # Test logging
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    print("All logging tests completed")

if __name__ == "__main__":
    main()
