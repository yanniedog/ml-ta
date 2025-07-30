#!/usr/bin/env python3
"""
Targeted test to isolate which test function is causing the issue.
"""

print("Starting targeted test...")

try:
    # Import everything needed
    from src.utils import load_config, setup_logging, set_deterministic_seed
    from src.data import DataLoader
    
    print("✓ All imports successful")
    
    # Load config and setup
    config = load_config("config/settings.yaml")
    setup_logging(config)
    set_deterministic_seed(config.app["seed"])
    
    print("✓ Config loaded and logging setup")
    
    # Test just the data loading function
    print("\n=== Testing data loading function ===")
    
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
            print(f"✓ Data loaded: {df.shape}")
            return df, True
        except Exception as e:
            print(f"✗ Data loading failed: {e}")
            return None, False
    
    # Call the function
    df, success = test_data_loading()
    print(f"Data loading result: success={success}, df shape={df.shape if df is not None else 'None'}")
    
    if success:
        print("✓ Data loading test passed!")
    else:
        print("✗ Data loading test failed!")
    
except Exception as e:
    print(f"ERROR in targeted test: {e}")
    import traceback
    traceback.print_exc()

print("Targeted test completed.")
