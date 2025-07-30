#!/usr/bin/env python3
"""
Minimal version of run_tests.py to isolate the silent exit issue.
"""

def main():
    print("=== MINIMAL TEST RUNNER ===")
    
    try:
        # Basic imports
        print("1. Importing modules...")
        from src.utils import load_config, setup_logging, set_deterministic_seed
        print("2. Modules imported successfully")
        
        # Load config
        print("3. Loading config...")
        config = load_config("config/settings.yaml")
        print("4. Config loaded")
        
        # Setup logging
        print("5. Setting up logging...")
        setup_logging(config)
        print("6. Logging setup complete")
        
        # Set random seed
        print("7. Setting random seed...")
        set_deterministic_seed(config.app["seed"])
        print("8. Random seed set")
        
        # Test data loading
        print("9. Testing data loading...")
        from src.data import DataLoader
        loader = DataLoader(config)
        df = loader.load_bronze_data("SOLUSDT", "1m")
        if df is None or df.empty:
            print("  - No data found, creating sample data...")
            df = loader.create_sample_data("SOLUSDT", "1m", 100)
        print(f"10. Data loaded: {df.shape}")
        
        print("\n=== MINIMAL TEST COMPLETED SUCCESSFULLY ===")
        
    except Exception as e:
        print(f"\n=== ERROR IN MINIMAL TEST ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
