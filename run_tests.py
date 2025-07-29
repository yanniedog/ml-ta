#!/usr/bin/env python3
"""
Test runner for the technical analysis system.
"""
import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests."""
    print("Running technical analysis tests...")
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("Error: pytest not found. Please install it with: pip install pytest")
        return False
    
    # Run tests
    test_dir = Path("tests")
    if not test_dir.exists():
        print("Error: tests directory not found")
        return False
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v", 
            "--tb=short"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)