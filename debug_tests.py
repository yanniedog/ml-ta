#!/usr/bin/env python3
"""Debug script to run tests individually."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, setup_logging, set_deterministic_seed
from src.data import DataLoader
from src.indicators import TechnicalIndicators
from src.features import FeatureEngineer
from src.labels import LabelConstructor
from src.model import ModelTrainer
from src.backtest import Backtester

from run_tests import (
    test_data_loading,
    test_technical_indicators,
    test_feature_engineering,
    test_label_construction,
    test_model_training,
    test_real_time_prediction,
    test_backtesting
)

def main():
    print("=" * 60)
    print("DEBUG TEST SUITE")
    print("=" * 60)
    
    # Test data loading
    print("\n--- Running test_data_loading ---")
    df, success = test_data_loading()
    print(f"Data loading success: {success}")
    if not success:
        print("Data loading failed")
        return
    print("--- Finished test_data_loading ---")
    
    # Test technical indicators
    print("\n--- Running test_technical_indicators ---")
    df_indicators, success = test_technical_indicators(df)
    print(f"Technical indicators success: {success}")
    if not success:
        print("Technical indicators failed")
        return
    print("--- Finished test_technical_indicators ---")
    
    # Test feature engineering
    print("\n--- Running test_feature_engineering ---")
    df_features, success = test_feature_engineering(df_indicators)
    print(f"Feature engineering success: {success}")
    if not success:
        print("Feature engineering failed")
        return
    print("--- Finished test_feature_engineering ---")
    
    # Test label construction
    print("\n--- Running test_label_construction ---")
    df_labels, success = test_label_construction(df_features)
    print(f"Label construction success: {success}")
    if not success:
        print("Label construction failed")
        return
    print("--- Finished test_label_construction ---")
    
    # Test model training
    print("\n--- Running test_model_training ---")
    trainer, success = test_model_training(df_labels)
    print(f"Model training success: {success}")
    if not success:
        print("Model training failed")
        return
    print("--- Finished test_model_training ---")
    
    # Test real-time prediction
    print("\n--- Running test_real_time_prediction ---")
    success = test_real_time_prediction(df_labels, trainer)
    print(f"Real-time prediction success: {success}")
    if not success:
        print("Real-time prediction failed")
        return
    print("--- Finished test_real_time_prediction ---")
    
    # Test backtesting
    print("\n--- Running test_backtesting ---")
    success = test_backtesting(df_labels, trainer)
    print(f"Backtesting success: {success}")
    if not success:
        print("Backtesting failed")
        return
    print("--- Finished test_backtesting ---")
    
    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
