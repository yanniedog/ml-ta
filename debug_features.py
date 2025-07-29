#!/usr/bin/env python3
"""
Debug script to identify feature alignment issues.
"""
import pandas as pd
import numpy as np
from src.utils import load_config
from src.data import DataLoader
from src.features import FeatureEngineer
from src.model import ModelTrainer

def debug_feature_alignment():
    """Debug feature alignment between training and prediction."""
    print("=" * 60)
    print("DEBUGGING: Feature Alignment Issue")
    print("=" * 60)
    
    try:
        # Load configuration
        config = load_config("config/settings.yaml")
        
        # Load data
        data_loader = DataLoader(config)
        df = data_loader.load_gold_data("SOLUSDT", "1m")
        
        if df.empty:
            print("❌ No data loaded")
            return
        
        print(f"✅ Data loaded: {len(df)} rows")
        
        # Create feature engineer
        feature_engineer = FeatureEngineer(config)
        
        # Split data for training
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        print(f"✅ Data split: {len(train_df)} train, {len(test_df)} test")
        
        # Build training features
        print("\n🔧 Building training features...")
        train_features = feature_engineer.build_feature_matrix(train_df, fit_pipeline=True)
        print(f"✅ Training features built: {train_features.shape}")
        
        # Get feature columns from training
        exclude_columns = ['timestamp']
        label_columns = [col for col in train_features.columns if col.startswith('label_')]
        return_columns = [col for col in train_features.columns if col.startswith('return_')]
        exclude_columns.extend(label_columns)
        exclude_columns.extend(return_columns)
        
        training_feature_columns = [col for col in train_features.columns if col not in exclude_columns]
        print(f"✅ Training feature columns ({len(training_feature_columns)}):")
        for i, col in enumerate(training_feature_columns[:10]):
            print(f"   {i+1:2d}. {col}")
        if len(training_feature_columns) > 10:
            print(f"   ... and {len(training_feature_columns) - 10} more")
        
        # Build test features
        print("\n🔧 Building test features...")
        test_features = feature_engineer.build_feature_matrix(test_df, fit_pipeline=False)
        print(f"✅ Test features built: {test_features.shape}")
        
        # Get test feature columns
        test_feature_columns = [col for col in test_features.columns if col not in exclude_columns]
        print(f"✅ Test feature columns ({len(test_feature_columns)}):")
        for i, col in enumerate(test_feature_columns[:10]):
            print(f"   {i+1:2d}. {col}")
        if len(test_feature_columns) > 10:
            print(f"   ... and {len(test_feature_columns) - 10} more")
        
        # Check for mismatches
        missing_in_test = set(training_feature_columns) - set(test_feature_columns)
        extra_in_test = set(test_feature_columns) - set(training_feature_columns)
        
        print(f"\n🔍 Feature Analysis:")
        print(f"   Training features: {len(training_feature_columns)}")
        print(f"   Test features: {len(test_feature_columns)}")
        print(f"   Missing in test: {len(missing_in_test)}")
        print(f"   Extra in test: {len(extra_in_test)}")
        
        if missing_in_test:
            print(f"\n❌ Missing features in test:")
            for col in list(missing_in_test)[:5]:
                print(f"   - {col}")
            if len(missing_in_test) > 5:
                print(f"   ... and {len(missing_in_test) - 5} more")
        
        if extra_in_test:
            print(f"\n⚠️ Extra features in test:")
            for col in list(extra_in_test)[:5]:
                print(f"   - {col}")
            if len(extra_in_test) > 5:
                print(f"   ... and {len(extra_in_test) - 5} more")
        
        # Try to fix the alignment
        print(f"\n🔧 Attempting to fix feature alignment...")
        aligned_test_features = feature_engineer.ensure_feature_consistency(test_features)
        print(f"✅ Aligned test features: {aligned_test_features.shape}")
        
        # Check if alignment worked
        aligned_feature_columns = [col for col in aligned_test_features.columns if col not in exclude_columns]
        missing_after_alignment = set(training_feature_columns) - set(aligned_feature_columns)
        
        if not missing_after_alignment:
            print(f"✅ Feature alignment successful!")
        else:
            print(f"❌ Feature alignment failed. Still missing: {len(missing_after_alignment)}")
            for col in list(missing_after_alignment)[:3]:
                print(f"   - {col}")
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_feature_alignment()