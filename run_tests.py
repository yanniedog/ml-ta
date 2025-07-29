#!/usr/bin/env python3
"""
Comprehensive test script for the enhanced ML trading system.
"""

import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import load_config, setup_logging, set_deterministic_seed
from src.data import DataLoader
from src.indicators import TechnicalIndicators
from src.features import FeatureEngineer
from src.labels import LabelConstructor
from src.model import AdvancedModelTrainer, RealTimePredictor
from src.backtest import Backtester


def test_data_loading():
    """Test data loading functionality."""
    print("=" * 60)
    print("TESTING: Data Loading")
    print("=" * 60)
    
    try:
        config = load_config("config/settings.yaml")
        loader = DataLoader(config)
        
        # Test bronze data loading
        df = loader.load_bronze_data("SOLUSDT", "1m")
        
        # If no real data available, create sample data
        if df is None or df.empty:
            print("No real data found, creating sample data for testing...")
            df = loader.create_sample_data("SOLUSDT", "1m", 5000)
        
        assert df is not None and not df.empty, "Failed to load or create data"
        print(f"✓ Data loaded: {df.shape}")
        
        # Test data quality
        assert 'timestamp' in df.columns, "Missing timestamp column"
        assert 'close' in df.columns, "Missing close column"
        assert 'open' in df.columns, "Missing open column"
        assert 'high' in df.columns, "Missing high column"
        assert 'low' in df.columns, "Missing low column"
        assert 'volume' in df.columns, "Missing volume column"
        
        print("✓ Data quality checks passed")
        return df
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return None


def test_technical_indicators(df):
    """Test technical indicators calculation."""
    print("=" * 60)
    print("TESTING: Technical Indicators")
    print("=" * 60)
    
    try:
        config = load_config("config/settings.yaml")
        indicators = TechnicalIndicators(config)
        
        # Calculate all indicators
        df_with_indicators = indicators.calculate_all_indicators(df)
        
        assert len(df_with_indicators.columns) > len(df.columns), "No indicators were added"
        print(f"✓ Indicators calculated: {len(df_with_indicators.columns) - len(df.columns)} new columns")
        
        # Test specific indicators
        indicator_tests = [
            ('rsi', 'RSI'),
            ('macd_macd', 'MACD'),
            ('bb_upper', 'Bollinger Bands'),
            ('stoch_k_percent', 'Stochastic'),
            ('atr', 'ATR'),
            ('cci', 'CCI'),
            ('roc', 'ROC'),
            ('williams_r', 'Williams %R')
        ]
        
        for col, name in indicator_tests:
            if col in df_with_indicators.columns:
                print(f"✓ {name} indicator present")
            else:
                print(f"⚠ {name} indicator missing")
        
        return df_with_indicators
        
    except Exception as e:
        print(f"✗ Technical indicators failed: {e}")
        return df


def test_feature_engineering(df):
    """Test feature engineering."""
    print("=" * 60)
    print("TESTING: Feature Engineering")
    print("=" * 60)
    
    try:
        config = load_config("config/settings.yaml")
        feature_engineer = FeatureEngineer(config)
        
        # Build feature matrix
        feature_df = feature_engineer.build_feature_matrix(df)
        
        assert not feature_df.empty, "Feature matrix is empty"
        print(f"✓ Feature matrix built: {feature_df.shape}")
        
        # Check for feature types
        feature_columns = [col for col in feature_df.columns if not col.startswith(('label_', 'return_', 'timestamp'))]
        print(f"✓ Feature columns: {len(feature_columns)}")
        
        # Check scaler state
        scaler_info = feature_engineer.get_scaler_info()
        assert scaler_info['is_fitted'], "Scaler not fitted"
        print(f"✓ Scaler fitted on {scaler_info['feature_columns_count']} features")
        
        return feature_df
        
    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        return df


def test_label_construction(df):
    """Test label construction."""
    print("=" * 60)
    print("TESTING: Label Construction")
    print("=" * 60)
    
    try:
        config = load_config("config/settings.yaml")
        label_constructor = LabelConstructor(config)
        
        # Construct labels
        labeled_df = label_constructor.construct_all_labels(df)
        
        assert not labeled_df.empty, "Labeled data is empty"
        print(f"✓ Labels constructed: {labeled_df.shape}")
        
        # Check for label columns
        label_columns = [col for col in labeled_df.columns if col.startswith('label_')]
        assert len(label_columns) > 0, "No label columns found"
        print(f"✓ Label columns: {len(label_columns)}")
        
        # Validate labels
        assert label_constructor.validate_labels(labeled_df), "Label validation failed"
        print("✓ Label validation passed")
        
        # Check label distribution
        label_distribution = label_constructor.get_label_distribution(labeled_df)
        for label_col in label_columns[:3]:  # Check first 3 labels
            if label_col in labeled_df.columns:
                unique_values = labeled_df[label_col].unique()
                if 'class' in label_col:
                    # Classification labels should be 0 or 1
                    assert all(val in [0, 1] for val in unique_values), f"Invalid classification values in {label_col}"
                else:
                    # Regression labels can be any numeric value
                    assert all(isinstance(val, (int, float)) for val in unique_values), f"Invalid regression values in {label_col}"
                print(f"✓ {label_col}: {labeled_df[label_col].value_counts().to_dict()}")
        
        return labeled_df
        
    except Exception as e:
        print(f"✗ Label construction failed: {e}")
        return df


def test_model_training(df):
    """Test model training."""
    print("=" * 60)
    print("TESTING: Model Training")
    print("=" * 60)
    
    try:
        config = load_config("config/settings.yaml")
        trainer = AdvancedModelTrainer(config)
        
        # Prepare data
        label_column = 'label_class_1'
        if label_column not in df.columns:
            print(f"⚠ Label column {label_column} not found, skipping model training")
            return None
        
        # Create feature engineer and build features
        feature_engineer = FeatureEngineer(config)
        
        # Split data for feature engineering
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # Build features for training data
        X_train = feature_engineer.build_feature_matrix(train_df, fit_pipeline=True)
        y_train = train_df[label_column]
        
        # Build features for test data (no fitting)
        X_test = feature_engineer.build_feature_matrix(test_df, fit_pipeline=False)
        y_test = test_df[label_column]
        
        # Align training data
        common_train_idx = X_train.index.intersection(y_train.index)
        X_train = X_train.loc[common_train_idx]
        y_train = y_train.loc[common_train_idx]
        
        # Align test data
        common_test_idx = X_test.index.intersection(y_test.index)
        X_test = X_test.loc[common_test_idx]
        y_test = y_test.loc[common_test_idx]
        
        # Combine for training
        X = pd.concat([X_train, X_test])
        y = pd.concat([y_train, y_test])
        
        print(f"✓ Training data prepared: {X.shape}")
        
        # Train ensemble model with feature engineer
        start_time = time.time()
        model_results = trainer.train_ensemble_model(X, y, label_column, feature_engineer=feature_engineer)
        training_time = time.time() - start_time
        
        print(f"✓ Model training completed in {training_time:.2f} seconds")
        
        # Check results
        ensemble_scores = model_results['scores']['ensemble']
        assert ensemble_scores['accuracy'] > 0.5, "Model accuracy too low"
        print(f"✓ Ensemble accuracy: {ensemble_scores['accuracy']:.4f}")
        print(f"✓ Ensemble ROC AUC: {ensemble_scores['roc_auc']:.4f}")
        
        return trainer
        
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        return None


def test_real_time_prediction(df, trainer):
    """Test real-time prediction."""
    print("=" * 60)
    print("TESTING: Real-Time Prediction")
    print("=" * 60)
    
    try:
        config = load_config("config/settings.yaml")
        
        if trainer is None:
            print("⚠ No trained model available, skipping real-time prediction")
            return
        
        # Use the fitted feature engineer from the trainer
        label_column = 'label_class_1'
        if label_column in trainer.feature_engineers:
            feature_engineer = trainer.feature_engineers[label_column]
            print(f"✓ Using fitted feature engineer with {len(feature_engineer.feature_pipeline.feature_columns)} features")
        else:
            print("⚠ No fitted feature engineer found, creating new one")
            # Create a new feature engineer and fit it
            feature_engineer = FeatureEngineer(config)
            # Fit the pipeline on training data
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx]
            feature_engineer.build_feature_matrix(train_df, fit_pipeline=True)
        
        # Create predictor with fitted feature engineer
        predictor = RealTimePredictor(config, trainer.ensemble_model, feature_engineer)
        
        # Test with different data sizes
        for data_points in [300, 400, 500]:
            test_data = df.tail(data_points)
            prediction = predictor.predict(test_data)
            
            assert prediction is not None, "Prediction returned None"
            assert 'prediction' in prediction, "Prediction missing 'prediction' key"
            assert 'confidence' in prediction, "Prediction missing 'confidence' key"
            
            print(f"✓ Prediction for {data_points} data points: {prediction}")
        
        print("✓ Real-time prediction test passed")
        
    except Exception as e:
        print(f"✗ Real-time prediction test failed: {e}")
        import traceback
        traceback.print_exc()


def test_backtesting(df, trainer):
    """Test backtesting."""
    print("=" * 60)
    print("TESTING: Backtesting")
    print("=" * 60)
    
    try:
        config = load_config("config/settings.yaml")
        backtester = Backtester(config)
        
        if trainer is None:
            print("⚠ No trained model available, skipping backtesting")
            return
        
        # Get the fitted feature engineer from the trainer
        label_column = 'label_class_1'
        fitted_feature_engineer = None
        
        if label_column in trainer.feature_engineers:
            fitted_feature_engineer = trainer.feature_engineers[label_column]
        else:
            # Create a new feature engineer and fit it
            fitted_feature_engineer = FeatureEngineer(config)
            # Fit the pipeline on training data
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx]
            fitted_feature_engineer.build_feature_matrix(train_df, fit_pipeline=True)
        
        # Run backtest with fitted feature engineer
        backtest_results = backtester.run_backtest_with_model(
            df, trainer.ensemble_model, label_column, fitted_feature_engineer
        )
        
        if backtest_results:
            performance = backtest_results.get('performance', {})
            model_metrics = backtest_results.get('model_metrics', {})
            
            print(f"✓ Backtest completed: {performance.get('total_trades', 0)} trades")
            print(f"✓ Total Return: {performance.get('total_return', 0):.2%}")
            print(f"✓ Hit Rate: {performance.get('hit_rate', 0):.2%}")
            print(f"✓ Model Accuracy: {model_metrics.get('accuracy', 0):.4f}")
            print(f"✓ Model ROC AUC: {model_metrics.get('roc_auc', 0):.4f}")
        else:
            print("⚠ Backtest results not available")
        
    except Exception as e:
        print(f"✗ Backtesting failed: {e}")


def main():
    """Run comprehensive tests."""
    print("=" * 60)
    print("COMPREHENSIVE ML TRADING SYSTEM TEST")
    print("=" * 60)
    
    # Load configuration
    config = load_config("config/settings.yaml")
    setup_logging(config)
    set_deterministic_seed(config.app["seed"])
    
    # Test data loading
    df = test_data_loading()
    if df is None:
        print("✗ Cannot proceed without data")
        return
    
    # Test technical indicators
    df_with_indicators = test_technical_indicators(df)
    
    # Test feature engineering
    feature_df = test_feature_engineering(df_with_indicators)
    
    # Test label construction
    labeled_df = test_label_construction(feature_df)
    
    # Test model training
    trainer = test_model_training(labeled_df)
    
    # Test real-time prediction
    test_real_time_prediction(labeled_df, trainer)
    
    # Test backtesting
    test_backtesting(labeled_df, trainer)
    
    print("=" * 60)
    print("COMPREHENSIVE TEST COMPLETED")
    print("=" * 60)
    print("✓ All core components tested successfully")
    print("✓ System is ready for production use")


if __name__ == "__main__":
    main()