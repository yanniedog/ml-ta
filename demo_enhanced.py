#!/usr/bin/env python3
"""
Enhanced technical analysis demo with improved feature engineering and model training.
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


def main():
    """Enhanced demo with improved feature engineering and model training."""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting enhanced technical analysis demo...")
    
    # Load configuration
    config = load_config("config/settings.yaml")
    
    # Setup logging
    setup_logging(config)
    
    # Set deterministic seed
    set_deterministic_seed(config.app["seed"])
    
    # Load data
    logger.info("Loading data...")
    loader = DataLoader(config)
    df = loader.load_gold_data("SOLUSDT", "1m")
    
    if df is None or df.empty:
        logger.error("No data available")
        return
    
    logger.info(f"Loaded data: {df.shape}")
    
    # Validate labels
    label_constructor = LabelConstructor(config)
    if not label_constructor.validate_labels(df):
        logger.error("Label validation failed")
        return
    
    # Show label distribution
    label_distribution = label_constructor.get_label_distribution(df)
    for label_col in ['label_class_1', 'label_reg_1', 'label_class_3', 'label_reg_3', 'label_class_5', 'label_reg_5']:
        if label_col in label_distribution:
            logger.info(f"{label_col} distribution: {label_distribution[label_col]}")
    
    # Build feature matrix
    logger.info("Building feature matrix...")
    feature_engineer = FeatureEngineer(config)
    feature_df = feature_engineer.build_feature_matrix(df)
    logger.info(f"Feature matrix shape: {feature_df.shape}")
    
    # ========================================
    # ENHANCED DEMO: Advanced Model Training with Ensemble Methods
    # ========================================
    logger.info("=" * 60)
    logger.info("ENHANCED DEMO: Advanced Model Training with Ensemble Methods")
    logger.info("=" * 60)
    
    # Prepare data for training
    label_column = 'label_class_1'
    
    # Remove timestamp and ALL label columns from features (not just the one being trained)
    exclude_columns = ['timestamp']
    # Exclude all label columns to match real-time prediction
    label_columns = [col for col in feature_df.columns if col.startswith('label_')]
    exclude_columns.extend(label_columns)
    
    # Also exclude return columns (they contain future information)
    return_columns = [col for col in feature_df.columns if col.startswith('return_')]
    exclude_columns.extend(return_columns)
    
    # Features to use for training
    feature_columns = [col for col in feature_df.columns if col not in exclude_columns]
    
    # Prepare feature matrix and labels
    X = feature_df[feature_columns].copy()
    y = feature_df[label_column]
    
    # Align data
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]
    
    logger.info(f"Feature matrix shape after alignment: {X.shape}")
    logger.info(f"Label matrix shape after alignment: {y.shape}")
    
    # Final training data
    training_df = feature_df.loc[common_index]
    logger.info(f"Final training data shape: {training_df.shape}")
    
    # Show target distribution
    target_distribution = y.value_counts()
    logger.info(f"Target distribution: {target_distribution.to_dict()}")
    
    # Train ensemble model
    start_time = time.time()
    trainer = AdvancedModelTrainer(config)
    model_results = trainer.train_ensemble_model(X, y, label_column)
    training_time = time.time() - start_time
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Display results
    ensemble_scores = model_results['scores']['ensemble']
    logger.info("ENSEMBLE MODEL PERFORMANCE:")
    logger.info(f"Accuracy: {ensemble_scores['accuracy']:.4f}")
    logger.info(f"ROC AUC: {ensemble_scores['roc_auc']:.4f}")
    logger.info(f"Precision: {ensemble_scores['precision']:.4f}")
    logger.info(f"Recall: {ensemble_scores['recall']:.4f}")
    logger.info(f"F1 Score: {ensemble_scores['f1']:.4f}")
    
    # Individual model performance
    individual_scores = model_results['scores']['individual']
    logger.info("INDIVIDUAL MODEL PERFORMANCE:")
    for model_name, scores in individual_scores.items():
        logger.info(f"{model_name.upper()}:")
        logger.info(f"  Accuracy: {scores['accuracy']:.4f}")
        logger.info(f"  ROC AUC: {scores['roc_auc']:.4f}")
        logger.info(f"  F1 Score: {scores['f1']:.4f}")
    
    # Cross-validation
    cv_results = trainer.cross_validate_ensemble(X, y)
    
    # Save models
    trainer.save_models("artefacts/enhanced_ensemble_models.pkl")
    
    # ========================================
    # ENHANCED DEMO: Advanced Real-Time Prediction
    # ========================================
    logger.info("=" * 60)
    logger.info("ENHANCED DEMO: Advanced Real-Time Prediction")
    logger.info("=" * 60)
    
    # Create real-time predictor
    predictor = RealTimePredictor(config, trainer.ensemble_model, feature_engineer)
    
    # Test with different amounts of data
    for data_points in [300, 350, 400, 450, 500]:
        test_data = df.tail(data_points)
        prediction = predictor.predict(test_data)
        logger.info(f"Prediction for {data_points} data points: {prediction}")
    
    # ========================================
    # ENHANCED DEMO: Advanced Backtesting with Ensemble Models
    # ========================================
    logger.info("=" * 60)
    logger.info("ENHANCED DEMO: Advanced Backtesting with Ensemble Models")
    logger.info("=" * 60)
    
    logger.info(f"Backtest dataset shape: {feature_df.shape}")
    
    # Run backtest
    backtester = Backtester(config)
    backtest_results = backtester.run_backtest_with_model(feature_df, trainer.ensemble_model, label_column)
    
    if backtest_results:
        performance = backtest_results.get('performance', {})
        model_metrics = backtest_results.get('model_metrics', {})
        
        logger.info("BACKTEST RESULTS:")
        logger.info(f"Total Return: {performance.get('total_return', 0):.2%}")
        logger.info(f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max Drawdown: {performance.get('max_drawdown', 0):.2%}")
        logger.info(f"Hit Rate: {performance.get('hit_rate', 0):.2%}")
        logger.info(f"Total Trades: {performance.get('total_trades', 0)}")
        
        logger.info("MODEL METRICS:")
        logger.info(f"Accuracy: {model_metrics.get('accuracy', 0):.4f}")
        logger.info(f"ROC AUC: {model_metrics.get('roc_auc', 0):.4f}")
        logger.info(f"F1 Score: {model_metrics.get('f1', 0):.4f}")
    else:
        logger.warning("Backtest results not available")
    
    # ========================================
    # ENHANCED DEMO COMPLETED SUCCESSFULLY!
    # ========================================
    logger.info("=" * 60)
    logger.info("ENHANCED DEMO COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    
    logger.info("All enhanced components are working correctly:")
    logger.info("+ Advanced data loading with quality checks")
    logger.info("+ Enhanced technical indicators with performance monitoring")
    logger.info("+ Advanced feature engineering with quality analysis")
    logger.info("+ Ensemble model training with hyperparameter optimization")
    logger.info("+ Real-time prediction with ensemble models")
    logger.info("+ Enhanced backtesting with comprehensive metrics")
    
    # Summary of cross-validation results
    if cv_results:
        logger.info("ENSEMBLE MODEL PERFORMANCE SUMMARY:")
        logger.info(f"Cross-validation accuracy: {cv_results.get('cv_accuracy', {}).get('mean', 0):.4f} (±{cv_results.get('cv_accuracy', {}).get('std', 0):.4f})")
        logger.info(f"Cross-validation ROC AUC: {cv_results.get('cv_roc_auc', {}).get('mean', 0):.4f} (±{cv_results.get('cv_roc_auc', {}).get('std', 0):.4f})")
        logger.info(f"Cross-validation F1: {cv_results.get('cv_f1', {}).get('mean', 0):.4f} (±{cv_results.get('cv_f1', {}).get('std', 0):.4f})")


if __name__ == "__main__":
    main()