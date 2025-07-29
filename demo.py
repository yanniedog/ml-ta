#!/usr/bin/env python3
"""
Enhanced demo script for the technical analysis system with comprehensive evaluation.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import load_config, setup_logging, set_deterministic_seed, save_parquet
from src.indicators import TechnicalIndicators
from src.features import FeatureEngineer
from src.labels import LabelConstructor
from src.model import ModelTrainer, RealTimePredictor
from src.backtest import Backtester


def generate_sample_data(n_samples=10000):
    """Generate sample OHLCV data for demonstration."""
    np.random.seed(42)
    
    # Generate realistic price data with trends and volatility
    returns = np.random.normal(0, 0.02, n_samples)
    # Add some trend
    trend = np.linspace(0, 0.1, n_samples)
    returns += trend
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data with realistic relationships
    data = {
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1min'),
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + abs(np.random.normal(0, 0.002, n_samples))),
        'low': prices * (1 - abs(np.random.normal(0, 0.002, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    }
    
    return pd.DataFrame(data)


def run_enhanced_demo():
    """Run a comprehensive demo of the enhanced system."""
    print("Starting Enhanced Technical Analysis Demo...")
    
    # Load configuration
    config = load_config("config/settings.yaml")
    
    # Setup logging
    logger = setup_logging(config)
    
    # Set deterministic seed
    set_deterministic_seed(config.app["seed"])
    
    # Generate sample data
    print("Generating sample data...")
    df = generate_sample_data(5000)
    
    # Save as bronze data
    bronze_dir = Path(config.paths["data"]) / "bronze"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    bronze_file = bronze_dir / "SOLUSDT_1m_bronze.parquet"
    save_parquet(df, str(bronze_file))
    print(f"Saved bronze data: {bronze_file}")
    
    # Calculate technical indicators
    print("Calculating technical indicators...")
    indicators = TechnicalIndicators(config)
    df_with_indicators = indicators.calculate_all_indicators(df)
    print(f"Calculated {len(df_with_indicators.columns) - len(df.columns)} indicators")
    
    # Engineer features
    print("Engineering features...")
    engineer = FeatureEngineer(config)
    feature_df = engineer.build_feature_matrix(df_with_indicators)
    print(f"Feature matrix shape: {feature_df.shape}")
    
    # Construct labels
    print("Constructing labels...")
    constructor = LabelConstructor(config)
    labeled_df = constructor.construct_all_labels(feature_df)
    print(f"Labeled data shape: {labeled_df.shape}")
    
    # Save as gold data
    gold_dir = Path(config.paths["data"]) / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    gold_file = gold_dir / "SOLUSDT_1m_gold.parquet"
    save_parquet(labeled_df, str(gold_file))
    print(f"Saved gold data: {gold_file}")
    
    # Train models with enhanced evaluation
    print("Training models with comprehensive evaluation...")
    trainer = ModelTrainer(config)
    label_columns = [col for col in labeled_df.columns if col.startswith('label_')]
    
    model_results = {}
    
    for i, label_col in enumerate(label_columns[:3]):  # Train first 3 models
        print(f"\nTraining model {i+1}/3 for {label_col}...")
        try:
            results = trainer.train_single_model(
                labeled_df, 
                label_col, 
                "auto",  # Auto-detect task type
                perform_cv=True,
                compute_shap=True
            )
            
            if results:  # Only process if training was successful
                model_results[label_col] = results
                
                # Print detailed results
                test_metrics = results['test_metrics']
                task_type = results['task_type']
                
                if task_type == "classification":
                    print(f"  Test Accuracy: {test_metrics.get('accuracy', 0):.3f}")
                    print(f"  ROC AUC: {test_metrics.get('roc_auc', 0):.3f}")
                    print(f"  Precision (Class 1): {test_metrics.get('precision_per_class', [0, 0])[1]:.3f}")
                    print(f"  Recall (Class 1): {test_metrics.get('recall_per_class', [0, 0])[1]:.3f}")
                else:  # regression
                    print(f"  R² Score: {test_metrics.get('r2', 0):.3f}")
                    print(f"  RMSE: {test_metrics.get('rmse', 0):.3f}")
                    print(f"  MAE: {test_metrics.get('mae', 0):.3f}")
                
                # Print top features
                feature_importance = results['feature_importance']
                print(f"  Top 5 Features: {feature_importance.head(5)['feature'].tolist()}")
            else:
                print(f"  Training failed for {label_col}")
            
        except Exception as e:
            print(f"  Error training model for {label_col}: {e}")
    
    # Run backtest with best model
    if model_results:
        best_label = max(model_results.keys(), 
                        key=lambda x: model_results[x]['test_metrics'].get('accuracy', 0))
        best_model = model_results[best_label]['model']
        
        print(f"\nRunning backtest with best model ({best_label})...")
        backtester = Backtester(config)
        backtest_results = backtester.run_backtest_with_model(labeled_df, best_model, best_label)
        
        performance = backtest_results['performance']
        print(f"Backtest Results:")
        print(f"  Total Return: {performance.get('total_return', 0):.2%}")
        print(f"  Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {performance.get('max_drawdown', 0):.2%}")
        print(f"  Hit Rate: {performance.get('hit_rate', 0):.2%}")
        print(f"  Total Trades: {performance.get('total_trades', 0)}")
    
    # Demonstrate real-time prediction
    print("\nDemonstrating real-time prediction...")
    if model_results:
        # Use the best model for real-time prediction
        best_model = model_results[best_label]['model']
        fitted_scaler = model_results[best_label]['model'].scaler if hasattr(model_results[best_label]['model'], 'scaler') else None
        predictor = RealTimePredictor(config, best_model, fitted_scaler)
        
        # Simulate real-time data (last 100 rows)
        live_data = df.tail(100)
        
        # Make predictions on recent data
        for i in range(min(10, len(live_data))):
            recent_data = live_data.iloc[:i+1]
            prediction = predictor.predict(recent_data)
            if prediction:
                print(f"  {prediction['timestamp']}: Prediction={prediction['prediction']}, "
                      f"Confidence={prediction['confidence']:.3f}")
        
        # Get prediction summary
        summary = predictor.get_prediction_summary()
        print(f"  Prediction Summary: {summary}")
    
    print("\nEnhanced demo completed successfully!")


def run_demo():
    """Run the original demo for backward compatibility."""
    run_enhanced_demo()


if __name__ == "__main__":
    run_enhanced_demo()