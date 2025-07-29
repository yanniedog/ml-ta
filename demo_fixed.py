#!/usr/bin/env python3
"""
Comprehensive demo script for the technical analysis system.
Demonstrates data loading, feature engineering, model training, and real-time predictions.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import load_config
from src.data import BinanceDataFetcher
from src.indicators import TechnicalIndicators
from src.features import FeatureEngineer
from src.labels import LabelConstructor
from src.model import LightGBMModel, ModelTrainer, RealTimePredictor
from src.backtest import Backtester
from src.report import ReportGenerator


def setup_logging():
    """Setup logging for demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/demo.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def demo_data_loading():
    """Demonstrate data loading functionality."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("DEMO: Data Loading")
    logger.info("=" * 60)
    
    try:
        config = load_config("config/settings.yaml")
        data_fetcher = BinanceDataFetcher(config)
        
        # Load existing data
        bronze_path = f"{config.paths['data']}/bronze/SOLUSDT_1m_bronze.parquet"
        
        if Path(bronze_path).exists():
            df = pd.read_parquet(bronze_path)
            logger.info(f"Loaded data: {df.shape}")
            logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            logger.info(f"Columns: {list(df.columns)}")
            return df
        else:
            logger.warning("No existing data found, creating sample data")
            # Create sample data for demo
            dates = pd.date_range('2023-01-01', periods=5000, freq='1min')
            df = pd.DataFrame({
                'timestamp': dates,
                'open': np.random.randn(5000).cumsum() + 100,
                'high': np.random.randn(5000).cumsum() + 102,
                'low': np.random.randn(5000).cumsum() + 98,
                'close': np.random.randn(5000).cumsum() + 100,
                'volume': np.random.randint(1000, 10000, 5000)
            })
            return df
            
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return None


def demo_technical_indicators(df):
    """Demonstrate technical indicators calculation."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("DEMO: Technical Indicators")
    logger.info("=" * 60)
    
    try:
        config = load_config("config/settings.yaml")
        indicators = TechnicalIndicators(config)
        
        # Calculate all indicators
        df_with_indicators = indicators.calculate_all_indicators(df)
        
        logger.info(f"Original data shape: {df.shape}")
        logger.info(f"Data with indicators shape: {df_with_indicators.shape}")
        logger.info(f"New indicator columns: {[col for col in df_with_indicators.columns if col not in df.columns]}")
        
        # Show some sample indicators
        sample_indicators = ['rsi', 'macd_macd', 'bb_upper', 'stoch_k_percent', 'adx_adx']
        for indicator in sample_indicators:
            if indicator in df_with_indicators.columns:
                value = df_with_indicators[indicator].iloc[-1]
                logger.info(f"{indicator}: {value:.4f}")
        
        return df_with_indicators
        
    except Exception as e:
        logger.error(f"Technical indicators failed: {e}")
        return None


def demo_feature_engineering(df):
    """Demonstrate feature engineering functionality."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("DEMO: Feature Engineering")
    logger.info("=" * 60)
    
    try:
        config = load_config("config/settings.yaml")
        feature_engineer = FeatureEngineer(config)
        
        # Build feature matrix
        feature_matrix = feature_engineer.build_feature_matrix(df, fit_scaler=True)
        
        logger.info(f"Original data shape: {df.shape}")
        logger.info(f"Feature matrix shape: {feature_matrix.shape}")
        
        # Show feature categories
        feature_categories = {
            'Price Features': [col for col in feature_matrix.columns if any(x in col for x in ['close', 'high', 'low', 'open'])],
            'Volume Features': [col for col in feature_matrix.columns if 'volume' in col],
            'Technical Indicators': [col for col in feature_matrix.columns if any(x in col for x in ['rsi', 'macd', 'bb_', 'stoch'])],
            'Regime Flags': [col for col in feature_matrix.columns if any(x in col for x in ['_trend', '_vol', '_squeeze', '_expansion'])],
            'Lagged Features': [col for col in feature_matrix.columns if 'lag_' in col],
            'Z-Score Features': [col for col in feature_matrix.columns if 'zscore' in col],
            'Interaction Features': [col for col in feature_matrix.columns if any(x in col for x in ['price_volume', 'rsi_macd', 'bb_position'])]
        }
        
        for category, features in feature_categories.items():
            if features:
                logger.info(f"{category}: {len(features)} features")
        
        # Show scaler info
        scaler_info = feature_engineer.get_scaler_info()
        logger.info(f"Scaler fitted: {scaler_info['is_fitted']}")
        logger.info(f"Features scaled: {scaler_info['feature_columns_count']}")
        
        return feature_matrix, feature_engineer
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return None, None


def demo_label_construction(df):
    """Demonstrate label construction functionality."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("DEMO: Label Construction")
    logger.info("=" * 60)
    
    try:
        config = load_config("config/settings.yaml")
        label_constructor = LabelConstructor(config)
        
        # Create various labels
        df_with_labels = label_constructor.construct_all_labels(df)
        
        logger.info(f"Original data shape: {df.shape}")
        logger.info(f"Data with labels shape: {df_with_labels.shape}")
        
        # Show label columns
        label_columns = [col for col in df_with_labels.columns if col.startswith('label_')]
        logger.info(f"Label columns: {label_columns}")
        
        # Show label distribution for binary labels
        for label_col in label_columns:
            if 'binary' in label_col or 'class' in label_col:
                distribution = df_with_labels[label_col].value_counts()
                logger.info(f"{label_col} distribution: {distribution.to_dict()}")
        
        return df_with_labels
        
    except Exception as e:
        logger.error(f"Label construction failed: {e}")
        return None


def demo_model_training(df_with_features, df_with_labels):
    """Demonstrate model training functionality."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("DEMO: Model Training")
    logger.info("=" * 60)
    
    try:
        config = load_config("config/settings.yaml")
        trainer = ModelTrainer(config)
        
        # Ensure both DataFrames have the same index
        common_index = df_with_features.index.intersection(df_with_labels.index)
        df_features_aligned = df_with_features.loc[common_index]
        df_labels_aligned = df_with_labels.loc[common_index]
        
        # Combine features and labels
        df_combined = pd.concat([df_features_aligned, df_labels_aligned[['label_class_1']]], axis=1)
        
        # Remove any rows with NaN values
        df_combined = df_combined.dropna()
        
        logger.info(f"Aligned data shape: {df_combined.shape}")
        logger.info(f"Label distribution: {df_combined['label_class_1'].value_counts().to_dict()}")
        
        # Train a single model
        results = trainer.train_single_model(
            df_combined, 
            'label_class_1', 
            task_type="classification",
            perform_cv=True,
            compute_shap=False  # Skip SHAP for faster demo
        )
        
        if results and 'model' in results:
            logger.info("Model training successful!")
            logger.info(f"Test accuracy: {results.get('test_accuracy', 'N/A')}")
            logger.info(f"Test ROC AUC: {results.get('test_roc_auc', 'N/A')}")
            
            # Show cross-validation results
            cv_results = results.get('cv_results', {})
            if cv_results:
                logger.info("Cross-validation results:")
                for metric, result in cv_results.items():
                    if isinstance(result, dict) and 'mean' in result:
                        logger.info(f"  {metric}: {result['mean']:.4f} (+/- {result['std']:.4f})")
            
            return results
        else:
            logger.error("Model training failed")
            return None
            
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return None


def demo_real_time_prediction(df_with_features, feature_engineer, trained_results):
    """Demonstrate real-time prediction functionality."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("DEMO: Real-Time Prediction")
    logger.info("=" * 60)
    
    try:
        config = load_config("config/settings.yaml")
        
        # Create real-time predictor
        predictor = RealTimePredictor(config, trained_results['model'], feature_engineer)
        
        # Create live data for prediction
        live_dates = pd.date_range('2023-02-01', periods=500, freq='1min')
        live_data = pd.DataFrame({
            'timestamp': live_dates,
            'open': np.random.randn(500).cumsum() + 100,
            'high': np.random.randn(500).cumsum() + 102,
            'low': np.random.randn(500).cumsum() + 98,
            'close': np.random.randn(500).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 500)
        })
        
        # Make multiple predictions
        predictions = []
        for i in range(5):
            # Use a subset of live data for each prediction
            subset_data = live_data.iloc[:300+i*50]
            prediction = predictor.predict(subset_data)
            predictions.append(prediction)
            logger.info(f"Prediction {i+1}: {prediction}")
        
        # Show prediction summary
        summary = predictor.get_prediction_summary()
        if summary:
            logger.info(f"Prediction summary: {summary}")
        
        return predictions
        
    except Exception as e:
        logger.error(f"Real-time prediction failed: {e}")
        return None


def demo_backtesting(df_with_features, df_with_labels, trained_results):
    """Demonstrate backtesting functionality."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("DEMO: Backtesting")
    logger.info("=" * 60)
    
    try:
        config = load_config("config/settings.yaml")
        backtester = Backtester(config)
        
        # Ensure both DataFrames have the same index
        common_index = df_with_features.index.intersection(df_with_labels.index)
        df_features_aligned = df_with_features.loc[common_index]
        df_labels_aligned = df_with_labels.loc[common_index]
        
        # Combine features and labels
        df_combined = pd.concat([df_features_aligned, df_labels_aligned[['label_class_1']]], axis=1)
        
        # Remove any rows with NaN values
        df_combined = df_combined.dropna()
        
        # Run backtest
        # First, we need to generate predictions for the backtest
        model = trained_results['model']
        X = df_combined.drop(columns=['label_class_1'])
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]  # Probability of class 1
        
        backtest_results = backtester.run_backtest(
            df_combined,
            pd.Series(predictions, index=df_combined.index),
            pd.Series(probabilities, index=df_combined.index)
        )
        
        if backtest_results:
            logger.info("Backtest completed successfully!")
            logger.info(f"Total return: {backtest_results.get('total_return', 'N/A')}")
            logger.info(f"Sharpe ratio: {backtest_results.get('sharpe_ratio', 'N/A')}")
            logger.info(f"Max drawdown: {backtest_results.get('max_drawdown', 'N/A')}")
            logger.info(f"Win rate: {backtest_results.get('win_rate', 'N/A')}")
        
        return backtest_results
        
    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        return None


def main():
    """Run the comprehensive demo."""
    logger = setup_logging()
    logger.info("Starting comprehensive technical analysis demo...")
    
    # Step 1: Data Loading
    df = demo_data_loading()
    if df is None:
        logger.error("Demo failed at data loading step")
        return
    
    # Step 2: Technical Indicators
    df_with_indicators = demo_technical_indicators(df)
    if df_with_indicators is None:
        logger.error("Demo failed at technical indicators step")
        return
    
    # Step 3: Feature Engineering
    feature_matrix, feature_engineer = demo_feature_engineering(df_with_indicators)
    if feature_matrix is None:
        logger.error("Demo failed at feature engineering step")
        return
    
    # Step 4: Label Construction
    df_with_labels = demo_label_construction(df_with_indicators)
    if df_with_labels is None:
        logger.error("Demo failed at label construction step")
        return
    
    # Step 5: Model Training
    training_results = demo_model_training(feature_matrix, df_with_labels)
    if training_results is None:
        logger.error("Demo failed at model training step")
        return
    
    # Step 6: Real-Time Prediction
    predictions = demo_real_time_prediction(feature_matrix, feature_engineer, training_results)
    if predictions is None:
        logger.error("Demo failed at real-time prediction step")
        return
    
    # Step 7: Backtesting
    backtest_results = demo_backtesting(feature_matrix, df_with_labels, training_results)
    if backtest_results is None:
        logger.error("Demo failed at backtesting step")
        return
    
    logger.info("=" * 60)
    logger.info("DEMO COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info("All components are working correctly:")
    logger.info("✓ Data loading")
    logger.info("✓ Technical indicators calculation")
    logger.info("✓ Feature engineering")
    logger.info("✓ Label construction")
    logger.info("✓ Model training")
    logger.info("✓ Real-time prediction")
    logger.info("✓ Backtesting")


if __name__ == "__main__":
    main()