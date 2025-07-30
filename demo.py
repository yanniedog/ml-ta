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
from src.model import LightGBMModel, AdvancedModelTrainer, RealTimePredictor
from src.backtest import Backtester
from src.report import ReportGenerator


def setup_logging():
    """Setup logging for demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('run_output.log'),
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
        
        # Analyze feature types
        price_features = [col for col in feature_matrix.columns if any(x in col for x in ['open', 'high', 'low', 'close', 'volume'])]
        volume_features = [col for col in feature_matrix.columns if 'volume' in col]
        technical_features = [col for col in feature_matrix.columns if col not in price_features + volume_features]
        regime_features = [col for col in feature_matrix.columns if 'regime' in col]
        lagged_features = [col for col in feature_matrix.columns if 'lag' in col]
        zscore_features = [col for col in feature_matrix.columns if 'zscore' in col]
        interaction_features = [col for col in feature_matrix.columns if 'interaction' in col]
        
        logger.info(f"Price Features: {len(price_features)} features")
        logger.info(f"Volume Features: {len(volume_features)} features")
        logger.info(f"Technical Indicators: {len(technical_features)} features")
        logger.info(f"Regime Flags: {len(regime_features)} features")
        logger.info(f"Lagged Features: {len(lagged_features)} features")
        logger.info(f"Z-Score Features: {len(zscore_features)} features")
        logger.info(f"Interaction Features: {len(interaction_features)} features")
        logger.info(f"Scaler fitted: {hasattr(feature_engineer, 'scaler')}")
        logger.info(f"Features scaled: {len(feature_matrix.columns)}")
        
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


def demo_model_training(df_with_features, df_with_labels, original_df):
    """Demonstrate model training functionality."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("DEMO: Model Training")
    logger.info("=" * 60)
    
    try:
        config = load_config("config/settings.yaml")
        model_trainer = AdvancedModelTrainer(config)
        
        # Ensure both DataFrames have the same index and length
        # Get the common index between features and labels
        common_index = df_with_features.index.intersection(df_with_labels.index)
        
        if len(common_index) == 0:
            logger.error("No common indices between feature matrix and label matrix")
            return None
        
        # Align both DataFrames to the common index
        df_features_aligned = df_with_features.loc[common_index]
        df_labels_aligned = df_with_labels.loc[common_index]
        
        logger.info(f"Feature matrix shape after alignment: {df_features_aligned.shape}")
        logger.info(f"Label matrix shape after alignment: {df_labels_aligned.shape}")
        
        # CRITICAL FIX: Ensure original OHLCV data is preserved for feature engineering
        # Get the original data that was used for feature engineering
        original_data = original_df  # This should be the original OHLCV data
        
        # Combine original data with labels
        df_combined = pd.concat([original_data, df_labels_aligned[['label_class_1']]], axis=1)
        
        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'label_class_1']
        missing_columns = [col for col in required_columns if col not in df_combined.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Check for NaN values in the target variable
        nan_count = df_combined['label_class_1'].isna().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in target variable, dropping them")
            df_combined = df_combined.dropna(subset=['label_class_1'])
        
        logger.info(f"Final combined dataset shape: {df_combined.shape}")
        
        # Train an ensemble model
        X = df_features_aligned
        y = df_labels_aligned['label_class_1']

        results = model_trainer.train_ensemble_model(
            X, 
            y, 
            'label_class_1',
            task_type="classification"
        )
        
        if results and 'models' in results:
            logger.info("Ensemble model training successful!")
            ensemble_scores = results.get('scores', {}).get('ensemble', {})
            if ensemble_scores:
                logger.info(f"Ensemble ROC AUC: {ensemble_scores.get('roc_auc', 'N/A'):.4f}")
                logger.info(f"Ensemble Accuracy: {ensemble_scores.get('accuracy', 'N/A'):.4f}")

            # Save the trained models
            model_path = Path(config.paths['artefacts']) / 'ensemble_model.joblib'
            model_trainer.save_models(model_path)
            logger.info(f"Ensemble model saved to {model_path}")

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
        predictor = RealTimePredictor(config, trained_results['models']['ensemble'], feature_engineer)
        
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


def demo_backtesting(df_with_features, df_with_labels, training_results):
    """Demonstrate backtesting functionality."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("DEMO: Backtesting")
    logger.info("=" * 60)
    
    try:
        config = load_config("config/settings.yaml")
        backtester = Backtester(config)
        
        # Ensure both DataFrames have the same index and length
        common_index = df_with_features.index.intersection(df_with_labels.index)
        
        if len(common_index) == 0:
            logger.error("No common indices between feature matrix and label matrix")
            return None
        
        # Align both DataFrames to the common index
        df_features_aligned = df_with_features.loc[common_index]
        df_labels_aligned = df_with_labels.loc[common_index]
        
        # Combine features and labels
        df_combined = pd.concat([df_features_aligned, df_labels_aligned[['label_class_1']]], axis=1)
        
        # Check for NaN values in the target variable
        nan_count = df_combined['label_class_1'].isna().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in target variable, dropping them")
            df_combined = df_combined.dropna(subset=['label_class_1'])
        
        logger.info(f"Backtest dataset shape: {df_combined.shape}")
        
        # Get the trained model from training results
        if training_results and 'models' in training_results and 'ensemble' in training_results['models']:
            model = training_results['models']['ensemble']
            
            # Run backtest with model
            backtest_results = backtester.run_backtest_with_model(
                df_combined,
                model,
                'label_class_1'
            )
            
            if backtest_results:
                logger.info("Backtest completed successfully!")
                performance = backtest_results.get('performance', {})
                logger.info(f"Total return: {performance.get('total_return', 'N/A')}")
                logger.info(f"Sharpe ratio: {performance.get('sharpe_ratio', 'N/A')}")
                logger.info(f"Max drawdown: {performance.get('max_drawdown', 'N/A')}")
                logger.info(f"Hit rate: {performance.get('hit_rate', 'N/A')}")
                logger.info(f"Total trades: {performance.get('total_trades', 'N/A')}")
            
            return backtest_results
        else:
            logger.error("No trained model available for backtesting")
            return None
        
    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        return None


def main():
    """Run the comprehensive demo."""
    logger = setup_logging()
    print("--- SCRIPT EXECUTION STARTED ---")
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
    logger.info("============================================================")
    logger.info("DEMO: Feature Engineering")
    logger.info("============================================================")

    try:
        # Initialize feature engineer
        config = load_config("config/settings.yaml")
        feature_engineer = FeatureEngineer(config)
        
        # Build feature matrix
        df_features = feature_engineer.build_feature_matrix(df_with_indicators)
        
        logger.info("Original data shape: %s", df_with_indicators.shape)
        if df_features is not None:
            logger.info("Feature matrix shape: %s", df_features.shape)
            feature_names = feature_engineer.get_feature_names()
            if feature_names:
                logger.info("Number of features: %d", len(feature_names))
                logger.info("Sample features: %s", feature_names[:5])
        
        # Analyze feature types
        price_features = [col for col in df_features.columns if any(x in col for x in ['open', 'high', 'low', 'close', 'volume'])]
        volume_features = [col for col in df_features.columns if 'volume' in col]
        technical_features = [col for col in df_features.columns if col not in price_features + volume_features]
        regime_features = [col for col in df_features.columns if 'regime' in col]
        lagged_features = [col for col in df_features.columns if 'lag' in col]
        zscore_features = [col for col in df_features.columns if 'zscore' in col]
        interaction_features = [col for col in df_features.columns if 'interaction' in col]
        
        logger.info(f"Price Features: {len(price_features)} features")
        logger.info(f"Volume Features: {len(volume_features)} features")
        logger.info(f"Technical Indicators: {len(technical_features)} features")
        logger.info(f"Regime Flags: {len(regime_features)} features")
        logger.info(f"Lagged Features: {len(lagged_features)} features")
        logger.info(f"Z-Score Features: {len(zscore_features)} features")
        logger.info(f"Interaction Features: {len(interaction_features)} features")
        logger.info(f"Scaler fitted: {hasattr(feature_engineer, 'scaler')}")
        logger.info(f"Features scaled: {len(feature_matrix.columns)}")
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        logger.error("Demo failed at feature engineering step")
        return
    
    # Step 4: Label Construction
    df_with_labels = demo_label_construction(df_with_indicators)
    if df_with_labels is None:
        logger.error("Demo failed at label construction step")
        return
    
    # Step 5: Model Training
    training_results = demo_model_training(feature_matrix, df_with_labels, df)
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
    logger.info("+ Data loading")
    logger.info("+ Technical indicators calculation")
    logger.info("+ Feature engineering")
    logger.info("+ Label construction")
    logger.info("+ Model training")
    logger.info("+ Real-time prediction")
    logger.info("+ Backtesting")


if __name__ == "__main__":
    main()