#!/usr/bin/env python3
"""
Small demo script to test the system with a smaller dataset.
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
            logging.FileHandler('logs/demo_small.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main():
    """Run small demo."""
    logger = setup_logging()
    logger.info("Starting small technical analysis demo...")
    logger.info("=" * 60)
    
    try:
        # Create smaller sample data
        dates = pd.date_range('2023-01-01', periods=1000, freq='1min')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(1000).cumsum() + 100,
            'high': np.random.randn(1000).cumsum() + 102,
            'low': np.random.randn(1000).cumsum() + 98,
            'close': np.random.randn(1000).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 1000)
        })
        
        logger.info(f"Created sample data: {df.shape}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Step 1: Technical Indicators
        logger.info("=" * 60)
        logger.info("DEMO: Technical Indicators")
        logger.info("=" * 60)
        
        config = load_config("config/settings.yaml")
        indicators = TechnicalIndicators(config)
        df_with_indicators = indicators.calculate_all_indicators(df)
        
        logger.info(f"Original data shape: {df.shape}")
        logger.info(f"Data with indicators shape: {df_with_indicators.shape}")
        
        # Step 2: Feature Engineering
        logger.info("=" * 60)
        logger.info("DEMO: Feature Engineering")
        logger.info("=" * 60)
        
        feature_engineer = FeatureEngineer(config)
        feature_matrix = feature_engineer.build_feature_matrix(df_with_indicators, fit_scaler=True)
        
        logger.info(f"Feature matrix shape: {feature_matrix.shape}")
        
        # Step 3: Label Construction
        logger.info("=" * 60)
        logger.info("DEMO: Label Construction")
        logger.info("=" * 60)
        
        label_constructor = LabelConstructor(config)
        df_with_labels = label_constructor.construct_all_labels(df_with_indicators)
        
        logger.info(f"Data with labels shape: {df_with_labels.shape}")
        
        # Step 4: Model Training
        logger.info("=" * 60)
        logger.info("DEMO: Model Training")
        logger.info("=" * 60)
        
        # Align data
        common_index = feature_matrix.index.intersection(df_with_labels.index)
        feature_matrix_aligned = feature_matrix.loc[common_index]
        df_with_labels_aligned = df_with_labels.loc[common_index]
        
        logger.info(f"Aligned feature matrix shape: {feature_matrix_aligned.shape}")
        logger.info(f"Aligned labels shape: {df_with_labels_aligned.shape}")
        
        # Combine data
        combined_df = pd.concat([feature_matrix_aligned, df_with_labels_aligned[['label_class_1']]], axis=1)
        
        # Remove NaN values
        combined_df = combined_df.dropna()
        logger.info(f"Final combined dataset shape: {combined_df.shape}")
        
        # Train model
        trainer = ModelTrainer(config)
        results = trainer.train_single_model(
            combined_df, 
            "label_class_1", 
            "classification",
            perform_cv=True,
            compute_shap=True
        )
        
        logger.info("✅ Model training completed successfully!")
        
        # Step 5: Real-time Prediction
        logger.info("=" * 60)
        logger.info("DEMO: Real-time Prediction")
        logger.info("=" * 60)
        
        # Create predictor
        trained_model = results['model']
        predictor = RealTimePredictor(config, trained_model, feature_engineer)
        
        # Test predictions
        for i in range(5):
            test_data = df_with_indicators.iloc[i*100:(i+1)*100]
            prediction = predictor.predict(test_data)
            logger.info(f"Prediction {i+1}: {prediction}")
        
        logger.info("✅ Real-time prediction completed successfully!")
        
        # Step 6: Backtesting
        logger.info("=" * 60)
        logger.info("DEMO: Backtesting")
        logger.info("=" * 60)
        
        backtester = Backtester(config)
        backtest_results = backtester.run_backtest_with_model(
            df_with_labels, 
            trained_model, 
            "label_class_1",
            feature_engineer
        )
        
        logger.info(f"Backtest results: {backtest_results}")
        logger.info("✅ Backtesting completed successfully!")
        
        logger.info("=" * 60)
        logger.info("SMALL DEMO COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 