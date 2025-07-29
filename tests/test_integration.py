"""
Integration tests for the complete ML trading pipeline.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os

from src.utils import Config
from src.data import DataLoader
from src.features import FeatureEngineer
from src.labels import LabelConstructor
from src.model import ModelTrainer, LightGBMModel
from src.backtest import Backtester
from src.risk_management import RiskManager, RiskLimits


@pytest.fixture
def sample_data():
    """Create realistic sample data for integration testing."""
    np.random.seed(42)
    n = 2000  # More data for realistic testing
    
    # Generate realistic price data with trends and volatility
    returns = np.random.normal(0, 0.02, n)
    # Add some trend
    trend = np.linspace(0, 0.05, n)
    returns += trend
    # Add some volatility clustering
    volatility = 0.02 + 0.01 * np.sin(np.linspace(0, 4*np.pi, n))
    returns *= volatility
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = {
        'timestamp': pd.date_range('2023-01-01', periods=n, freq='1min'),
        'open': prices * (1 + np.random.normal(0, 0.001, n)),
        'high': prices * (1 + abs(np.random.normal(0, 0.002, n))),
        'low': prices * (1 - abs(np.random.normal(0, 0.002, n))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def config():
    """Create test configuration."""
    config_dict = {
        "app": {"seed": 42},
        "data": {
            "symbols": ["TEST"], 
            "intervals": ["1m"],
            "horizons": [1, 5, 10, 20, 50]
        },
        "binance": {"base_url": "https://api.binance.com"},
        "indicators": {
            "sma": [20, 50],
            "ema": [12, 26],
            "macd": [12, 26, 9],
            "rsi": [14],
            "stochastic": [14, 3, 3],
            "bollinger": [20, 2],
            "parabolic_sar": [0.02, 0.2],
            "ichimoku": [9, 26, 52],
            "atr": [14],
            "cci": [20],
            "roc": [14],
            "williams_r": [14],
            "keltner": [20, 10, 2],
            "supertrend": [10, 3],
            "dpo": [20],
            "trix": [15, 9],
            "momentum": [10],
            "adx": [14],
            "mfi": [14],
            "chaikin_money_flow": [20],
            "vwap": [1],
            "chaikin_oscillator": [3, 10],
            "ease_of_movement": [14],
            "force_index": [13],
            "volume_roc": [14],
            "nvi": [255],
            "mfv": [20],
            "close_std": [20]
        },
        "model": {"params": {}},
        "walkforward": {"training_bars": {"1m": 1000}, "test_bars": {"1m": 100}},
        "backtest": {
            "taker_fee_bps": 10, 
            "slippage_bps": 2, 
            "position_threshold": 0.5, 
            "fixed_notional": 10000,
            "max_position_size": 0.1,
            "max_portfolio_risk": 0.02,
            "max_drawdown": 0.15,
            "max_leverage": 2.0,
            "min_margin_ratio": 0.5
        },
        "features": {"lags": [1, 2], "interactions": True, "regime_flags": True, "z_score_windows": [20]},
        "paths": {"data": "data", "logs": "logs", "artefacts": "artefacts"},
        "logging": {"level": "INFO", "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "file": "logs/app.log"},
        "shap": {"sample_size": 1000, "max_display": 20}
    }
    
    return Config(**config_dict)


class TestEndToEndPipeline:
    """Test the complete end-to-end pipeline."""
    
    def test_data_loading_and_processing(self, config, sample_data):
        """Test data loading and processing pipeline."""
        # Save sample data to bronze layer
        bronze_path = f"{config.paths['data']}/bronze"
        os.makedirs(bronze_path, exist_ok=True)
        sample_data.to_parquet(f"{bronze_path}/TEST_1m_bronze.parquet")
        
        # Test data loading
        loader = DataLoader(config)
        loaded_data = loader.load_bronze_data("TEST", "1m")
        
        assert loaded_data is not None
        assert not loaded_data.empty
        assert len(loaded_data) == len(sample_data)
        
        # Test feature engineering
        feature_engineer = FeatureEngineer(config)
        feature_matrix = feature_engineer.build_feature_matrix(loaded_data, fit_pipeline=True)
        
        assert not feature_matrix.empty
        assert len(feature_matrix) > 0
        assert len(feature_matrix.columns) > 10  # Should have many features
        
        # Check that pipeline is fitted
        assert feature_engineer.is_pipeline_fitted
        assert feature_engineer.feature_pipeline.is_scaler_fitted
    
    def test_label_construction(self, config, sample_data):
        """Test label construction pipeline."""
        # Construct labels
        label_constructor = LabelConstructor(config)
        df_with_labels = label_constructor.construct_all_labels(sample_data)
        
        # Check that labels were created
        label_columns = [col for col in df_with_labels.columns if col.startswith('label_')]
        assert len(label_columns) > 0
        
        # Check that labels are binary for classification
        for label_col in label_columns:
            if 'class' in label_col:
                unique_values = df_with_labels[label_col].dropna().unique()
                assert set(unique_values).issubset({0, 1})
    
    def test_model_training_pipeline(self, config, sample_data):
        """Test complete model training pipeline."""
        # Prepare data with labels
        label_constructor = LabelConstructor(config)
        df_with_labels = label_constructor.construct_all_labels(sample_data)
        
        # Train model
        trainer = ModelTrainer(config)
        results = trainer.train_single_model(
            df_with_labels, 
            "label_class_1", 
            "classification",
            perform_cv=True,
            compute_shap=True
        )
        
        # Check results
        assert 'model' in results
        assert 'feature_engineer' in results
        assert 'test_metrics' in results
        assert 'cv_scores' in results
        
        # Check model performance
        test_metrics = results['test_metrics']
        assert 'accuracy' in test_metrics
        assert 'roc_auc' in test_metrics
        assert 'precision' in test_metrics
        assert 'recall' in test_metrics
        assert 'f1' in test_metrics
        
        # Check that accuracy is realistic (not perfect)
        assert test_metrics['accuracy'] < 0.95  # Should not be perfect
        assert test_metrics['accuracy'] > 0.4   # Should be better than random
    
    def test_real_time_prediction(self, config, sample_data):
        """Test real-time prediction pipeline."""
        # Prepare data and train model
        label_constructor = LabelConstructor(config)
        df_with_labels = label_constructor.construct_all_labels(sample_data)
        
        trainer = ModelTrainer(config)
        results = trainer.train_single_model(df_with_labels, "label_class_1", "classification")
        
        # Get trained model and feature engineer
        model = results['model']
        feature_engineer = results['feature_engineer']
        
        # Test real-time prediction
        from src.model import RealTimePredictor
        predictor = RealTimePredictor(config, model, feature_engineer)
        
        # Use last 100 rows for prediction
        latest_data = sample_data.tail(100)
        prediction = predictor.predict(latest_data)
        
        # Check prediction format
        assert 'prediction' in prediction
        assert 'confidence' in prediction
        assert 'probability' in prediction
        
        # Check prediction values
        assert prediction['prediction'] in [0, 1, None]
        assert 0 <= prediction['confidence'] <= 1
        assert 0 <= prediction['probability'] <= 1
    
    def test_backtesting_pipeline(self, config, sample_data):
        """Test backtesting pipeline with realistic constraints."""
        # Create simple strategy signals
        signals = pd.DataFrame({
            'timestamp': sample_data['timestamp'],
            'signal': np.random.choice([-1, 0, 1], len(sample_data)),
            'confidence': np.random.uniform(0.5, 1.0, len(sample_data))
        })
        
        # Run backtest
        backtester = Backtester(config)
        results = backtester.run_backtest(sample_data, signals)
        
        # Check results structure
        assert 'trades' in results
        assert 'portfolio_value' in results
        assert 'returns' in results
        assert 'metrics' in results
        
        # Check that trades were executed
        assert len(results['trades']) > 0
        
        # Check performance metrics
        metrics = results['metrics']
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics
        
        # Check that drawdown is reasonable
        assert metrics['max_drawdown'] < 0.5  # Should not exceed 50%
    
    def test_risk_management_integration(self, config, sample_data):
        """Test risk management integration."""
        # Create risk manager
        risk_limits = RiskLimits()
        risk_manager = RiskManager(config, risk_limits)
        
        # Test position management
        from src.risk_management import Position, PositionSide
        
        position = Position(
            symbol="TEST",
            side=PositionSide.LONG,
            size=100,
            entry_price=100,
            entry_time=pd.Timestamp.now(),
            stop_loss=95,
            take_profit=110
        )
        
        # Add position
        success = risk_manager.add_position(position)
        assert success
        assert len(risk_manager.positions) == 1
        
        # Check portfolio risk
        current_prices = {"TEST": 105}
        risk_metrics = risk_manager.calculate_portfolio_risk(current_prices)
        
        assert 'total_exposure' in risk_metrics
        assert 'leverage' in risk_metrics
        assert 'margin_ratio' in risk_metrics
        
        # Close position
        trade_result = risk_manager.close_position("TEST", 110, pd.Timestamp.now())
        assert trade_result is not None
        assert len(risk_manager.positions) == 0
    
    def test_pipeline_persistence(self, config, sample_data):
        """Test pipeline persistence (save/load)."""
        # Train model
        label_constructor = LabelConstructor(config)
        df_with_labels = label_constructor.construct_all_labels(sample_data)
        
        trainer = ModelTrainer(config)
        results = trainer.train_single_model(df_with_labels, "label_class_1", "classification")
        
        # Save models
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer.save_all_models(temp_dir)
            
            # Check that files were saved
            saved_files = os.listdir(temp_dir)
            assert len(saved_files) > 0
            
            # Test loading (this would require implementing load functionality)
            # For now, just check that files exist
            for file in saved_files:
                assert os.path.exists(os.path.join(temp_dir, file))
    
    def test_data_leakage_prevention(self, config, sample_data):
        """Test that data leakage is properly prevented."""
        # Split data chronologically
        split_idx = int(len(sample_data) * 0.8)
        train_data = sample_data.iloc[:split_idx]
        test_data = sample_data.iloc[split_idx:]
        
        # Add labels
        label_constructor = LabelConstructor(config)
        train_with_labels = label_constructor.construct_all_labels(train_data)
        test_with_labels = label_constructor.construct_all_labels(test_data)
        
        # Train model
        trainer = ModelTrainer(config)
        results = trainer.train_single_model(train_with_labels, "label_class_1", "classification")
        
        # Check that test accuracy is realistic
        test_metrics = results['test_metrics']
        assert test_metrics['accuracy'] < 0.95  # Should not be perfect
        assert test_metrics['accuracy'] > 0.4   # Should be better than random
        
        # Check cross-validation scores
        cv_scores = results['cv_scores']
        if cv_scores is not None:
            for metric, scores in cv_scores.items():
                assert len(scores) > 0
                assert scores.mean() < 0.95  # Should not be perfect
    
    def test_error_handling(self, config):
        """Test error handling throughout the pipeline."""
        # Test with empty data
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            feature_engineer = FeatureEngineer(config)
            feature_engineer.build_feature_matrix(empty_df, fit_pipeline=True)
        
        # Test with missing columns
        invalid_df = pd.DataFrame({'some_column': [1, 2, 3]})
        
        with pytest.raises(KeyError):
            feature_engineer = FeatureEngineer(config)
            feature_engineer.build_feature_matrix(invalid_df, fit_pipeline=True)
    
    def test_performance_benchmarks(self, config, sample_data):
        """Test performance benchmarks."""
        import time
        
        # Test feature engineering performance
        start_time = time.time()
        feature_engineer = FeatureEngineer(config)
        feature_matrix = feature_engineer.build_feature_matrix(sample_data, fit_pipeline=True)
        feature_time = time.time() - start_time
        
        # Should be reasonably fast
        assert feature_time < 30.0  # Less than 30 seconds for 2000 rows
        
        # Test model training performance
        label_constructor = LabelConstructor(config)
        df_with_labels = label_constructor.construct_all_labels(sample_data)
        
        start_time = time.time()
        trainer = ModelTrainer(config)
        results = trainer.train_single_model(df_with_labels, "label_class_1", "classification")
        training_time = time.time() - start_time
        
        # Should be reasonably fast
        assert training_time < 60.0  # Less than 60 seconds for training


class TestScalability:
    """Test scalability of the pipeline."""
    
    def test_large_dataset_handling(self, config):
        """Test handling of large datasets."""
        # Create larger dataset
        n = 10000
        large_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=n, freq='1min'),
            'open': np.random.randn(n).cumsum() + 100,
            'high': np.random.randn(n).cumsum() + 102,
            'low': np.random.randn(n).cumsum() + 98,
            'close': np.random.randn(n).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, n)
        })
        
        # Test feature engineering
        feature_engineer = FeatureEngineer(config)
        feature_matrix = feature_engineer.build_feature_matrix(large_data, fit_pipeline=True)
        
        assert not feature_matrix.empty
        assert len(feature_matrix) > 0
        
        # Test model training
        label_constructor = LabelConstructor(config)
        df_with_labels = label_constructor.construct_all_labels(large_data)
        
        trainer = ModelTrainer(config)
        results = trainer.train_single_model(df_with_labels, "label_class_1", "classification")
        
        assert results is not None
        assert 'test_metrics' in results


if __name__ == "__main__":
    pytest.main([__file__])