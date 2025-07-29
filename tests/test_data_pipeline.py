"""
Comprehensive tests for data pipeline to prevent data leakage.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.features import FeatureEngineer, FeaturePipeline
from src.utils import Config


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 1000
    
    # Generate realistic price data
    returns = np.random.normal(0, 0.02, n)
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


class TestDataLeakagePrevention:
    """Test data leakage prevention mechanisms."""
    
    def test_feature_pipeline_fit_transform_separation(self, config, sample_data):
        """Test that fit and transform are properly separated."""
        pipeline = FeaturePipeline(config)
        
        # Split data chronologically
        split_idx = int(len(sample_data) * 0.8)
        train_data = sample_data.iloc[:split_idx]
        test_data = sample_data.iloc[split_idx:]
        
        # Fit on training data only
        pipeline.fit(train_data)
        
        # Transform test data (should not refit)
        transformed_test = pipeline.transform(test_data)
        
        # Check that scaler was fitted on training data only
        assert pipeline.is_scaler_fitted
        assert pipeline.fitted_on_data is not None
        assert len(pipeline.fitted_on_data) == len(train_data)
        
        # Check that test data was transformed without refitting
        assert len(transformed_test) == len(test_data)
    
    def test_feature_columns_consistency(self, config, sample_data):
        """Test that feature columns are consistent between fit and transform."""
        pipeline = FeaturePipeline(config)
        
        # Split data
        split_idx = int(len(sample_data) * 0.8)
        train_data = sample_data.iloc[:split_idx]
        test_data = sample_data.iloc[split_idx:]
        
        # Fit pipeline
        pipeline.fit(train_data)
        original_feature_columns = pipeline.feature_columns.copy()
        
        # Transform test data
        transformed_test = pipeline.transform(test_data)
        
        # Check feature columns consistency
        assert pipeline.feature_columns == original_feature_columns
        assert set(transformed_test.columns) >= set(original_feature_columns)
    
    def test_no_future_information_leakage(self, config, sample_data):
        """Test that no future information leaks into features."""
        # Add some future-looking columns
        sample_data['future_price'] = sample_data['close'].shift(-1)
        sample_data['future_return'] = sample_data['close'].pct_change().shift(-1)
        
        pipeline = FeaturePipeline(config)
        
        # Split data chronologically
        split_idx = int(len(sample_data) * 0.8)
        train_data = sample_data.iloc[:split_idx]
        
        # Fit pipeline
        pipeline.fit(train_data)
        
        # Check that future columns are not included in features
        future_columns = ['future_price', 'future_return']
        for col in future_columns:
            assert col not in pipeline.feature_columns
    
    def test_timestamp_exclusion(self, config, sample_data):
        """Test that timestamp is excluded from features."""
        pipeline = FeaturePipeline(config)
        
        # Fit pipeline
        pipeline.fit(sample_data)
        
        # Check that timestamp is not in feature columns
        assert 'timestamp' not in pipeline.feature_columns
    
    def test_label_exclusion(self, config, sample_data):
        """Test that label columns are excluded from features."""
        # Add label columns
        sample_data['label_class_1'] = (sample_data['close'] > sample_data['close'].shift(1)).astype(int)
        sample_data['label_reg_1'] = sample_data['close'].pct_change()
        
        pipeline = FeaturePipeline(config)
        
        # Fit pipeline
        pipeline.fit(sample_data)
        
        # Check that label columns are not in feature columns
        label_columns = [col for col in sample_data.columns if col.startswith('label_')]
        for col in label_columns:
            assert col not in pipeline.feature_columns


class TestFeatureScalingConsistency:
    """Test feature scaling consistency."""
    
    def test_scaler_fit_only_on_training(self, config, sample_data):
        """Test that scaler is fitted only on training data."""
        pipeline = FeaturePipeline(config)
        
        # Split data
        split_idx = int(len(sample_data) * 0.8)
        train_data = sample_data.iloc[:split_idx]
        test_data = sample_data.iloc[split_idx:]
        
        # Fit on training data
        pipeline.fit(train_data)
        
        # Get training statistics
        train_stats = train_data[pipeline.feature_columns].describe()
        
        # Transform test data
        transformed_test = pipeline.transform(test_data)
        
        # Check that test data was transformed using training statistics
        test_features = transformed_test[pipeline.feature_columns]
        
        # The transformed data should have different statistics than original test data
        original_test_stats = test_data[pipeline.feature_columns].describe()
        assert not test_features.equals(test_data[pipeline.feature_columns])
    
    def test_scaler_persistence(self, config, sample_data):
        """Test that scaler state persists correctly."""
        pipeline = FeaturePipeline(config)
        
        # Split data
        split_idx = int(len(sample_data) * 0.8)
        train_data = sample_data.iloc[:split_idx]
        test_data = sample_data.iloc[split_idx:]
        
        # Fit pipeline
        pipeline.fit(train_data)
        
        # Save and reload pipeline state
        import tempfile
        import joblib
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pipeline.save_scaler_state(f.name)
            
            # Create new pipeline and load state
            new_pipeline = FeaturePipeline(config)
            new_pipeline.load_scaler_state(f.name)
            
            # Transform test data with both pipelines
            original_transformed = pipeline.transform(test_data)
            loaded_transformed = new_pipeline.transform(test_data)
            
            # Results should be identical
            pd.testing.assert_frame_equal(original_transformed, loaded_transformed)
    
    def test_extreme_value_handling(self, config, sample_data):
        """Test handling of extreme values."""
        # Add extreme values
        sample_data.loc[0, 'close'] = 1e10  # Very large value
        sample_data.loc[1, 'close'] = -1e10  # Very negative value
        sample_data.loc[2, 'close'] = np.inf  # Infinity
        sample_data.loc[3, 'close'] = -np.inf  # Negative infinity
        
        pipeline = FeaturePipeline(config)
        
        # Fit pipeline
        pipeline.fit(sample_data)
        
        # Check that extreme values are handled
        assert pipeline.is_scaler_fitted
        
        # Transform data
        transformed = pipeline.transform(sample_data)
        
        # Check that no infinite values remain
        for col in pipeline.feature_columns:
            assert not np.isinf(transformed[col]).any()


class TestFeatureEngineering:
    """Test feature engineering functionality."""
    
    def test_feature_engineer_initialization(self, config):
        """Test feature engineer initialization."""
        engineer = FeatureEngineer(config)
        
        assert engineer.config == config
        assert engineer.indicators is not None
        assert engineer.feature_pipeline is not None
        assert not engineer.is_pipeline_fitted
    
    def test_build_feature_matrix_training(self, config, sample_data):
        """Test building feature matrix for training."""
        engineer = FeatureEngineer(config)
        
        # Build feature matrix for training
        feature_matrix = engineer.build_feature_matrix(sample_data, fit_pipeline=True)
        
        # Check that pipeline is fitted
        assert engineer.is_pipeline_fitted
        assert engineer.feature_pipeline.is_scaler_fitted
        
        # Check feature matrix
        assert not feature_matrix.empty
        assert len(feature_matrix) > 0
    
    def test_build_feature_matrix_testing(self, config, sample_data):
        """Test building feature matrix for testing."""
        engineer = FeatureEngineer(config)
        
        # Split data
        split_idx = int(len(sample_data) * 0.8)
        train_data = sample_data.iloc[:split_idx]
        test_data = sample_data.iloc[split_idx:]
        
        # Build feature matrix for training
        train_matrix = engineer.build_feature_matrix(train_data, fit_pipeline=True)
        
        # Build feature matrix for testing
        test_matrix = engineer.build_feature_matrix(test_data, fit_pipeline=False)
        
        # Check that both matrices have the same features
        train_features = [col for col in train_matrix.columns if not col.startswith(('label_', 'return_', 'timestamp'))]
        test_features = [col for col in test_matrix.columns if not col.startswith(('label_', 'return_', 'timestamp'))]
        
        assert set(train_features) == set(test_features)
    
    def test_regime_flags_creation(self, config, sample_data):
        """Test regime flags creation."""
        engineer = FeatureEngineer(config)
        
        # Add some technical indicators
        sample_data['rsi'] = 50 + np.random.normal(0, 10, len(sample_data))
        sample_data['macd_macd'] = np.random.normal(0, 1, len(sample_data))
        sample_data['macd_signal'] = np.random.normal(0, 1, len(sample_data))
        
        # Add regime flags
        result = engineer.add_regime_flags(sample_data)
        
        # Check that regime flags were added
        regime_columns = [col for col in result.columns if any(x in col for x in ['regime', 'rsi_', 'macd_'])]
        assert len(regime_columns) > 0
    
    def test_lagged_features_creation(self, config, sample_data):
        """Test lagged features creation."""
        engineer = FeatureEngineer(config)
        
        # Add lagged features
        result = engineer.add_lags(sample_data)
        
        # Check that lagged features were added
        lag_columns = [col for col in result.columns if '_lag_' in col]
        assert len(lag_columns) > 0
        
        # Check specific lags
        expected_lags = config.features["lags"]
        for lag in expected_lags:
            lag_col = f'close_lag_{lag}'
            if lag_col in result.columns:
                # Check that lagged values are correct
                assert result[lag_col].iloc[lag] == sample_data['close'].iloc[0]
    
    def test_feature_interactions_creation(self, config, sample_data):
        """Test feature interactions creation."""
        engineer = FeatureEngineer(config)
        
        # Add some technical indicators
        sample_data['rsi'] = 50 + np.random.normal(0, 10, len(sample_data))
        sample_data['volume'] = np.random.lognormal(10, 1, len(sample_data))
        
        # Add interactions
        result = engineer.add_interactions(sample_data)
        
        # Check that interactions were added
        interaction_columns = [col for col in result.columns if any(x in col for x in ['price_volume', 'rsi_macd'])]
        assert len(interaction_columns) > 0


class TestErrorHandling:
    """Test error handling in data pipeline."""
    
    def test_empty_dataframe_handling(self, config):
        """Test handling of empty DataFrame."""
        engineer = FeatureEngineer(config)
        
        # Create empty DataFrame
        empty_df = pd.DataFrame()
        
        # Should handle gracefully
        with pytest.raises(ValueError):
            engineer.build_feature_matrix(empty_df, fit_pipeline=True)
    
    def test_missing_required_columns(self, config):
        """Test handling of missing required columns."""
        engineer = FeatureEngineer(config)
        
        # Create DataFrame without required columns
        df = pd.DataFrame({'some_column': [1, 2, 3]})
        
        # Should handle gracefully
        with pytest.raises(KeyError):
            engineer.build_feature_matrix(df, fit_pipeline=True)
    
    def test_nan_handling(self, config, sample_data):
        """Test handling of NaN values."""
        # Add NaN values
        sample_data.loc[0, 'close'] = np.nan
        sample_data.loc[1, 'volume'] = np.nan
        
        engineer = FeatureEngineer(config)
        
        # Should handle NaN values gracefully
        feature_matrix = engineer.build_feature_matrix(sample_data, fit_pipeline=True)
        
        # Check that NaN values are handled
        assert not feature_matrix.isna().all().all()
    
    def test_infinity_handling(self, config, sample_data):
        """Test handling of infinity values."""
        # Add infinity values
        sample_data.loc[0, 'close'] = np.inf
        sample_data.loc[1, 'volume'] = -np.inf
        
        engineer = FeatureEngineer(config)
        
        # Should handle infinity values gracefully
        feature_matrix = engineer.build_feature_matrix(sample_data, fit_pipeline=True)
        
        # Check that infinity values are handled
        assert not np.isinf(feature_matrix.select_dtypes(include=[np.number])).any().any()


class TestPerformance:
    """Test performance characteristics."""
    
    def test_memory_usage(self, config, sample_data):
        """Test memory usage optimization."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            
            engineer = FeatureEngineer(config)
            
            # Build feature matrix
            feature_matrix = engineer.build_feature_matrix(sample_data, fit_pipeline=True)
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 100MB)
            assert memory_increase < 100 * 1024 * 1024
        except ImportError:
            # Skip test if psutil is not available
            pytest.skip("psutil not available, skipping memory usage test")
    
    def test_processing_speed(self, config, sample_data):
        """Test processing speed."""
        import time
        
        engineer = FeatureEngineer(config)
        
        # Measure processing time
        start_time = time.time()
        feature_matrix = engineer.build_feature_matrix(sample_data, fit_pipeline=True)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Processing should be reasonably fast (less than 10 seconds for 1000 rows)
        assert processing_time < 10.0
    
    def test_scalability(self, config):
        """Test scalability with larger datasets."""
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
        
        engineer = FeatureEngineer(config)
        
        # Should handle larger datasets
        feature_matrix = engineer.build_feature_matrix(large_data, fit_pipeline=True)
        
        assert len(feature_matrix) > 0
        assert len(feature_matrix.columns) > 5


if __name__ == "__main__":
    pytest.main([__file__])