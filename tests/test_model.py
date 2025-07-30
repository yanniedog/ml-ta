import pytest
import pandas as pd
import numpy as np
import os
import tempfile

from src.utils import Config
from src.model import ModelTrainer, RealTimePredictor, RealTimeEnsemblePredictor
from src.features import FeatureEngineer

@pytest.fixture
def sample_data():
    """Create realistic sample data for model testing."""
    np.random.seed(42)
    n = 500
    prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
    data = {
        'timestamp': pd.to_datetime(pd.date_range('2023-01-01', periods=n, freq='1min')),
        'open': prices + np.random.normal(0, 0.1, n),
        'high': prices + np.random.normal(0, 0.1, n) + 0.1,
        'low': prices + np.random.normal(0, 0.1, n) - 0.1,
        'close': prices,
        'volume': np.random.lognormal(10, 1, n)
    }
    df = pd.DataFrame(data).set_index('timestamp')
    # Add dummy label for testing
    df['label_class_5'] = (df['close'].shift(-5) > df['close']).astype(int)
    df['label_reg_5'] = df['close'].shift(-5) / df['close'] - 1
    return df.dropna()

@pytest.fixture
def config():
    """Create test configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dict = {
            "app": {"seed": 42},
            "data": {"horizons": [5, 10]},
            "indicators": {"sma": [{'window': 10}, {'window': 20}], "rsi": [{'window': 14}]},
            "features": {"lags": [1, 2], "interactions": False, "regime_detection": False, "rolling_window_features": []},
            "model": {
                "name": "test_model",
                "path": os.path.join(tmpdir, "models"),
                "lgbm_params": {"objective": "binary", "metric": "auc", "n_estimators": 10},
                "tune_hyperparameters": False,
                "cv_folds": 3,
                "cv_strategy": "expanding",
                "cv_gap_size": 1,
                "cv_test_size": 0.2
            },
            "shap": {"enabled": False},
            "paths": {"artefacts": os.path.join(tmpdir, "artefacts")}
        }
        os.makedirs(config_dict['model']['path'], exist_ok=True)
        os.makedirs(config_dict['paths']['artefacts'], exist_ok=True)
        yield Config.from_dict(config_dict)

class TestModelTrainer:
    """Tests for the ModelTrainer class."""

    def test_train_classification_model(self, config, sample_data):
        """Test training a classification model."""
        trainer = ModelTrainer(config)
        label_column = 'label_class_5'
        results = trainer.train_single_model(sample_data, label_column, task_type='classification')

        assert 'model' in results
        assert 'test_metrics' in results
        assert 'feature_importance' in results
        assert results['test_metrics']['accuracy'] >= 0.0
        assert results['test_metrics']['roc_auc'] >= 0.0

    def test_train_regression_model(self, config, sample_data):
        """Test training a regression model."""
        trainer = ModelTrainer(config)
        label_column = 'label_reg_5'
        config.model['lgbm_params']['objective'] = 'regression'
        results = trainer.train_single_model(sample_data, label_column, task_type='regression')

        assert 'model' in results
        assert 'test_metrics' in results
        assert 'r2' in results['test_metrics']
        assert 'rmse' in results['test_metrics']

    def test_train_ensemble_model(self, config, sample_data):
        """Test training an ensemble model."""
        trainer = ModelTrainer(config)
        label_column = 'label_class_5'
        results = trainer.train_ensemble_model(sample_data, label_column, task_type='classification')

        assert 'model' in results
        assert 'test_metrics' in results
        assert 'accuracy' in results['test_metrics']
        assert results['test_metrics']['accuracy'] >= 0.0

    def test_train_ensemble_model_with_hpo(self, config, sample_data):
        """Test training an ensemble model with hyperparameter optimization."""
        trainer = ModelTrainer(config)
        label_column = 'label_class_5'
        results = trainer.train_ensemble_model(sample_data, label_column, task_type='classification', optimize_hyperparams=True)

        assert 'model' in results
        assert 'test_metrics' in results
        assert 'accuracy' in results['test_metrics']
        assert results['test_metrics']['accuracy'] >= 0.0


class TestRealTimePredictor:
    """Tests for the RealTimePredictor class."""

    def test_real_time_prediction(self, config, sample_data):
        """Test real-time prediction and history tracking."""
        trainer = ModelTrainer(config)
        label_column = 'label_class_5'
        train_results = trainer.train_single_model(sample_data, label_column, task_type='classification')

        predictor = RealTimePredictor(config, train_results['model'], trainer.feature_engineers[label_column])

        prediction_data = sample_data.tail(10)
        y_true = prediction_data[label_column]
        result = predictor.predict(prediction_data, y_true)

        assert 'prediction' in result
        assert 'confidence' in result
        assert len(predictor.prediction_history) == 1

        summary = predictor.get_prediction_summary()
        assert summary['total_predictions'] == 1
        assert summary['accuracy'] >= 0.0


class TestRealTimeEnsemblePredictor:
    """Tests for the RealTimeEnsemblePredictor class."""

    def test_ensemble_prediction(self, config, sample_data):
        """Test real-time ensemble prediction."""
        trainer = ModelTrainer(config)
        trainer.train_single_model(sample_data, 'label_class_5', task_type='classification')
        
        ensemble_predictor = RealTimeEnsemblePredictor(config, trainer.results)
        assert len(ensemble_predictor.predictors) > 0

        prediction_data = sample_data.tail(10)
        result = ensemble_predictor.predict(prediction_data)

        assert 'prediction' in result
        assert 'confidence' in result
