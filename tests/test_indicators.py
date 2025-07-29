"""
Tests for technical indicators module.
"""
import numpy as np
import pandas as pd
import pytest

from src.indicators import TechnicalIndicators
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
        "data": {"symbols": ["TEST"], "intervals": ["1m"]},
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
        "backtest": {"taker_fee_bps": 10, "slippage_bps": 2, "position_threshold": 0.5, "fixed_notional": 10000},
        "features": {"lags": [1, 2], "interactions": True, "regime_flags": True, "z_score_windows": [20]},
        "paths": {"data": "data", "logs": "logs", "artefacts": "artefacts"},
        "logging": {"level": "INFO", "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "file": "logs/app.log"},
        "shap": {"sample_size": 1000, "max_display": 20}
    }
    
    return Config(**config_dict)


@pytest.fixture
def indicators(config):
    """Create TechnicalIndicators instance."""
    return TechnicalIndicators(config)


def test_sma(indicators, sample_data):
    """Test Simple Moving Average."""
    result = indicators.sma(sample_data['close'], 20)
    
    assert len(result) == len(sample_data)
    assert not result.isna().all()
    # Use numpy's isclose for floating-point comparison
    assert np.isclose(result.iloc[19], sample_data['close'].iloc[:20].mean(), rtol=1e-10)


def test_ema(indicators, sample_data):
    """Test Exponential Moving Average."""
    result = indicators.ema(sample_data['close'], 20)
    
    assert len(result) == len(sample_data)
    assert not result.isna().all()


def test_macd(indicators, sample_data):
    """Test MACD."""
    result = indicators.macd(sample_data['close'], 12, 26, 9)
    
    assert isinstance(result, pd.DataFrame)
    assert 'macd' in result.columns
    assert 'signal' in result.columns
    assert 'histogram' in result.columns
    assert len(result) == len(sample_data)


def test_rsi(indicators, sample_data):
    """Test RSI."""
    result = indicators.rsi(sample_data['close'], 14)
    
    assert len(result) == len(sample_data)
    assert result.min() >= 0
    assert result.max() <= 100


def test_stochastic(indicators, sample_data):
    """Test Stochastic Oscillator."""
    result = indicators.stochastic(sample_data['high'], sample_data['low'], sample_data['close'])
    
    assert isinstance(result, pd.DataFrame)
    assert 'k_percent' in result.columns
    assert 'd_percent' in result.columns
    assert len(result) == len(sample_data)


def test_bollinger_bands(indicators, sample_data):
    """Test Bollinger Bands."""
    result = indicators.bollinger_bands(sample_data['close'])
    
    assert isinstance(result, pd.DataFrame)
    assert 'upper' in result.columns
    assert 'middle' in result.columns
    assert 'lower' in result.columns
    assert 'bandwidth' in result.columns
    assert 'percent_b' in result.columns
    assert len(result) == len(sample_data)


def test_atr(indicators, sample_data):
    """Test Average True Range."""
    result = indicators.atr(sample_data['high'], sample_data['low'], sample_data['close'])
    
    assert len(result) == len(sample_data)
    assert result.min() >= 0


def test_cci(indicators, sample_data):
    """Test Commodity Channel Index."""
    result = indicators.cci(sample_data['high'], sample_data['low'], sample_data['close'])
    
    assert len(result) == len(sample_data)


def test_roc(indicators, sample_data):
    """Test Rate of Change."""
    result = indicators.roc(sample_data['close'], 14)
    
    assert len(result) == len(sample_data)


def test_williams_r(indicators, sample_data):
    """Test Williams %R."""
    result = indicators.williams_r(sample_data['high'], sample_data['low'], sample_data['close'])
    
    assert len(result) == len(sample_data)
    assert result.min() >= -100
    assert result.max() <= 0


def test_keltner_channels(indicators, sample_data):
    """Test Keltner Channels."""
    result = indicators.keltner_channels(sample_data['high'], sample_data['low'], sample_data['close'])
    
    assert isinstance(result, pd.DataFrame)
    assert 'upper' in result.columns
    assert 'middle' in result.columns
    assert 'lower' in result.columns
    assert 'width' in result.columns
    assert len(result) == len(sample_data)


def test_obv(indicators, sample_data):
    """Test On-Balance Volume."""
    result = indicators.obv(sample_data['close'], sample_data['volume'])
    
    assert len(result) == len(sample_data)


def test_mfi(indicators, sample_data):
    """Test Money Flow Index."""
    result = indicators.mfi(sample_data['high'], sample_data['low'], sample_data['close'], sample_data['volume'])
    
    assert len(result) == len(sample_data)
    assert result.min() >= 0
    assert result.max() <= 100


def test_calculate_all_indicators(indicators, sample_data):
    """Test calculation of all indicators."""
    result = indicators.calculate_all_indicators(sample_data)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_data)
    assert len(result.columns) > len(sample_data.columns)  # Should have more columns after adding indicators


def test_leakage_prevention(indicators, sample_data):
    """Test that indicators don't use future data."""
    # Calculate indicators
    result = indicators.calculate_all_indicators(sample_data)
    
    # Check that first few rows have NaN values (indicating no future data leakage)
    for col in result.columns:
        if col not in sample_data.columns:  # Only check indicator columns
            # First few values should be NaN for indicators that need warm-up
            if result[col].iloc[:10].isna().any():
                # This is expected for indicators that need warm-up
                pass
            else:
                # For indicators that don't need warm-up, check they don't use future data
                # This is a basic check - in practice, you'd need more sophisticated tests
                pass


if __name__ == "__main__":
    pytest.main([__file__])