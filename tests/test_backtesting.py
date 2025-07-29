"""
Comprehensive tests for backtesting with realistic trading constraints.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.backtest import Backtester
from src.risk_management import RiskManager, RiskLimits, Position, PositionSide, MarketImpactModel
from src.utils import Config


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 1000
    
    # Generate realistic price data with some trends
    returns = np.random.normal(0, 0.02, n)
    # Add some trend
    trend = np.linspace(0, 0.1, n)
    returns += trend
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


class TestMarketImpactModel:
    """Test market impact modeling."""
    
    def test_slippage_calculation(self, config):
        """Test slippage calculation."""
        impact_model = MarketImpactModel(config)
        
        # Test different scenarios
        scenarios = [
            {'order_size': 1000, 'market_volume': 1000000, 'price': 100, 'volatility': 0.02},
            {'order_size': 10000, 'market_volume': 1000000, 'price': 100, 'volatility': 0.05},
            {'order_size': 100000, 'market_volume': 1000000, 'price': 100, 'volatility': 0.01}
        ]
        
        for scenario in scenarios:
            slippage = impact_model.calculate_slippage(**scenario)
            
            # Check bounds
            assert slippage >= impact_model.min_slippage
            assert slippage <= impact_model.max_slippage
            
            # Larger orders should have higher slippage
            if scenario['order_size'] > 1000:
                assert slippage > impact_model.min_slippage
    
    def test_effective_price_calculation(self, config):
        """Test effective price calculation."""
        impact_model = MarketImpactModel(config)
        
        base_price = 100.0
        slippage = 0.001  # 0.1%
        
        # Test long position
        long_price = impact_model.calculate_effective_price(PositionSide.LONG, base_price, slippage)
        assert long_price > base_price
        
        # Test short position
        short_price = impact_model.calculate_effective_price(PositionSide.SHORT, base_price, slippage)
        assert short_price < base_price
        
        # Test flat position
        flat_price = impact_model.calculate_effective_price(PositionSide.FLAT, base_price, slippage)
        assert flat_price == base_price


class TestPositionSizing:
    """Test position sizing logic."""
    
    def test_risk_based_position_sizing(self, config):
        """Test risk-based position sizing."""
        risk_limits = RiskLimits()
        position_sizer = RiskManager(config, risk_limits).position_sizer
        
        capital = 100000
        risk_per_trade = 0.01  # 1%
        entry_price = 100
        stop_loss = 95
        
        position_size = position_sizer.calculate_position_size(capital, risk_per_trade, entry_price, stop_loss)
        
        # Check that position size is reasonable
        assert position_size > 0
        assert position_size * entry_price <= capital * risk_limits.max_position_size
    
    def test_capital_based_position_sizing(self, config):
        """Test capital-based position sizing."""
        risk_limits = RiskLimits()
        position_sizer = RiskManager(config, risk_limits).position_sizer
        
        capital = 100000
        risk_per_trade = 0.01
        entry_price = 100
        
        position_size = position_sizer.calculate_position_size(capital, risk_per_trade, entry_price)
        
        # Check that position size respects capital limits
        position_value = position_size * entry_price
        max_allowed = capital * risk_limits.max_position_size
        
        assert position_value <= max_allowed
    
    def test_risk_metrics_calculation(self, config):
        """Test risk metrics calculation."""
        risk_limits = RiskLimits()
        risk_manager = RiskManager(config, risk_limits)
        
        # Create some positions
        positions = [
            Position(
                symbol="TEST",
                side=PositionSide.LONG,
                size=100,
                entry_price=100,
                entry_time=pd.Timestamp.now()
            ),
            Position(
                symbol="TEST2",
                side=PositionSide.SHORT,
                size=50,
                entry_price=200,
                entry_time=pd.Timestamp.now()
            )
        ]
        
        current_prices = {"TEST": 110, "TEST2": 190}
        
        metrics = risk_manager.position_sizer.calculate_risk_metrics(positions, current_prices)
        
        # Check metrics
        assert 'total_exposure' in metrics
        assert 'total_market_value' in metrics
        assert 'leverage' in metrics
        assert 'margin_ratio' in metrics
        assert 'position_count' in metrics
        
        assert metrics['position_count'] == 2


class TestRiskManagement:
    """Test risk management functionality."""
    
    def test_position_limits_check(self, config):
        """Test position limits checking."""
        risk_limits = RiskLimits()
        risk_manager = RiskManager(config, risk_limits)
        
        # Test valid position
        valid_position = Position(
            symbol="TEST",
            side=PositionSide.LONG,
            size=100,
            entry_price=100,
            entry_time=pd.Timestamp.now()
        )
        
        is_allowed, message = risk_manager.check_position_limits(valid_position, 100000)
        assert is_allowed
        assert "approved" in message
        
        # Test position that exceeds limits
        large_position = Position(
            symbol="TEST",
            side=PositionSide.LONG,
            size=10000,  # Very large position
            entry_price=100,
            entry_time=pd.Timestamp.now()
        )
        
        is_allowed, message = risk_manager.check_position_limits(large_position, 100000)
        assert not is_allowed
        assert "exceeds" in message
    
    def test_position_management(self, config):
        """Test position management."""
        risk_limits = RiskLimits()
        risk_manager = RiskManager(config, risk_limits)
        
        # Add position
        position = Position(
            symbol="TEST",
            side=PositionSide.LONG,
            size=100,
            entry_price=100,
            entry_time=pd.Timestamp.now(),
            stop_loss=95,
            take_profit=110
        )
        
        success = risk_manager.add_position(position)
        assert success
        assert len(risk_manager.positions) == 1
        
        # Close position
        trade_result = risk_manager.close_position("TEST", 110, pd.Timestamp.now())
        assert trade_result is not None
        assert trade_result['symbol'] == "TEST"
        assert trade_result['net_pnl'] > 0  # Should be profitable
        assert len(risk_manager.positions) == 0
    
    def test_stop_loss_execution(self, config):
        """Test stop loss execution."""
        risk_limits = RiskLimits()
        risk_manager = RiskManager(config, risk_limits)
        
        # Add position with stop loss
        position = Position(
            symbol="TEST",
            side=PositionSide.LONG,
            size=100,
            entry_price=100,
            entry_time=pd.Timestamp.now(),
            stop_loss=95
        )
        
        risk_manager.add_position(position)
        
        # Trigger stop loss
        current_prices = {"TEST": 94}  # Below stop loss
        closed_trades = risk_manager.check_stop_losses(current_prices, pd.Timestamp.now())
        
        assert len(closed_trades) == 1
        assert len(risk_manager.positions) == 0
    
    def test_portfolio_risk_calculation(self, config):
        """Test portfolio risk calculation."""
        risk_limits = RiskLimits()
        risk_manager = RiskManager(config, risk_limits)
        
        # Add some positions
        positions = [
            Position(
                symbol="TEST",
                side=PositionSide.LONG,
                size=100,
                entry_price=100,
                entry_time=pd.Timestamp.now()
            )
        ]
        
        risk_manager.positions = positions
        current_prices = {"TEST": 110}
        
        risk_metrics = risk_manager.calculate_portfolio_risk(current_prices)
        
        # Check risk metrics
        assert 'total_exposure' in risk_metrics
        assert 'total_market_value' in risk_metrics
        assert 'leverage' in risk_metrics
        assert 'margin_ratio' in risk_metrics
        assert 'var_95' in risk_metrics


class TestBacktesterWithRealisticConstraints:
    """Test backtester with realistic trading constraints."""
    
    def test_backtester_initialization(self, config):
        """Test backtester initialization."""
        backtester = Backtester(config)
        
        assert backtester.config == config
        assert backtester.risk_manager is not None
        assert backtester.market_impact is not None
    
    def test_simple_strategy_backtest(self, config, sample_data):
        """Test simple strategy backtest."""
        backtester = Backtester(config)
        
        # Create simple strategy signals
        signals = pd.DataFrame({
            'timestamp': sample_data['timestamp'],
            'signal': np.random.choice([-1, 0, 1], len(sample_data)),
            'confidence': np.random.uniform(0.5, 1.0, len(sample_data))
        })
        
        # Run backtest
        results = backtester.run_backtest(sample_data, signals)
        
        # Check results
        assert 'trades' in results
        assert 'portfolio_value' in results
        assert 'returns' in results
        assert 'metrics' in results
        
        # Check that trades were executed
        assert len(results['trades']) > 0
    
    def test_market_impact_integration(self, config, sample_data):
        """Test market impact integration in backtest."""
        backtester = Backtester(config)
        
        # Create signals
        signals = pd.DataFrame({
            'timestamp': sample_data['timestamp'],
            'signal': np.random.choice([-1, 0, 1], len(sample_data)),
            'confidence': np.random.uniform(0.5, 1.0, len(sample_data))
        })
        
        # Run backtest
        results = backtester.run_backtest(sample_data, signals)
        
        # Check that market impact was considered
        trades = results['trades']
        if len(trades) > 0:
            # Check that execution prices differ from market prices
            for trade in trades:
                if 'execution_price' in trade and 'market_price' in trade:
                    assert trade['execution_price'] != trade['market_price']
    
    def test_risk_limits_enforcement(self, config, sample_data):
        """Test that risk limits are enforced."""
        backtester = Backtester(config)
        
        # Create aggressive signals
        signals = pd.DataFrame({
            'timestamp': sample_data['timestamp'],
            'signal': np.ones(len(sample_data)),  # All buy signals
            'confidence': np.ones(len(sample_data))
        })
        
        # Run backtest
        results = backtester.run_backtest(sample_data, signals)
        
        # Check that risk limits were respected
        portfolio_values = results['portfolio_value']
        if len(portfolio_values) > 1:
            # Check for reasonable drawdown
            max_value = portfolio_values.max()
            min_value = portfolio_values.min()
            drawdown = (max_value - min_value) / max_value
            
            assert drawdown < 0.5  # Should not exceed 50% drawdown
    
    def test_transaction_costs(self, config, sample_data):
        """Test transaction costs calculation."""
        backtester = Backtester(config)
        
        # Create signals
        signals = pd.DataFrame({
            'timestamp': sample_data['timestamp'],
            'signal': np.random.choice([-1, 0, 1], len(sample_data)),
            'confidence': np.random.uniform(0.5, 1.0, len(sample_data))
        })
        
        # Run backtest
        results = backtester.run_backtest(sample_data, signals)
        
        # Check that transaction costs were applied
        trades = results['trades']
        if len(trades) > 0:
            for trade in trades:
                if 'transaction_cost' in trade:
                    assert trade['transaction_cost'] >= 0
    
    def test_position_sizing_with_risk(self, config, sample_data):
        """Test position sizing with risk management."""
        backtester = Backtester(config)
        
        # Create signals with varying confidence
        signals = pd.DataFrame({
            'timestamp': sample_data['timestamp'],
            'signal': np.random.choice([-1, 0, 1], len(sample_data)),
            'confidence': np.random.uniform(0.5, 1.0, len(sample_data))
        })
        
        # Run backtest
        results = backtester.run_backtest(sample_data, signals)
        
        # Check that position sizes are reasonable
        trades = results['trades']
        if len(trades) > 0:
            for trade in trades:
                if 'position_size' in trade:
                    # Position size should be positive and reasonable
                    assert trade['position_size'] > 0
                    assert trade['position_size'] <= backtester.config.backtest['fixed_notional'] / 100  # Max position value


class TestStressTesting:
    """Test stress testing functionality."""
    
    def test_stress_test_scenarios(self, config):
        """Test stress test scenario creation."""
        from src.risk_management import StressTester
        
        stress_tester = StressTester(config)
        scenarios = stress_tester.create_scenarios()
        
        assert len(scenarios) > 0
        for scenario in scenarios:
            assert 'name' in scenario
            assert 'description' in scenario
            assert 'prices' in scenario
    
    def test_stress_test_execution(self, config):
        """Test stress test execution."""
        from src.risk_management import StressTester, RiskManager, RiskLimits
        
        stress_tester = StressTester(config)
        risk_manager = RiskManager(config, RiskLimits())
        
        # Add some positions
        position = Position(
            symbol="SOLUSDT",
            side=PositionSide.LONG,
            size=100,
            entry_price=100,
            entry_time=pd.Timestamp.now()
        )
        risk_manager.add_position(position)
        
        # Run stress tests
        scenarios = stress_tester.create_scenarios()
        results = stress_tester.run_stress_tests(risk_manager, scenarios)
        
        assert len(results) == len(scenarios)
        for scenario_name, result in results.items():
            assert 'risk_metrics' in result
            assert 'scenario' in result


class TestPerformanceMetrics:
    """Test performance metrics calculation."""
    
    def test_sharpe_ratio_calculation(self, config):
        """Test Sharpe ratio calculation."""
        # Create sample returns
        returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
        
        from src.backtest import calculate_sharpe_ratio
        sharpe = calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_max_drawdown_calculation(self, config):
        """Test maximum drawdown calculation."""
        # Create sample portfolio values
        portfolio_values = 10000 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 252)))
        
        from src.backtest import calculate_max_drawdown
        max_dd = calculate_max_drawdown(portfolio_values)
        
        assert isinstance(max_dd, float)
        assert 0 <= max_dd <= 1  # Should be between 0 and 1
    
    def test_win_rate_calculation(self, config):
        """Test win rate calculation."""
        # Create sample trades
        trades = [
            {'pnl': 100},
            {'pnl': -50},
            {'pnl': 200},
            {'pnl': -30},
            {'pnl': 150}
        ]
        
        from src.backtest import calculate_win_rate
        win_rate = calculate_win_rate(trades)
        
        assert isinstance(win_rate, float)
        assert 0 <= win_rate <= 1
    
    def test_profit_factor_calculation(self, config):
        """Test profit factor calculation."""
        # Create sample trades
        trades = [
            {'pnl': 100},
            {'pnl': -50},
            {'pnl': 200},
            {'pnl': -30},
            {'pnl': 150}
        ]
        
        from src.backtest import calculate_profit_factor
        profit_factor = calculate_profit_factor(trades)
        
        assert isinstance(profit_factor, float)
        assert profit_factor > 0


class TestErrorHandling:
    """Test error handling in backtesting."""
    
    def test_invalid_signals_handling(self, config, sample_data):
        """Test handling of invalid signals."""
        backtester = Backtester(config)
        
        # Create invalid signals (missing required columns)
        invalid_signals = pd.DataFrame({
            'timestamp': sample_data['timestamp']
            # Missing 'signal' and 'confidence' columns
        })
        
        # Should handle gracefully
        with pytest.raises(KeyError):
            backtester.run_backtest(sample_data, invalid_signals)
    
    def test_empty_data_handling(self, config):
        """Test handling of empty data."""
        backtester = Backtester(config)
        
        # Create empty data
        empty_data = pd.DataFrame()
        empty_signals = pd.DataFrame()
        
        # Should handle gracefully
        with pytest.raises(ValueError):
            backtester.run_backtest(empty_data, empty_signals)
    
    def test_mismatched_timestamps(self, config, sample_data):
        """Test handling of mismatched timestamps."""
        backtester = Backtester(config)
        
        # Create signals with different timestamps
        signals = pd.DataFrame({
            'timestamp': pd.date_range('2023-02-01', periods=len(sample_data), freq='1min'),
            'signal': np.random.choice([-1, 0, 1], len(sample_data)),
            'confidence': np.random.uniform(0.5, 1.0, len(sample_data))
        })
        
        # Should handle gracefully
        with pytest.raises(ValueError):
            backtester.run_backtest(sample_data, signals)


if __name__ == "__main__":
    pytest.main([__file__])