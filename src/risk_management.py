"""
Risk management module for realistic trading constraints.
"""
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

from .utils import Config


class PositionSide(Enum):
    """Position side enumeration."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Position:
    """Position data class."""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    entry_time: pd.Timestamp
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_loss: Optional[float] = None


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    max_position_size: float = 0.1  # 10% of capital per position
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk
    max_drawdown: float = 0.15  # 15% max drawdown
    max_leverage: float = 2.0  # 2x max leverage
    min_margin_ratio: float = 0.5  # 50% minimum margin ratio


class MarketImpactModel:
    """Market impact model for realistic slippage calculation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Market impact parameters
        self.impact_factor = 0.0001  # 1 bps per $1M
        self.min_slippage = 0.0001  # 1 bps minimum
        self.max_slippage = 0.01  # 100 bps maximum
    
    def calculate_slippage(self, order_size: float, market_volume: float, 
                          price: float, volatility: float) -> float:
        """Calculate realistic slippage based on order size and market conditions."""
        
        # Base slippage from order size relative to market volume
        volume_impact = (order_size * price) / (market_volume * price)
        
        # Volatility adjustment
        volatility_impact = min(volatility * 0.1, 0.01)  # Cap at 1%
        
        # Calculate total slippage
        slippage = self.impact_factor * volume_impact + volatility_impact
        
        # Apply bounds
        slippage = max(self.min_slippage, min(slippage, self.max_slippage))
        
        return slippage
    
    def calculate_effective_price(self, side: PositionSide, base_price: float, 
                                slippage: float) -> float:
        """Calculate effective execution price including slippage."""
        if side == PositionSide.LONG:
            return base_price * (1 + slippage)
        elif side == PositionSide.SHORT:
            return base_price * (1 - slippage)
        else:
            return base_price


class PositionSizer:
    """Position sizing based on risk management rules."""
    
    def __init__(self, config: Config, risk_limits: RiskLimits):
        self.config = config
        self.risk_limits = risk_limits
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size(self, capital: float, risk_per_trade: float, 
                              entry_price: float, stop_loss: Optional[float] = None) -> float:
        """Calculate position size based on risk management rules."""
        
        # Risk-based position sizing
        if stop_loss is not None:
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share > 0:
                risk_based_size = (capital * risk_per_trade) / risk_per_share
            else:
                risk_based_size = 0
        else:
            # Default to 1% risk per trade
            risk_based_size = (capital * risk_per_trade) / (entry_price * 0.01)
        
        # Capital-based position sizing
        max_capital_size = capital * self.risk_limits.max_position_size
        capital_based_size = max_capital_size / entry_price
        
        # Take the smaller of the two
        position_size = min(risk_based_size, capital_based_size)
        
        # Ensure positive size
        position_size = max(0, position_size)
        
        self.logger.info(f"Calculated position size: {position_size:.4f} shares")
        return position_size
    
    def calculate_risk_metrics(self, positions: List[Position], 
                             current_prices: Dict[str, float]) -> Dict:
        """Calculate current risk metrics."""
        total_exposure = 0
        total_market_value = 0
        
        for position in positions:
            current_price = current_prices.get(position.symbol, position.entry_price)
            market_value = position.size * current_price
            total_market_value += market_value
            
            if position.side == PositionSide.LONG:
                total_exposure += market_value
            elif position.side == PositionSide.SHORT:
                total_exposure -= market_value
        
        # Calculate leverage
        leverage = abs(total_exposure) / total_market_value if total_market_value > 0 else 0
        
        # Calculate margin ratio
        margin_ratio = 1 - (abs(total_exposure) / total_market_value) if total_market_value > 0 else 1
        
        return {
            'total_exposure': total_exposure,
            'total_market_value': total_market_value,
            'leverage': leverage,
            'margin_ratio': margin_ratio,
            'position_count': len(positions)
        }


class RiskManager:
    """Comprehensive risk management system."""
    
    def __init__(self, config: Config, risk_limits: RiskLimits):
        self.config = config
        self.risk_limits = risk_limits
        self.logger = logging.getLogger(__name__)
        self.position_sizer = PositionSizer(config, risk_limits)
        self.market_impact = MarketImpactModel(config)
        
        # Risk tracking
        self.positions: List[Position] = []
        self.risk_history: List[Dict] = []
    
    def check_position_limits(self, new_position: Position, 
                            current_capital: float) -> Tuple[bool, str]:
        """Check if new position violates risk limits."""
        
        # Check position size limit
        position_value = new_position.size * new_position.entry_price
        if position_value > current_capital * self.risk_limits.max_position_size:
            return False, f"Position size {position_value:.2f} exceeds limit"
        
        # Check leverage limit
        test_positions = self.positions + [new_position]
        risk_metrics = self.position_sizer.calculate_risk_metrics(
            test_positions, {new_position.symbol: new_position.entry_price}
        )
        
        if risk_metrics['leverage'] > self.risk_limits.max_leverage:
            return False, f"Leverage {risk_metrics['leverage']:.2f} exceeds limit"
        
        if risk_metrics['margin_ratio'] < self.risk_limits.min_margin_ratio:
            return False, f"Margin ratio {risk_metrics['margin_ratio']:.2f} below limit"
        
        return True, "Position approved"
    
    def add_position(self, position: Position) -> bool:
        """Add a new position with risk checks."""
        # Calculate current capital (simplified)
        current_capital = sum(p.size * p.entry_price for p in self.positions) + 10000
        
        # Check risk limits
        is_allowed, message = self.check_position_limits(position, current_capital)
        
        if is_allowed:
            self.positions.append(position)
            self.logger.info(f"Added position: {position.symbol} {position.side.value} {position.size}")
            return True
        else:
            self.logger.warning(f"Position rejected: {message}")
            return False
    
    def close_position(self, symbol: str, exit_price: float, 
                      exit_time: pd.Timestamp) -> Optional[Dict]:
        """Close a position and calculate P&L."""
        for i, position in enumerate(self.positions):
            if position.symbol == symbol:
                # Calculate P&L
                if position.side == PositionSide.LONG:
                    pnl = (exit_price - position.entry_price) * position.size
                else:  # SHORT
                    pnl = (position.entry_price - exit_price) * position.size
                
                # Calculate transaction costs
                entry_cost = self.market_impact.calculate_slippage(
                    position.size, 1000000, position.entry_price, 0.02
                ) * position.entry_price * position.size
                
                exit_cost = self.market_impact.calculate_slippage(
                    position.size, 1000000, exit_price, 0.02
                ) * exit_price * position.size
                
                total_cost = entry_cost + exit_cost
                net_pnl = pnl - total_cost
                
                # Remove position
                closed_position = self.positions.pop(i)
                
                trade_result = {
                    'symbol': symbol,
                    'side': position.side.value,
                    'entry_price': position.entry_price,
                    'exit_price': exit_price,
                    'size': position.size,
                    'gross_pnl': pnl,
                    'total_cost': total_cost,
                    'net_pnl': net_pnl,
                    'entry_time': position.entry_time,
                    'exit_time': exit_time,
                    'duration': exit_time - position.entry_time
                }
                
                self.logger.info(f"Closed position: {trade_result}")
                return trade_result
        
        return None
    
    def check_stop_losses(self, current_prices: Dict[str, float], 
                         current_time: pd.Timestamp) -> List[Dict]:
        """Check and execute stop losses."""
        closed_trades = []
        
        for position in self.positions[:]:  # Copy list to avoid modification during iteration
            current_price = current_prices.get(position.symbol)
            if current_price is None:
                continue
            
            # Check stop loss
            if position.stop_loss is not None:
                if (position.side == PositionSide.LONG and current_price <= position.stop_loss) or \
                   (position.side == PositionSide.SHORT and current_price >= position.stop_loss):
                    
                    trade_result = self.close_position(position.symbol, position.stop_loss, current_time)
                    if trade_result:
                        closed_trades.append(trade_result)
            
            # Check take profit
            if position.take_profit is not None:
                if (position.side == PositionSide.LONG and current_price >= position.take_profit) or \
                   (position.side == PositionSide.SHORT and current_price <= position.take_profit):
                    
                    trade_result = self.close_position(position.symbol, position.take_profit, current_time)
                    if trade_result:
                        closed_trades.append(trade_result)
        
        return closed_trades
    
    def calculate_portfolio_risk(self, current_prices: Dict[str, float]) -> Dict:
        """Calculate comprehensive portfolio risk metrics."""
        if not self.positions:
            return {
                'total_exposure': 0,
                'total_market_value': 0,
                'leverage': 0,
                'margin_ratio': 1,
                'position_count': 0,
                'var_95': 0,
                'max_drawdown': 0
            }
        
        # Get risk metrics
        risk_metrics = self.position_sizer.calculate_risk_metrics(self.positions, current_prices)
        
        # Calculate VaR (simplified)
        position_returns = []
        for position in self.positions:
            current_price = current_prices.get(position.symbol, position.entry_price)
            if position.side == PositionSide.LONG:
                ret = (current_price - position.entry_price) / position.entry_price
            else:
                ret = (position.entry_price - current_price) / position.entry_price
            position_returns.append(ret)
        
        if position_returns:
            var_95 = np.percentile(position_returns, 5)
        else:
            var_95 = 0
        
        return {
            **risk_metrics,
            'var_95': var_95,
            'max_drawdown': 0  # Would need historical data to calculate
        }
    
    def get_positions_summary(self) -> Dict:
        """Get summary of current positions."""
        if not self.positions:
            return {'total_positions': 0, 'positions': []}
        
        summary = {
            'total_positions': len(self.positions),
            'long_positions': len([p for p in self.positions if p.side == PositionSide.LONG]),
            'short_positions': len([p for p in self.positions if p.side == PositionSide.SHORT]),
            'positions': []
        }
        
        for position in self.positions:
            summary['positions'].append({
                'symbol': position.symbol,
                'side': position.side.value,
                'size': position.size,
                'entry_price': position.entry_price,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit
            })
        
        return summary


class StressTester:
    """Stress testing framework for portfolio risk assessment."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def run_stress_tests(self, portfolio: RiskManager, 
                        scenarios: List[Dict]) -> Dict:
        """Run stress tests on portfolio."""
        results = {}
        
        for scenario in scenarios:
            scenario_name = scenario.get('name', 'Unknown')
            self.logger.info(f"Running stress test: {scenario_name}")
            
            # Apply scenario to current prices
            scenario_prices = scenario.get('prices', {})
            
            # Calculate portfolio risk under scenario
            risk_metrics = portfolio.calculate_portfolio_risk(scenario_prices)
            
            results[scenario_name] = {
                'risk_metrics': risk_metrics,
                'scenario': scenario
            }
        
        return results
    
    def create_scenarios(self) -> List[Dict]:
        """Create standard stress test scenarios."""
        scenarios = [
            {
                'name': 'Market Crash',
                'description': '50% decline in all assets',
                'prices': {'SOLUSDT': 50, 'BTCUSDT': 20000}
            },
            {
                'name': 'Volatility Spike',
                'description': 'High volatility scenario',
                'prices': {'SOLUSDT': 80, 'BTCUSDT': 35000}
            },
            {
                'name': 'Liquidity Crisis',
                'description': 'Reduced liquidity scenario',
                'prices': {'SOLUSDT': 60, 'BTCUSDT': 25000}
            }
        ]
        
        return scenarios


def main():
    """Test risk management functionality."""
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from src.utils import load_config, setup_logging
    
    # Load configuration
    config = load_config("config/settings.yaml")
    setup_logging(config)
    
    # Create risk manager
    risk_limits = RiskLimits()
    risk_manager = RiskManager(config, risk_limits)
    
    # Test position management
    position = Position(
        symbol="SOLUSDT",
        side=PositionSide.LONG,
        size=100,
        entry_price=100.0,
        entry_time=pd.Timestamp.now(),
        stop_loss=95.0,
        take_profit=110.0
    )
    
    # Add position
    success = risk_manager.add_position(position)
    print(f"Position added: {success}")
    
    # Get portfolio summary
    summary = risk_manager.get_positions_summary()
    print(f"Portfolio summary: {summary}")


if __name__ == "__main__":
    main()