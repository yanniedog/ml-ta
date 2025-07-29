"""
Backtesting module for technical analysis strategies.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import Config, calculate_sharpe_ratio, calculate_max_drawdown, calculate_calmar_ratio


class Backtester:
    """Backtesting engine with transaction costs and realistic position management."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.taker_fee_bps = config.backtest["taker_fee_bps"] / 10000  # Convert to decimal
        self.slippage_bps = config.backtest["slippage_bps"] / 10000
        self.position_threshold = config.backtest["position_threshold"]
        self.fixed_notional = config.backtest["fixed_notional"]
        
        # Total transaction cost per side
        self.total_cost_per_side = self.taker_fee_bps + self.slippage_bps
        
    def calculate_position_size(self, price: float) -> float:
        """Calculate position size based on fixed notional."""
        return self.fixed_notional / price
    
    def calculate_transaction_cost(self, price: float, position_size: float, side: str) -> float:
        """Calculate transaction cost for a trade."""
        notional = price * position_size
        cost = notional * self.total_cost_per_side
        return cost
    
    def execute_trade(self, entry_price: float, exit_price: float, position_size: float) -> Dict:
        """Execute a trade and calculate P&L with costs."""
        # Entry costs
        entry_cost = self.calculate_transaction_cost(entry_price, position_size, "buy")
        
        # Exit costs
        exit_cost = self.calculate_transaction_cost(exit_price, position_size, "sell")
        
        # Calculate P&L
        gross_pnl = (exit_price - entry_price) * position_size
        total_cost = entry_cost + exit_cost
        net_pnl = gross_pnl - total_cost
        
        return {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'gross_pnl': gross_pnl,
            'entry_cost': entry_cost,
            'exit_cost': exit_cost,
            'total_cost': total_cost,
            'net_pnl': net_pnl,
            'notional': entry_price * position_size
        }
    
    def run_backtest(self, df: pd.DataFrame, predictions: pd.Series, 
                    probabilities: Optional[pd.Series] = None) -> Dict:
        """Run backtest on predictions."""
        self.logger.info("Starting backtest")
        
        if len(df) != len(predictions):
            raise ValueError("DataFrame and predictions must have same length")
        
        # Initialize tracking variables
        position = 0  # 0 = no position, 1 = long
        entry_price = 0
        entry_index = 0
        trades = []
        equity_curve = []
        current_equity = self.fixed_notional
        
        # Get price data
        prices = df['close'].values
        timestamps = df['timestamp'].values
        
        for i in range(len(df)):
            current_price = prices[i]
            current_prediction = predictions.iloc[i]
            current_prob = probabilities.iloc[i] if probabilities is not None else 0.5
            
            # Check for position entry
            if position == 0 and current_prediction == 1 and current_prob >= self.position_threshold:
                # Enter long position at next open
                if i + 1 < len(df):
                    entry_price = df.iloc[i + 1]['open']  # Next open
                    position_size = self.calculate_position_size(entry_price)
                    position = 1
                    entry_index = i + 1
                    self.logger.debug(f"Entered long position at {entry_price}")
            
            # Check for position exit
            elif position == 1:
                # Exit after horizon or if prediction changes
                horizon = 1  # Default horizon, could be made configurable
                exit_condition = (i >= entry_index + horizon) or (current_prediction == 0)
                
                if exit_condition:
                    # Exit at current close
                    exit_price = current_price
                    trade_result = self.execute_trade(entry_price, exit_price, position_size)
                    
                    # Add trade metadata
                    trade_result.update({
                        'entry_time': timestamps[entry_index],
                        'exit_time': timestamps[i],
                        'entry_index': entry_index,
                        'exit_index': i,
                        'hold_period': i - entry_index
                    })
                    
                    trades.append(trade_result)
                    
                    # Update equity
                    current_equity += trade_result['net_pnl']
                    
                    # Reset position
                    position = 0
                    entry_price = 0
                    entry_index = 0
                    
                    self.logger.debug(f"Exited position at {exit_price}, P&L: {trade_result['net_pnl']:.2f}")
            
            # Record equity curve
            equity_curve.append(current_equity)
        
        # Close any open position at the end
        if position == 1:
            exit_price = prices[-1]
            trade_result = self.execute_trade(entry_price, exit_price, position_size)
            trade_result.update({
                'entry_time': timestamps[entry_index],
                'exit_time': timestamps[-1],
                'entry_index': entry_index,
                'exit_index': len(df) - 1,
                'hold_period': len(df) - 1 - entry_index
            })
            trades.append(trade_result)
            current_equity += trade_result['net_pnl']
            equity_curve[-1] = current_equity
        
        # Calculate performance metrics
        performance = self.calculate_performance_metrics(trades, equity_curve, df)
        
        self.logger.info(f"Backtest completed: {len(trades)} trades, Final equity: {current_equity:.2f}")
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'performance': performance,
            'final_equity': current_equity
        }
    
    def calculate_performance_metrics(self, trades: List[Dict], equity_curve: List[float], 
                                   df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not trades:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'calmar_ratio': 0,
                'profit_factor': 0,
                'hit_rate': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'avg_hold_period': 0,
                'turnover': 0
            }
        
        # Convert trades to DataFrame for easier analysis
        trades_df = pd.DataFrame(trades)
        
        # Basic metrics
        total_return = (equity_curve[-1] - self.fixed_notional) / self.fixed_notional
        total_trades = len(trades)
        winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
        losing_trades = len(trades_df[trades_df['net_pnl'] < 0])
        
        # Win/loss metrics
        avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean() if losing_trades > 0 else 0
        largest_win = trades_df['net_pnl'].max()
        largest_loss = trades_df['net_pnl'].min()
        
        # Hit rate and profit factor
        hit_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
        
        # Risk metrics
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        
        sharpe_ratio = calculate_sharpe_ratio(returns) if len(returns) > 0 else 0
        max_drawdown = calculate_max_drawdown(equity_series)
        calmar_ratio = calculate_calmar_ratio(returns, max_drawdown)
        
        # Turnover
        total_notional = trades_df['notional'].sum()
        turnover = total_notional / self.fixed_notional if self.fixed_notional > 0 else 0
        
        # Average hold period
        avg_hold_period = trades_df['hold_period'].mean()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'profit_factor': profit_factor,
            'hit_rate': hit_rate,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_hold_period': avg_hold_period,
            'turnover': turnover
        }
    
    def run_backtest_with_model(self, df: pd.DataFrame, model, label_column: str) -> Dict:
        """Run backtest using trained model predictions."""
        self.logger.info(f"Running backtest with model for {label_column}")
        
        # Prepare features
        exclude_columns = ['timestamp', label_column]
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        feature_columns = [col for col in feature_columns if not col.startswith('return_')]
        
        X = df[feature_columns]
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Convert to pandas Series
        predictions_series = pd.Series(predictions, index=df.index)
        probabilities_series = pd.Series(probabilities, index=df.index) if probabilities is not None else None
        
        # Run backtest
        results = self.run_backtest(df, predictions_series, probabilities_series)
        
        return results
    
    def save_backtest_results(self, results: Dict, symbol: str, interval: str, 
                            label_column: str, output_dir: str) -> None:
        """Save backtest results to files."""
        from .utils import ensure_directory, save_parquet
        
        ensure_directory(output_dir)
        
        # Save trades
        trades_df = pd.DataFrame(results['trades'])
        trades_file = f"{output_dir}/{symbol}_{interval}_{label_column}_trades.parquet"
        save_parquet(trades_df, trades_file)
        
        # Save equity curve
        equity_df = pd.DataFrame({
            'equity': results['equity_curve']
        })
        equity_file = f"{output_dir}/{symbol}_{interval}_{label_column}_equity.parquet"
        save_parquet(equity_df, equity_file)
        
        # Save performance metrics
        performance_df = pd.DataFrame([results['performance']])
        performance_file = f"{output_dir}/{symbol}_{interval}_{label_column}_performance.csv"
        performance_df.to_csv(performance_file, index=False)
        
        self.logger.info(f"Saved backtest results to {output_dir}")


def main():
    """Main function for backtesting."""
    from .utils import load_config, setup_logging, set_deterministic_seed, load_parquet
    from .model import ModelTrainer
    
    # Load configuration
    config = load_config("config/settings.yaml")
    
    # Setup logging
    logger = setup_logging(config)
    
    # Set deterministic seed
    set_deterministic_seed(config.app["seed"])
    
    # Load sample data and run backtest
    try:
        gold_dir = f"{config.paths['data']}/gold"
        gold_files = list(Path(gold_dir).glob("*.parquet"))
        
        if gold_files:
            df = load_parquet(str(gold_files[0]))
            
            # Train a model first
            trainer = ModelTrainer(config)
            label_columns = [col for col in df.columns if col.startswith('label_')]
            
            if label_columns:
                results = trainer.train_single_model(df, label_columns[0], "classification")
                model = results['model']
                
                # Run backtest
                backtester = Backtester(config)
                backtest_results = backtester.run_backtest_with_model(df, model, label_columns[0])
                
                print("Backtest Results:")
                print(f"Total Return: {backtest_results['performance']['total_return']:.2%}")
                print(f"Sharpe Ratio: {backtest_results['performance']['sharpe_ratio']:.2f}")
                print(f"Max Drawdown: {backtest_results['performance']['max_drawdown']:.2%}")
                print(f"Hit Rate: {backtest_results['performance']['hit_rate']:.2%}")
                print(f"Total Trades: {backtest_results['performance']['total_trades']}")
                
        else:
            logger.error("No gold data files found")
            
    except Exception as e:
        logger.error(f"Error in backtesting: {e}")


if __name__ == "__main__":
    main()