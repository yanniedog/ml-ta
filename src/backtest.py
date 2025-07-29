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
        
        # Handle timestamp column - use index if timestamp not available
        if 'timestamp' in df.columns:
            timestamps = df['timestamp'].values
        else:
            # Use index as timestamp if no timestamp column
            timestamps = df.index.values
        
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
    
    def run_backtest_with_model(self, df: pd.DataFrame, model, label_column: str, 
                               fitted_feature_engineer=None) -> Dict:
        """Run backtest using trained model predictions with consistent feature preparation."""
        self.logger.info(f"Running backtest with model for {label_column}")
        
        try:
            # Use the fitted feature engineer if provided, otherwise create a new one
            if fitted_feature_engineer is None:
                from .features import FeatureEngineer
                feature_engineer = FeatureEngineer(self.config)
                # Build feature matrix using the same method as training
                feature_df = feature_engineer.build_feature_matrix(df, fit_pipeline=False)
            else:
                # Use the fitted feature engineer
                feature_df = fitted_feature_engineer.build_feature_matrix(df, fit_pipeline=False)
            
            if feature_df.empty:
                self.logger.error("No features available for backtesting")
                return {}
            
            # Get actual labels from original data first
            if label_column not in df.columns:
                self.logger.error(f"Label column {label_column} not found in original data")
                return {}
            
            y_true = df[label_column].loc[feature_df.index]
            
            # Prepare features and labels using the same method as model training
            # Remove timestamp and ALL label columns from features (not just the one being predicted)
            exclude_columns = ['timestamp']
            # Exclude all label columns to match model training
            label_columns = [col for col in feature_df.columns if col.startswith('label_')]
            exclude_columns.extend(label_columns)
            
            # Also exclude return columns (they contain future information)
            return_columns = [col for col in feature_df.columns if col.startswith('return_')]
            exclude_columns.extend(return_columns)
            
            # Features to use for prediction
            feature_columns = [col for col in feature_df.columns if col not in exclude_columns]
            
            if not feature_columns:
                self.logger.error("No features available for backtesting")
                return {}
            
            # Prepare feature matrix
            X = feature_df[feature_columns].copy()
            
            # Ensure feature consistency if using fitted feature engineer
            if fitted_feature_engineer is not None:
                X = fitted_feature_engineer.ensure_feature_consistency(X)
            
            # Clean data - handle infinity and extreme values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
            
            # Clip extreme values
            for col in X.columns:
                if X[col].dtype in ['float64', 'float32']:
                    Q1 = X[col].quantile(0.01)
                    Q3 = X[col].quantile(0.99)
                    X[col] = X[col].clip(Q1, Q3)
            
            # Handle different model types
            if hasattr(model, 'predict_proba'):
                # Direct model (LightGBM, etc.)
                predictions_proba = model.predict_proba(X)
                if len(predictions_proba.shape) > 1:
                    predictions = predictions_proba[:, 1]  # Get positive class probability
                else:
                    predictions = predictions_proba
            elif hasattr(model, 'predict'):
                # Ensemble or other model types
                predictions = model.predict(X)
                if hasattr(model, 'predict_proba'):
                    predictions_proba = model.predict_proba(X)
                    if len(predictions_proba.shape) > 1:
                        predictions = predictions_proba[:, 1]
                    else:
                        predictions = predictions_proba
                else:
                    # Convert binary predictions to probabilities
                    predictions = predictions.astype(float)
            else:
                self.logger.error("Model does not have predict method")
                return {}
            
            # Convert predictions to pandas Series for proper indexing
            predictions_series = pd.Series(predictions, index=feature_df.index)
            
            # Run backtest with predictions
            backtest_results = self.run_backtest(feature_df, predictions_series, predictions_series)
            
            # Add model performance metrics
            from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
            
            # Convert predictions to binary for classification metrics
            predictions_binary = (predictions > 0.5).astype(int)
            
            model_metrics = {
                'accuracy': accuracy_score(y_true, predictions_binary),
                'roc_auc': roc_auc_score(y_true, predictions),
                'precision': precision_score(y_true, predictions_binary),
                'recall': recall_score(y_true, predictions_binary),
                'f1': f1_score(y_true, predictions_binary)
            }
            
            backtest_results['model_metrics'] = model_metrics
            self.logger.info(f"Model metrics: {model_metrics}")
            
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"Error in backtest with model: {e}")
            return {}
    
    def calculate_advanced_metrics(self, equity_curve: pd.Series, trades: List[Dict]) -> Dict:
        """Calculate advanced performance metrics."""
        try:
            # Basic metrics
            total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
            returns = equity_curve.pct_change().dropna()
            
            # Risk metrics
            volatility = returns.std() * np.sqrt(252)  # Annualized
            sharpe_ratio = (returns.mean() * 252) / volatility if volatility > 0 else 0
            
            # Drawdown analysis
            rolling_max = equity_curve.expanding().max()
            drawdown = (equity_curve - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Trade analysis
            if trades:
                winning_trades = [t for t in trades if t['net_pnl'] > 0]
                losing_trades = [t for t in trades if t['net_pnl'] < 0]
                
                win_rate = len(winning_trades) / len(trades) if trades else 0
                avg_win = np.mean([t['net_pnl'] for t in winning_trades]) if winning_trades else 0
                avg_loss = np.mean([t['net_pnl'] for t in losing_trades]) if losing_trades else 0
                
                profit_factor = abs(sum([t['net_pnl'] for t in winning_trades]) / 
                                 sum([t['net_pnl'] for t in losing_trades])) if losing_trades else float('inf')
                
                # Risk-adjusted metrics
                calmar_ratio = (total_return * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
                sortino_ratio = (returns.mean() * 252) / (returns[returns < 0].std() * np.sqrt(252)) if returns[returns < 0].std() > 0 else 0
                
                # Additional metrics
                max_consecutive_wins = self._calculate_max_consecutive(trades, 'win')
                max_consecutive_losses = self._calculate_max_consecutive(trades, 'loss')
                avg_trade_duration = np.mean([t.get('hold_period', 0) for t in trades]) if trades else 0
                
                # Volatility metrics
                var_95 = np.percentile(returns, 5)  # 95% VaR
                cvar_95 = returns[returns <= var_95].mean()  # Conditional VaR
                
                return {
                    'total_return': total_return,
                    'annualized_return': total_return * 252 / len(equity_curve),
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'sortino_ratio': sortino_ratio,
                    'max_drawdown': max_drawdown,
                    'calmar_ratio': calmar_ratio,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'max_consecutive_wins': max_consecutive_wins,
                    'max_consecutive_losses': max_consecutive_losses,
                    'avg_trade_duration': avg_trade_duration,
                    'var_95': var_95,
                    'cvar_95': cvar_95,
                    'total_trades': len(trades),
                    'winning_trades': len(winning_trades),
                    'losing_trades': len(losing_trades)
                }
            else:
                return {
                    'total_return': total_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'total_trades': 0
                }
                
        except Exception as e:
            self.logger.error(f"Error calculating advanced metrics: {e}")
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0
            }
    
    def _calculate_max_consecutive(self, trades: List[Dict], trade_type: str) -> int:
        """Calculate maximum consecutive wins or losses."""
        if not trades:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            is_win = trade['net_pnl'] > 0
            is_target = (trade_type == 'win' and is_win) or (trade_type == 'loss' and not is_win)
            
            if is_target:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
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