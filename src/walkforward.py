"""
Walk-forward analysis module for time series validation.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .features import FeatureEngineer
from .labels import LabelConstructor
from .model import LightGBMModel, ModelTrainer
from .backtest import Backtester
from .utils import Config, ensure_directory, save_parquet


class WalkForwardAnalysis:
    """Walk-forward analysis with rolling window validation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.feature_engineer = FeatureEngineer(config)
        self.label_constructor = LabelConstructor(config)
        self.model_trainer = ModelTrainer(config)
        self.backtester = Backtester(config)
        
        # Ensure artefacts directory exists
        ensure_directory(config.paths["artefacts"])
    
    def create_folds(self, df: pd.DataFrame, symbol: str, interval: str) -> List[Dict]:
        """Create rolling folds for walk-forward analysis."""
        self.logger.info(f"Creating folds for {symbol} {interval}")
        
        # Get fold sizes from config
        training_bars = self.config.walkforward["training_bars"][interval]
        test_bars = self.config.walkforward["test_bars"][interval]
        
        folds = []
        total_samples = len(df)
        
        # Calculate number of folds
        remaining_samples = total_samples - training_bars
        num_folds = remaining_samples // test_bars
        
        self.logger.info(f"Creating {num_folds} folds with {training_bars} training bars and {test_bars} test bars")
        
        for fold in range(num_folds):
            # Calculate indices
            train_start = fold * test_bars
            train_end = train_start + training_bars
            test_start = train_end
            test_end = test_start + test_bars
            
            # Ensure we don't exceed data bounds
            if test_end > total_samples:
                break
            
            fold_data = {
                'fold': fold,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_data': df.iloc[train_start:train_end],
                'test_data': df.iloc[test_start:test_end]
            }
            
            folds.append(fold_data)
        
        self.logger.info(f"Created {len(folds)} folds")
        return folds
    
    def train_and_evaluate_fold(self, fold_data: Dict, label_column: str) -> Dict:
        """Train and evaluate model for a single fold."""
        fold_num = fold_data['fold']
        train_data = fold_data['train_data']
        test_data = fold_data['test_data']
        
        self.logger.info(f"Processing fold {fold_num}")
        
        try:
            # Prepare features for training data
            X_train, y_train = self.prepare_features_and_labels(train_data, label_column)
            
            # Prepare features for test data
            X_test, y_test = self.prepare_features_and_labels(test_data, label_column)
            
            # Determine task type
            task_type = "classification" if "class" in label_column else "regression"
            
            # Train model
            model = LightGBMModel(self.config)
            model.train_model(X_train, y_train, X_test, y_test, task_type)
            
            # Evaluate model
            val_metrics = model.evaluate_model(X_test, y_test, task_type)
            
            # Get feature importance
            feature_importance = model.get_feature_importance()
            
            # Run backtest on test data
            backtest_results = self.backtester.run_backtest_with_model(test_data, model, label_column)
            
            # Make predictions for test data
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1] if task_type == "classification" else None
            
            results = {
                'fold': fold_num,
                'model': model,
                'val_metrics': val_metrics,
                'feature_importance': feature_importance,
                'backtest_results': backtest_results,
                'predictions': predictions,
                'probabilities': probabilities,
                'y_true': y_test.values,
                'train_size': len(train_data),
                'test_size': len(test_data)
            }
            
            self.logger.info(f"Completed fold {fold_num}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in fold {fold_num}: {e}")
            return None
    
    def prepare_features_and_labels(self, df: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels from DataFrame."""
        # Remove timestamp and label columns from features
        exclude_columns = ['timestamp', label_column]
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Also exclude return columns (they contain future information)
        feature_columns = [col for col in feature_columns if not col.startswith('return_')]
        
        X = df[feature_columns]
        y = df[label_column]
        
        return X, y
    
    def run_walkforward_analysis(self, df: pd.DataFrame, symbol: str, interval: str, 
                                label_column: str) -> Dict:
        """Run complete walk-forward analysis."""
        self.logger.info(f"Starting walk-forward analysis for {symbol} {interval} {label_column}")
        
        # Create folds
        folds = self.create_folds(df, symbol, interval)
        
        if not folds:
            self.logger.error("No folds created")
            return {}
        
        # Process each fold
        fold_results = []
        
        for fold_data in tqdm(folds, desc=f"Processing folds for {symbol} {interval}"):
            result = self.train_and_evaluate_fold(fold_data, label_column)
            if result is not None:
                fold_results.append(result)
        
        if not fold_results:
            self.logger.error("No successful fold results")
            return {}
        
        # Aggregate results
        aggregated_results = self.aggregate_fold_results(fold_results)
        
        # Save results
        self.save_walkforward_results(aggregated_results, symbol, interval, label_column)
        
        self.logger.info(f"Completed walk-forward analysis: {len(fold_results)} successful folds")
        
        return aggregated_results
    
    def aggregate_fold_results(self, fold_results: List[Dict]) -> Dict:
        """Aggregate results from all folds."""
        if not fold_results:
            return {}
        
        # Aggregate metrics
        all_val_metrics = []
        all_backtest_metrics = []
        all_feature_importances = []
        all_predictions = []
        all_probabilities = []
        all_y_true = []
        
        for result in fold_results:
            all_val_metrics.append(result['val_metrics'])
            all_backtest_metrics.append(result['backtest_results']['performance'])
            all_feature_importances.append(result['feature_importance'])
            all_predictions.extend(result['predictions'])
            if result['probabilities'] is not None:
                all_probabilities.extend(result['probabilities'])
            all_y_true.extend(result['y_true'])
        
        # Calculate average metrics
        avg_val_metrics = {}
        if all_val_metrics:
            for key in all_val_metrics[0].keys():
                if isinstance(all_val_metrics[0][key], (int, float)):
                    values = [metrics[key] for metrics in all_val_metrics if key in metrics]
                    avg_val_metrics[key] = np.mean(values)
        
        avg_backtest_metrics = {}
        if all_backtest_metrics:
            for key in all_backtest_metrics[0].keys():
                if isinstance(all_backtest_metrics[0][key], (int, float)):
                    values = [metrics[key] for metrics in all_backtest_metrics if key in metrics]
                    avg_backtest_metrics[key] = np.mean(values)
        
        # Aggregate feature importance
        if all_feature_importances:
            feature_importance_df = pd.concat(all_feature_importances, ignore_index=True)
            aggregated_importance = feature_importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
        else:
            aggregated_importance = pd.Series()
        
        return {
            'num_folds': len(fold_results),
            'avg_val_metrics': avg_val_metrics,
            'avg_backtest_metrics': avg_backtest_metrics,
            'feature_importance': aggregated_importance,
            'all_predictions': all_predictions,
            'all_probabilities': all_probabilities,
            'all_y_true': all_y_true,
            'fold_results': fold_results
        }
    
    def save_walkforward_results(self, results: Dict, symbol: str, interval: str, label_column: str) -> None:
        """Save walk-forward results to files."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{self.config.paths['artefacts']}/walkforward_{timestamp}"
        ensure_directory(output_dir)
        
        # Save aggregated metrics
        metrics_file = f"{output_dir}/{symbol}_{interval}_{label_column}_metrics.csv"
        metrics_df = pd.DataFrame([results['avg_val_metrics']])
        metrics_df.to_csv(metrics_file, index=False)
        
        # Save backtest metrics
        backtest_file = f"{output_dir}/{symbol}_{interval}_{label_column}_backtest.csv"
        backtest_df = pd.DataFrame([results['avg_backtest_metrics']])
        backtest_df.to_csv(backtest_file, index=False)
        
        # Save feature importance
        importance_file = f"{output_dir}/{symbol}_{interval}_{label_column}_importance.csv"
        results['feature_importance'].to_csv(importance_file)
        
        # Save predictions
        predictions_file = f"{output_dir}/{symbol}_{interval}_{label_column}_predictions.csv"
        predictions_df = pd.DataFrame({
            'y_true': results['all_y_true'],
            'predictions': results['all_predictions']
        })
        if results['all_probabilities']:
            predictions_df['probabilities'] = results['all_probabilities']
        predictions_df.to_csv(predictions_file, index=False)
        
        self.logger.info(f"Saved walk-forward results to {output_dir}")
    
    def run_all_analyses(self) -> Dict:
        """Run walk-forward analysis for all symbols, intervals, and horizons."""
        from .utils import load_parquet
        from pathlib import Path
        
        self.logger.info("Starting walk-forward analysis for all combinations")
        
        # Load gold data
        gold_dir = f"{self.config.paths['data']}/gold"
        gold_path = Path(gold_dir)
        
        if not gold_path.exists():
            self.logger.error("Gold data directory does not exist")
            return {}
        
        gold_files = list(gold_path.glob("*.parquet"))
        
        if not gold_files:
            self.logger.error("No gold data files found")
            return {}
        
        all_results = {}
        
        for gold_file in gold_files:
            try:
                # Parse symbol and interval from filename
                filename = gold_file.stem
                parts = filename.split('_')
                if len(parts) >= 2:
                    symbol = parts[0]
                    interval = parts[1]
                    
                    self.logger.info(f"Processing {symbol} {interval}")
                    
                    # Load data
                    df = load_parquet(str(gold_file))
                    
                    # Find label columns
                    label_columns = [col for col in df.columns if col.startswith('label_')]
                    
                    for label_column in label_columns:
                        try:
                            results = self.run_walkforward_analysis(df, symbol, interval, label_column)
                            key = f"{symbol}_{interval}_{label_column}"
                            all_results[key] = results
                            
                        except Exception as e:
                            self.logger.error(f"Error processing {symbol} {interval} {label_column}: {e}")
                            continue
                    
                else:
                    self.logger.warning(f"Could not parse symbol and interval from {filename}")
                    
            except Exception as e:
                self.logger.error(f"Error processing {gold_file}: {e}")
                continue
        
        self.logger.info(f"Completed walk-forward analysis for {len(all_results)} combinations")
        return all_results


def main():
    """Main function for walk-forward analysis."""
    from .utils import load_config, setup_logging, set_deterministic_seed
    
    # Load configuration
    config = load_config("config/settings.yaml")
    
    # Setup logging
    logger = setup_logging(config)
    
    # Set deterministic seed
    set_deterministic_seed(config.app["seed"])
    
    # Run walk-forward analysis
    walkforward = WalkForwardAnalysis(config)
    results = walkforward.run_all_analyses()
    
    # Print summary
    if results:
        print(f"Completed walk-forward analysis for {len(results)} combinations")
        for key, result in results.items():
            if result:
                print(f"{key}: {result['num_folds']} folds, Avg Accuracy: {result['avg_val_metrics'].get('accuracy', 0):.3f}")


if __name__ == "__main__":
    main()