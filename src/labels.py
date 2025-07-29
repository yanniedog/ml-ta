"""
Label construction module for technical analysis.
"""
import logging
from typing import Tuple

import numpy as np
import pandas as pd

from .utils import Config, ensure_directory, save_parquet


class LabelConstructor:
    """Constructs labels for classification and regression tasks."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Ensure gold directory exists
        ensure_directory(f"{config.paths['data']}/gold")
    
    def calculate_returns(self, prices: pd.Series, horizon: int) -> pd.Series:
        """Calculate returns for given horizon."""
        future_prices = prices.shift(-horizon)
        returns = (future_prices - prices) / prices
        return returns
    
    def create_classification_labels(self, returns: pd.Series, threshold_bps: float = 0.0) -> pd.Series:
        """Create binary classification labels."""
        threshold = threshold_bps / 10000  # Convert bps to decimal
        labels = (returns > threshold).astype(int)
        return labels
    
    def create_regression_labels(self, returns: pd.Series) -> pd.Series:
        """Create regression labels (raw returns)."""
        return returns
    
    def construct_labels(self, df: pd.DataFrame, horizon: int, 
                        task_type: str = "classification", threshold_bps: float = 0.0) -> pd.DataFrame:
        """Construct labels for given horizon and task type."""
        self.logger.info(f"Constructing {task_type} labels for horizon {horizon}")
        
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")
        
        # Calculate returns
        returns = self.calculate_returns(df['close'], horizon)
        
        # Create labels based on task type
        if task_type == "classification":
            labels = self.create_classification_labels(returns, threshold_bps)
            label_name = f"label_class_{horizon}"
        elif task_type == "regression":
            labels = self.create_regression_labels(returns)
            label_name = f"label_reg_{horizon}"
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Add labels to DataFrame
        result_df = df.copy()
        result_df[label_name] = labels
        result_df[f"return_{horizon}"] = returns
        
        # Drop rows where we don't have future data (last horizon rows)
        result_df = result_df.dropna(subset=[label_name])
        
        self.logger.info(f"Created {len(result_df)} labeled samples for horizon {horizon}")
        return result_df
    
    def construct_all_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construct labels for all horizons and task types."""
        self.logger.info("Constructing all labels")
        
        result_df = df.copy()
        horizons = self.config.data["horizons"]
        
        for horizon in horizons:
            # Classification labels
            df_class = self.construct_labels(df, horizon, "classification")
            result_df[f"label_class_{horizon}"] = df_class[f"label_class_{horizon}"]
            result_df[f"return_{horizon}"] = df_class[f"return_{horizon}"]
            
            # Regression labels
            df_reg = self.construct_labels(df, horizon, "regression")
            result_df[f"label_reg_{horizon}"] = df_reg[f"label_reg_{horizon}"]
        
        # Drop rows with any NaN labels
        label_columns = [col for col in result_df.columns if col.startswith('label_')]
        result_df = result_df.dropna(subset=label_columns)
        
        self.logger.info(f"Final labeled dataset shape: {result_df.shape}")
        return result_df
    
    def save_gold_data(self, df: pd.DataFrame, symbol: str, interval: str) -> str:
        """Save labeled data as gold parquet."""
        gold_dir = f"{self.config.paths['data']}/gold"
        filename = f"{symbol}_{interval}_gold.parquet"
        filepath = f"{gold_dir}/{filename}"
        
        save_parquet(df, filepath)
        
        self.logger.info(f"Saved gold data to {filepath}")
        return filepath
    
    def process_all_data(self) -> None:
        """Process all silver data into gold labels."""
        from .utils import load_parquet
        from pathlib import Path
        
        silver_dir = f"{self.config.paths['data']}/silver"
        silver_path = Path(silver_dir)
        
        if not silver_path.exists():
            self.logger.error("Silver data directory does not exist")
            return
        
        # Find all silver parquet files
        silver_files = list(silver_path.glob("*.parquet"))
        
        if not silver_files:
            self.logger.error("No silver parquet files found")
            return
        
        for silver_file in silver_files:
            try:
                # Parse symbol and interval from filename
                filename = silver_file.stem
                parts = filename.split('_')
                if len(parts) >= 2:
                    symbol = parts[0]
                    interval = parts[1]
                    
                    self.logger.info(f"Processing labels for {symbol} {interval}")
                    
                    # Load silver data
                    df = load_parquet(str(silver_file))
                    
                    # Construct labels
                    labeled_df = self.construct_all_labels(df)
                    
                    # Save gold data
                    self.save_gold_data(labeled_df, symbol, interval)
                    
                else:
                    self.logger.warning(f"Could not parse symbol and interval from {filename}")
                    
            except Exception as e:
                self.logger.error(f"Error processing {silver_file}: {e}")
                continue
    
    def get_label_distribution(self, df: pd.DataFrame) -> dict:
        """Get distribution of labels for analysis."""
        distribution = {}
        
        label_columns = [col for col in df.columns if col.startswith('label_')]
        
        for col in label_columns:
            if col in df.columns:
                value_counts = df[col].value_counts()
                distribution[col] = {
                    'total': len(df[col]),
                    'positive': value_counts.get(1, 0),
                    'negative': value_counts.get(0, 0),
                    'positive_rate': value_counts.get(1, 0) / len(df[col]) if len(df[col]) > 0 else 0
                }
        
        return distribution
    
    def validate_labels(self, df: pd.DataFrame) -> bool:
        """Validate that labels are properly constructed."""
        self.logger.info("Validating labels")
        
        # Check for required columns
        required_columns = ['timestamp', 'close']
        for col in required_columns:
            if col not in df.columns:
                self.logger.error(f"Missing required column: {col}")
                return False
        
        # Check for label columns
        label_columns = [col for col in df.columns if col.startswith('label_')]
        if not label_columns:
            self.logger.error("No label columns found")
            return False
        
        # Check for return columns
        return_columns = [col for col in df.columns if col.startswith('return_')]
        if not return_columns:
            self.logger.error("No return columns found")
            return False
        
        # Check for NaN values in labels
        for col in label_columns:
            if df[col].isna().any():
                self.logger.error(f"NaN values found in {col}")
                return False
        
        # Check label values
        for col in label_columns:
            unique_values = df[col].unique()
            if not all(val in [0, 1] for val in unique_values):
                self.logger.error(f"Invalid values in {col}: {unique_values}")
                return False
        
        self.logger.info("Label validation passed")
        return True


def main():
    """Main function for label construction."""
    from .utils import load_config, setup_logging, set_deterministic_seed
    
    # Load configuration
    config = load_config("config/settings.yaml")
    
    # Setup logging
    logger = setup_logging(config)
    
    # Set deterministic seed
    set_deterministic_seed(config.app["seed"])
    
    # Process labels
    constructor = LabelConstructor(config)
    constructor.process_all_data()


if __name__ == "__main__":
    main()