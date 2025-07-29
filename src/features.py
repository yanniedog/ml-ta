"""
Feature engineering module for technical analysis with proper train/test separation.
"""
import logging
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

from .indicators import TechnicalIndicators
from .utils import Config, ensure_directory, save_parquet


class FeaturePipeline:
    """Feature pipeline with proper train/test separation to prevent data leakage."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.indicators = TechnicalIndicators(config)
        self.scaler = RobustScaler()
        self.is_scaler_fitted = False
        self.feature_columns = None
        self.fitted_on_data = None
        
        # Ensure silver directory exists
        ensure_directory(f"{config.paths['data']}/silver")
    
    def fit(self, X_train: pd.DataFrame) -> 'FeaturePipeline':
        """Fit the feature pipeline on training data only."""
        self.logger.info("Fitting feature pipeline on training data")
        
        # Store training data for reference
        self.fitted_on_data = X_train.copy()
        
        # Get feature columns (exclude timestamp and labels)
        exclude_columns = ['timestamp']
        label_columns = [col for col in X_train.columns if col.startswith('label_')]
        return_columns = [col for col in X_train.columns if col.startswith('return_')]
        exclude_columns.extend(label_columns)
        exclude_columns.extend(return_columns)
        
        # Also exclude future-looking columns
        future_columns = [col for col in X_train.columns if col.startswith('future_')]
        exclude_columns.extend(future_columns)
        
        self.feature_columns = [col for col in X_train.columns if col not in exclude_columns]
        
        # Fit scaler only on training features
        X_train_features = X_train[self.feature_columns].copy()
        
        # Handle infinity and extreme values
        X_train_features = self._clean_data(X_train_features)
        
        # Fit scaler
        self.scaler.fit(X_train_features)
        self.is_scaler_fitted = True
        
        self.logger.info(f"Fitted scaler on {len(self.feature_columns)} features")
        return self
    
    def save_scaler_state(self, filepath: str) -> None:
        """Save scaler state to file."""
        if not self.is_scaler_fitted:
            raise ValueError("Scaler not fitted yet")
        
        import joblib
        scaler_state = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'is_scaler_fitted': self.is_scaler_fitted
        }
        
        try:
            joblib.dump(scaler_state, filepath)
            self.logger.info(f"Scaler state saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving scaler state: {e}")
    
    def load_scaler_state(self, filepath: str) -> None:
        """Load scaler state from file."""
        import joblib
        
        try:
            scaler_state = joblib.load(filepath)
            self.scaler = scaler_state['scaler']
            self.feature_columns = scaler_state['feature_columns']
            self.is_scaler_fitted = scaler_state['is_scaler_fitted']
            self.logger.info(f"Scaler state loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading scaler state: {e}")
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler."""
        if not self.is_scaler_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        self.logger.info("Transforming features")
        
        # Ensure we have the same features as training
        if self.feature_columns is None:
            raise ValueError("Feature columns not set. Call fit() first.")
        
        # Get features
        X_features = X[self.feature_columns].copy()
        
        # Clean data
        X_features = self._clean_data(X_features)
        
        # Transform
        X_scaled = self.scaler.transform(X_features)
        X_transformed = pd.DataFrame(X_scaled, columns=self.feature_columns, index=X.index)
        
        # Add back non-feature columns
        for col in X.columns:
            if col not in self.feature_columns:
                X_transformed[col] = X[col]
        
        self.logger.info(f"Transformed {len(self.feature_columns)} features")
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step (for training data only)."""
        return self.fit(X).transform(X)
    
    def _clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling infinity and extreme values."""
        X_clean = X.copy()
        
        # Replace infinity values
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with median
        X_clean = X_clean.fillna(X_clean.median())
        
        # Clip extreme values
        for col in X_clean.columns:
            if X_clean[col].dtype in ['float64', 'float32']:
                Q1 = X_clean[col].quantile(0.001)
                Q3 = X_clean[col].quantile(0.999)
                X_clean[col] = X_clean[col].clip(Q1, Q3)
                
                # Additional check for very large values
                if X_clean[col].abs().max() > 1e6:
                    X_clean[col] = X_clean[col].clip(-1e6, 1e6)
        
        return X_clean


class FeatureEngineer:
    """Feature engineering for technical analysis with proper data leakage prevention."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.indicators = TechnicalIndicators(config)
        self.feature_pipeline = FeaturePipeline(config)
        self.is_pipeline_fitted = False
        
        # Ensure silver directory exists
        ensure_directory(f"{config.paths['data']}/silver")
    
    def add_regime_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime flags based on technical indicators."""
        self.logger.info("Adding regime flags")
        
        result_df = df.copy()
        
        # ADX regime (trend strength)
        if 'adx_adx' in df.columns:
            result_df['adx_strong_trend'] = (df['adx_adx'] > 25).astype(int)
            result_df['adx_weak_trend'] = (df['adx_adx'] < 20).astype(int)
        
        # ATR regime (volatility)
        if 'atr' in df.columns:
            atr_mean = df['atr'].rolling(window=50).mean()
            result_df['atr_high_vol'] = (df['atr'] > atr_mean * 1.5).astype(int)
            result_df['atr_low_vol'] = (df['atr'] < atr_mean * 0.5).astype(int)
        
        # Bollinger Bands regime
        if 'bb_bandwidth' in df.columns:
            bb_mean = df['bb_bandwidth'].rolling(window=50).mean()
            result_df['bb_squeeze'] = (df['bb_bandwidth'] < bb_mean * 0.5).astype(int)
            result_df['bb_expansion'] = (df['bb_bandwidth'] > bb_mean * 1.5).astype(int)
        
        # RSI regime
        if 'rsi' in df.columns:
            result_df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            result_df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
            result_df['rsi_neutral'] = ((df['rsi'] >= 30) & (df['rsi'] <= 70)).astype(int)
        
        # MACD regime
        if 'macd_macd' in df.columns and 'macd_signal' in df.columns:
            result_df['macd_bullish'] = (df['macd_macd'] > df['macd_signal']).astype(int)
            result_df['macd_bearish'] = (df['macd_macd'] < df['macd_signal']).astype(int)
        
        # Stochastic regime
        if 'stoch_k_percent' in df.columns:
            result_df['stoch_oversold'] = (df['stoch_k_percent'] < 20).astype(int)
            result_df['stoch_overbought'] = (df['stoch_k_percent'] > 80).astype(int)
        
        # Volume regime
        if 'volume' in df.columns:
            volume_mean = df['volume'].rolling(window=50).mean()
            result_df['volume_high'] = (df['volume'] > volume_mean * 1.5).astype(int)
            result_df['volume_low'] = (df['volume'] < volume_mean * 0.5).astype(int)
        
        # Price regime
        if 'close' in df.columns:
            sma_50 = df['close'].rolling(window=50).mean()
            sma_200 = df['close'].rolling(window=200).mean()
            result_df['price_above_sma50'] = (df['close'] > sma_50).astype(int)
            result_df['price_above_sma200'] = (df['close'] > sma_200).astype(int)
            result_df['golden_cross'] = ((sma_50 > sma_200) & (sma_50.shift(1) <= sma_200.shift(1))).astype(int)
            result_df['death_cross'] = ((sma_50 < sma_200) & (sma_50.shift(1) >= sma_200.shift(1))).astype(int)
        
        self.logger.info(f"Added {len(result_df.columns) - len(df.columns)} regime flags")
        return result_df
    
    def add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features."""
        self.logger.info("Adding lagged features")
        
        result_df = df.copy()
        lags = self.config.features["lags"]
        
        # Add lags for key price and volume features
        key_features = ['close', 'volume', 'high', 'low', 'open']
        
        for feature in key_features:
            if feature in df.columns:
                for lag in lags:
                    result_df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        # Add lags for technical indicators
        indicator_features = ['rsi', 'macd_macd', 'bb_upper', 'bb_lower', 'stoch_k_percent', 'atr']
        
        for feature in indicator_features:
            if feature in df.columns:
                for lag in lags:
                    result_df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        self.logger.info(f"Added {len(result_df.columns) - len(df.columns)} lagged features")
        return result_df
    
    def add_rolling_z_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling z-scores for key features."""
        self.logger.info("Adding rolling z-scores")
        
        result_df = df.copy()
        windows = self.config.features["z_score_windows"]
        
        # Calculate z-scores for key features
        key_features = ['close', 'volume', 'rsi', 'macd_macd']
        
        for feature in key_features:
            if feature in df.columns:
                for window in windows:
                    rolling_mean = df[feature].rolling(window=window).mean()
                    rolling_std = df[feature].rolling(window=window).std()
                    z_score = (df[feature] - rolling_mean) / rolling_std
                    result_df[f'{feature}_zscore_{window}'] = z_score
        
        self.logger.info(f"Added {len(result_df.columns) - len(df.columns)} rolling z-scores")
        return result_df
    
    def add_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feature interactions."""
        self.logger.info("Adding feature interactions")
        
        result_df = df.copy()
        
        # Price-volume interactions
        if 'close' in df.columns and 'volume' in df.columns:
            result_df['price_volume'] = df['close'] * df['volume']
            result_df['price_volume_ratio'] = df['close'] / df['volume']
        
        # Technical indicator interactions
        if 'rsi' in df.columns and 'macd_macd' in df.columns:
            result_df['rsi_macd'] = df['rsi'] * df['macd_macd']
        
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            result_df['bb_range'] = df['bb_upper'] - df['bb_lower']
            result_df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volatility interactions
        if 'atr' in df.columns and 'close' in df.columns:
            result_df['atr_price_ratio'] = df['atr'] / df['close']
        
        self.logger.info(f"Added {len(result_df.columns) - len(df.columns)} feature interactions")
        return result_df
    
    def build_feature_matrix(self, df: pd.DataFrame, fit_pipeline: bool = True, 
                           training_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Build feature matrix with proper train/test separation."""
        self.logger.info("Building feature matrix")
        
        # Check for empty DataFrame
        if df.empty:
            raise ValueError("Cannot build feature matrix from empty DataFrame")
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")
        
        # Calculate technical indicators
        df_with_indicators = self.indicators.calculate_all_indicators(df)
        
        # Add regime flags
        if self.config.features.get("regime_flags", True):
            df_with_indicators = self.add_regime_flags(df_with_indicators)
        
        # Add lagged features
        df_with_indicators = self.add_lags(df_with_indicators)
        
        # Add rolling z-scores
        df_with_indicators = self.add_rolling_z_scores(df_with_indicators)
        
        # Add feature interactions
        if self.config.features.get("interactions", True):
            df_with_indicators = self.add_interactions(df_with_indicators)
        
        # Remove any datetime columns that might have been added (but preserve timestamp)
        datetime_columns = df_with_indicators.select_dtypes(include=['datetime64']).columns
        # Only remove datetime columns that are not 'timestamp'
        columns_to_remove = [col for col in datetime_columns if col != 'timestamp']
        if len(columns_to_remove) > 0:
            self.logger.warning(f"Removing datetime columns: {columns_to_remove}")
            df_with_indicators = df_with_indicators.drop(columns=columns_to_remove)
        
        # Ensure only numeric columns are included in the final feature matrix
        # Exclude timestamp and any non-numeric columns
        exclude_columns = ['timestamp']
        label_columns = [col for col in df_with_indicators.columns if col.startswith('label_')]
        return_columns = [col for col in df_with_indicators.columns if col.startswith('return_')]
        exclude_columns.extend(label_columns)
        exclude_columns.extend(return_columns)
        
        # Get only numeric columns for features
        numeric_columns = df_with_indicators.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # Create final feature matrix with only numeric features (NO timestamp)
        final_df = df_with_indicators[feature_columns].copy()
        
        # Store feature columns in pipeline if fitting
        if fit_pipeline:
            self.feature_pipeline.feature_columns = feature_columns.copy()
            self.logger.info(f"Stored {len(feature_columns)} feature columns in pipeline")
        
        # Apply pipeline transformation
        if fit_pipeline:
            self.logger.info("Fitting feature pipeline on training data")
            final_df = self.feature_pipeline.fit_transform(final_df)
            self.is_pipeline_fitted = True
        else:
            if self.is_pipeline_fitted:
                self.logger.info("Transforming features")
                final_df = self.feature_pipeline.transform(final_df)
            else:
                self.logger.warning("Pipeline not fitted, returning untransformed features")
        
        # Clean up any remaining NaN values
        final_df = self.drop_nan_rows(final_df)
        
        self.logger.info(f"Final feature matrix shape: {final_df.shape}")
        return final_df
    
    def build_live_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build feature matrix for live prediction (no fitting)."""
        if not self.is_pipeline_fitted:
            raise ValueError("Pipeline must be fitted before live prediction")
        
        return self.build_feature_matrix(df, fit_pipeline=False)
    
    def ensure_feature_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure feature columns match the fitted pipeline."""
        if not self.is_pipeline_fitted:
            self.logger.warning("Pipeline not fitted, returning original DataFrame")
            return df
        
        if self.feature_pipeline.feature_columns is None:
            self.logger.warning("Feature columns not set in pipeline, returning original DataFrame")
            return df
        
        # Get the expected feature columns from the fitted pipeline
        expected_columns = self.feature_pipeline.feature_columns
        
        # Check if all expected columns are present
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            self.logger.error(f"Missing expected feature columns: {missing_columns}")
            # Add missing columns with zeros
            for col in missing_columns:
                df[col] = 0.0
        
        # Remove extra columns that are not in the expected set
        extra_columns = [col for col in df.columns if col not in expected_columns]
        if extra_columns:
            self.logger.warning(f"Removing extra columns: {extra_columns}")
            df = df[expected_columns]
        
        # Ensure the same order as during training
        df = df[expected_columns]
        
        self.logger.info(f"Feature consistency ensured: {len(df.columns)} features")
        return df
    
    def get_fitted_pipeline(self) -> FeaturePipeline:
        """Get the fitted feature pipeline."""
        if not self.is_pipeline_fitted:
            raise ValueError("Pipeline not fitted yet")
        return self.feature_pipeline
    
    def is_pipeline_ready(self) -> bool:
        """Check if pipeline is ready for prediction."""
        return self.is_pipeline_fitted and self.feature_pipeline.is_scaler_fitted
    
    def save_pipeline_state(self, filepath: str) -> None:
        """Save pipeline state for later use."""
        if not self.is_pipeline_fitted:
            raise ValueError("Pipeline not fitted yet")
        
        import joblib
        pipeline_data = {
            'feature_pipeline': self.feature_pipeline,
            'is_pipeline_fitted': self.is_pipeline_fitted
        }
        joblib.dump(pipeline_data, filepath)
        self.logger.info(f"Saved pipeline state to {filepath}")
    
    def load_pipeline_state(self, filepath: str) -> None:
        """Load pipeline state from file."""
        import joblib
        pipeline_data = joblib.load(filepath)
        
        self.feature_pipeline = pipeline_data['feature_pipeline']
        self.is_pipeline_fitted = pipeline_data['is_pipeline_fitted']
        
        self.logger.info(f"Loaded pipeline state from {filepath}")
    
    def get_pipeline_info(self) -> Dict:
        """Get information about the fitted pipeline."""
        if not self.is_pipeline_fitted:
            return {'is_fitted': False}
        
        return {
            'is_fitted': True,
            'feature_columns_count': len(self.feature_pipeline.feature_columns),
            'scaler_fitted': self.feature_pipeline.is_scaler_fitted,
            'feature_columns': self.feature_pipeline.feature_columns
        }
    
    def get_scaler_info(self) -> Dict:
        """Get information about the fitted scaler (alias for get_pipeline_info)."""
        return self.get_pipeline_info()
    
    def drop_nan_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with NaN values and log the percentage."""
        initial_rows = len(df)
        
        # Check which columns have NaN values
        nan_counts = df.isnull().sum()
        columns_with_nans = nan_counts[nan_counts > 0]
        
        if len(columns_with_nans) > 0:
            self.logger.warning(f"Columns with NaN values: {dict(columns_with_nans)}")
            
            # For test data, we should be more lenient with NaN values
            # Only drop rows where more than 50% of features are NaN
            feature_columns = [col for col in df.columns if col not in ['timestamp']]
            label_columns = [col for col in df.columns if col.startswith('label_')]
            return_columns = [col for col in df.columns if col.startswith('return_')]
            
            # Exclude label and return columns from NaN check
            check_columns = [col for col in feature_columns if col not in label_columns + return_columns]
            
            if len(check_columns) > 0:
                # Calculate percentage of NaN values per row
                nan_percentage = df[check_columns].isnull().sum(axis=1) / len(check_columns)
                
                # Keep rows where less than 50% of features are NaN
                df_clean = df[nan_percentage < 0.5]
                
                # Fill remaining NaN values with 0 for numeric columns
                numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
                df_clean[numeric_columns] = df_clean[numeric_columns].fillna(0)
                
                dropped_rows = initial_rows - len(df_clean)
                dropped_percentage = (dropped_rows / initial_rows) * 100 if initial_rows > 0 else 0
                
                self.logger.info(f"Dropped {dropped_rows} rows with excessive NaN values ({dropped_percentage:.1f}%)")
                self.logger.info(f"Remaining rows: {len(df_clean)}")
                
                return df_clean
            else:
                # No feature columns to check, return original
                return df
        else:
            # No NaN values found
            return df
    
    def ensure_consistent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has consistent features with training data."""
        if not self.is_pipeline_fitted:
            raise ValueError("Pipeline not fitted yet")
        
        if self.feature_pipeline.feature_columns is None:
            raise ValueError("Feature columns not set")
        
        # Add missing columns with zeros
        for col in self.feature_pipeline.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Remove extra columns
        columns_to_keep = self.feature_pipeline.feature_columns + ['timestamp']
        label_columns = [col for col in df.columns if col.startswith('label_')]
        return_columns = [col for col in df.columns if col.startswith('return_')]
        columns_to_keep.extend(label_columns)
        columns_to_keep.extend(return_columns)
        
        return df[columns_to_keep]
    
    def save_silver_data(self, df: pd.DataFrame, symbol: str, interval: str) -> str:
        """Save processed data to silver layer."""
        filename = f"{symbol}_{interval}_silver.parquet"
        filepath = f"{self.config.paths['data']}/silver/{filename}"
        
        save_parquet(df, filepath)
        self.logger.info(f"Saved silver data: {filepath}")
        return filepath
    
    def process_all_data(self) -> None:
        """Process all data in the bronze layer."""
        from pathlib import Path
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        
        from src.utils import load_config, setup_logging
        
        bronze_path = f"{self.config.paths['data']}/bronze"
        bronze_files = list(Path(bronze_path).glob("*.parquet"))
        
        for file_path in bronze_files:
            try:
                # Extract symbol and interval from filename
                filename = file_path.stem
                parts = filename.split('_')
                symbol = parts[0]
                interval = parts[1]
                
                # Load bronze data
                df = pd.read_parquet(file_path)
                
                # Process data
                processed_df = self.build_feature_matrix(df, fit_pipeline=True)
                
                # Save to silver layer
                self.save_silver_data(processed_df, symbol, interval)
                
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")


def main():
    """Main function for feature engineering."""
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from src.utils import load_config, setup_logging
    
    # Load configuration
    config = load_config("config/settings.yaml")
    setup_logging(config)
    
    # Create feature engineer
    feature_engineer = FeatureEngineer(config)
    
    # Process all data
    feature_engineer.process_all_data()


if __name__ == "__main__":
    main()