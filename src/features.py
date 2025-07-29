"""
Feature engineering module for technical analysis.
"""
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .indicators import TechnicalIndicators
from .utils import Config, ensure_directory, save_parquet


class FeatureEngineer:
    """Feature engineering for technical analysis."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.indicators = TechnicalIndicators(config)
        self.scaler = RobustScaler()
        
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
        
        # Add lags for key technical indicators
        key_indicators = ['rsi', 'macd_macd', 'stoch_k_percent', 'williams_r', 'cci', 'roc']
        
        for indicator in key_indicators:
            if indicator in df.columns:
                for lag in lags:
                    result_df[f'{indicator}_lag_{lag}'] = df[indicator].shift(lag)
        
        self.logger.info(f"Added {len(result_df.columns) - len(df.columns)} lagged features")
        return result_df
    
    def add_rolling_z_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling z-scores for key features."""
        self.logger.info("Adding rolling z-scores")
        
        result_df = df.copy()
        z_score_windows = self.config.features["z_score_windows"]
        
        # Key features for z-scoring
        key_features = ['close', 'volume', 'rsi', 'macd_macd', 'cci', 'roc']
        
        for feature in key_features:
            if feature in df.columns:
                for window in z_score_windows:
                    rolling_mean = df[feature].rolling(window=window).mean()
                    rolling_std = df[feature].rolling(window=window).std()
                    z_score = (df[feature] - rolling_mean) / rolling_std
                    result_df[f'{feature}_zscore_{window}'] = z_score
        
        self.logger.info(f"Added {len(result_df.columns) - len(df.columns)} rolling z-scores")
        return result_df
    
    def add_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feature interactions."""
        if not self.config.features["interactions"]:
            return df
        
        self.logger.info("Adding feature interactions")
        
        result_df = df.copy()
        
        # Price-volume interactions
        if 'close' in df.columns and 'volume' in df.columns:
            result_df['price_volume'] = df['close'] * df['volume']
            result_df['price_volume_ratio'] = df['close'] / df['volume']
        
        # RSI-MACD interaction
        if 'rsi' in df.columns and 'macd_macd' in df.columns:
            result_df['rsi_macd'] = df['rsi'] * df['macd_macd']
        
        # Bollinger Bands interactions
        if 'close' in df.columns and 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            result_df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR-Volume interaction
        if 'atr' in df.columns and 'volume' in df.columns:
            result_df['atr_volume'] = df['atr'] * df['volume']
        
        # Stochastic-RSI interaction
        if 'stoch_k_percent' in df.columns and 'rsi' in df.columns:
            result_df['stoch_rsi'] = df['stoch_k_percent'] * df['rsi']
        
        self.logger.info(f"Added {len(result_df.columns) - len(df.columns)} feature interactions")
        return result_df
    
    def scale_features(self, df: pd.DataFrame, fit_on_training: bool = True, 
                      training_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Scale features using robust scaling."""
        self.logger.info("Scaling features")
        
        # Create a new scaler instance for this operation
        scaler = RobustScaler()
        
        # Select numeric columns for scaling
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col not in ['timestamp']]
        
        # Ensure unique columns and sort for consistency
        feature_columns = sorted(list(set(feature_columns)))
        
        # Use only columns that actually exist in the DataFrame
        available_columns = [col for col in feature_columns if col in df.columns]
        feature_columns = available_columns
        
        # Check for duplicate column names in the DataFrame
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        if duplicate_columns:
            self.logger.warning(f"Found duplicate columns: {duplicate_columns}")
        
        self.logger.info(f"Found {len(feature_columns)} features to scale")
        self.logger.info(f"DataFrame shape: {df.shape}")
        self.logger.info(f"Numeric columns: {len(numeric_columns)}")
        
        # Fill NaN values with 0 for scaling
        df_filled = df[feature_columns].fillna(0)
        
        # Replace infinite values with 0
        df_filled = df_filled.replace([np.inf, -np.inf], 0)
        
        self.logger.info(f"df_filled shape: {df_filled.shape}")
        self.logger.info(f"df_filled columns count: {len(df_filled.columns)}")
        self.logger.info(f"feature_columns count: {len(feature_columns)}")
        
        if fit_on_training and training_data is not None:
            # Fit scaler on training data
            training_features = training_data[feature_columns].fillna(0)
            training_features = training_features.replace([np.inf, -np.inf], 0)
            scaler.fit(training_features)
        elif fit_on_training:
            # Fit scaler on current data if no training data provided
            scaler.fit(df_filled)
        
        # Scale features
        scaled_features = scaler.transform(df_filled)
        
        self.logger.info(f"Scaled features shape: {scaled_features.shape}")
        self.logger.info(f"Feature columns length: {len(feature_columns)}")
        
        # Create DataFrame with the same columns and index
        scaled_df = pd.DataFrame(scaled_features, columns=feature_columns, index=df.index)
        
        # Combine with non-numeric columns
        result_df = df.copy()
        for col in feature_columns:
            result_df[col] = scaled_df[col]
        
        self.logger.info(f"Scaled {len(feature_columns)} features")
        return result_df
    
    def drop_nan_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with NaN values."""
        initial_count = len(df)
        result_df = df.dropna()
        final_count = len(result_df)
        
        dropped_count = initial_count - final_count
        self.logger.info(f"Dropped {dropped_count} rows with NaN values ({dropped_count/initial_count*100:.1f}%)")
        
        return result_df
    
    def build_feature_matrix(self, df: pd.DataFrame, fit_scaler: bool = True,
                           training_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Build complete feature matrix."""
        self.logger.info("Building feature matrix")
        
        # Calculate all technical indicators
        df_with_indicators = self.indicators.calculate_all_indicators(df)
        
        # Add regime flags
        df_with_regimes = self.add_regime_flags(df_with_indicators)
        
        # Add lags
        df_with_lags = self.add_lags(df_with_regimes)
        
        # Add rolling z-scores
        df_with_z_scores = self.add_rolling_z_scores(df_with_lags)
        
        # Add interactions
        df_with_interactions = self.add_interactions(df_with_z_scores)
        
        # Scale features
        df_scaled = self.scale_features(df_with_interactions, fit_scaler, training_data)
        
        # Drop NaN rows
        df_clean = self.drop_nan_rows(df_scaled)
        
        self.logger.info(f"Final feature matrix shape: {df_clean.shape}")
        return df_clean
    
    def save_silver_data(self, df: pd.DataFrame, symbol: str, interval: str) -> str:
        """Save processed data as silver parquet."""
        silver_dir = f"{self.config.paths['data']}/silver"
        filename = f"{symbol}_{interval}_silver.parquet"
        filepath = f"{silver_dir}/{filename}"
        
        save_parquet(df, filepath)
        
        self.logger.info(f"Saved silver data to {filepath}")
        return filepath
    
    def process_all_data(self) -> None:
        """Process all bronze data into silver features."""
        from .utils import load_parquet
        
        bronze_dir = f"{self.config.paths['data']}/bronze"
        bronze_path = Path(bronze_dir)
        
        if not bronze_path.exists():
            self.logger.error("Bronze data directory does not exist")
            return
        
        # Find all bronze parquet files
        bronze_files = list(bronze_path.glob("*.parquet"))
        
        if not bronze_files:
            self.logger.error("No bronze parquet files found")
            return
        
        for bronze_file in bronze_files:
            try:
                # Parse symbol and interval from filename
                filename = bronze_file.stem
                parts = filename.split('_')
                if len(parts) >= 2:
                    symbol = parts[0]
                    interval = parts[1]
                    
                    self.logger.info(f"Processing {symbol} {interval}")
                    
                    # Load bronze data
                    df = load_parquet(str(bronze_file))
                    
                    # Build feature matrix
                    feature_df = self.build_feature_matrix(df)
                    
                    # Save silver data
                    self.save_silver_data(feature_df, symbol, interval)
                    
                else:
                    self.logger.warning(f"Could not parse symbol and interval from {filename}")
                    
            except Exception as e:
                self.logger.error(f"Error processing {bronze_file}: {e}")
                continue


def main():
    """Main function for feature engineering."""
    from .utils import load_config, setup_logging, set_deterministic_seed
    
    # Load configuration
    config = load_config("config/settings.yaml")
    
    # Setup logging
    logger = setup_logging(config)
    
    # Set deterministic seed
    set_deterministic_seed(config.app["seed"])
    
    # Process features
    engineer = FeatureEngineer(config)
    engineer.process_all_data()


if __name__ == "__main__":
    main()