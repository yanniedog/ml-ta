"""
Data fetching and processing module for Binance klines.
"""
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from .utils import Config, ensure_directory, save_parquet, load_parquet


class BinanceDataFetcher:
    """Fetches klines data from Binance API with rate limiting and error handling."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.base_url = config.binance["base_url"]
        self.endpoint = config.binance["klines_endpoint"]
        self.limit = config.binance["limit"]
        self.delay_ms = config.binance["request_delay_ms"]
        self.max_retries = config.binance["max_retries"]
        self.backoff_factor = config.binance["backoff_factor"]
        
        # Ensure data directories exist
        ensure_directory(config.paths["data"])
        ensure_directory(f"{config.paths['data']}/raw")
        ensure_directory(f"{config.paths['data']}/bronze")
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _make_request(self, symbol: str, interval: str, start_time: int, end_time: int) -> List:
        """Make API request with retry logic."""
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': self.limit
        }
        
        url = f"{self.base_url}{self.endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Rate limiting
            time.sleep(self.delay_ms / 1000)
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for {symbol} {interval}: {e}")
            raise
    
    def _convert_timestamp(self, timestamp_ms: int) -> datetime:
        """Convert millisecond timestamp to datetime."""
        return datetime.fromtimestamp(timestamp_ms / 1000)
    
    def _parse_klines(self, klines_data: List) -> pd.DataFrame:
        """Parse klines data into DataFrame."""
        if not klines_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(klines_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Select only OHLCV columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    def _check_data_gaps(self, df: pd.DataFrame, interval: str) -> None:
        """Check for data gaps and log them."""
        if df.empty:
            return
        
        # Calculate expected time difference based on interval
        interval_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        
        expected_diff = pd.Timedelta(minutes=interval_minutes.get(interval, 1))
        
        # Calculate actual time differences
        time_diffs = df['timestamp'].diff()
        
        # Find gaps (differences > 1.5x expected)
        gaps = time_diffs > (expected_diff * 1.5)
        
        if gaps.any():
            gap_indices = df[gaps].index
            self.logger.warning(f"Found {len(gap_indices)} data gaps in {interval} data")
            
            for idx in gap_indices[:5]:  # Log first 5 gaps
                gap_start = df.loc[idx - 1, 'timestamp'] if idx > 0 else None
                gap_end = df.loc[idx, 'timestamp']
                self.logger.warning(f"Gap from {gap_start} to {gap_end}")
    
    def _deduplicate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate timestamps."""
        initial_count = len(df)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        final_count = len(df)
        
        if initial_count != final_count:
            self.logger.info(f"Removed {initial_count - final_count} duplicate records")
        
        return df
    
    def _drop_incomplete_candles(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Drop incomplete final candles."""
        if df.empty:
            return df
        
        # For now, we'll keep all candles as Binance provides complete data
        # This could be enhanced to check for incomplete candles based on volume patterns
        return df
    
    def fetch_klines(self, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch klines data for given symbol and interval."""
        self.logger.info(f"Fetching {symbol} {interval} data from {start_date} to {end_date}")
        
        # Convert dates to timestamps
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)
        
        all_klines = []
        current_start = start_ts
        
        with tqdm(desc=f"Fetching {symbol} {interval}", unit="request") as pbar:
            while current_start < end_ts:
                current_end = min(current_start + (self.limit - 1) * self._get_interval_ms(interval), end_ts)
                
                try:
                    klines_data = self._make_request(symbol, interval, current_start, current_end)
                    all_klines.extend(klines_data)
                    
                    if len(klines_data) < self.limit:
                        break
                    
                    # Move to next batch
                    current_start = current_end + 1
                    pbar.update(1)
                    
                except Exception as e:
                    self.logger.error(f"Failed to fetch data: {e}")
                    break
        
        # Parse and process data
        df = self._parse_klines(all_klines)
        
        if not df.empty:
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Check for gaps
            self._check_data_gaps(df, interval)
            
            # Deduplicate
            df = self._deduplicate_data(df)
            
            # Drop incomplete candles
            df = self._drop_incomplete_candles(df, interval)
            
            self.logger.info(f"Fetched {len(df)} records for {symbol} {interval}")
        else:
            self.logger.warning(f"No data fetched for {symbol} {interval}")
        
        return df
    
    def _get_interval_ms(self, interval: str) -> int:
        """Get interval duration in milliseconds."""
        interval_ms = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        return interval_ms.get(interval, 60 * 1000)
    
    def save_raw_data(self, df: pd.DataFrame, symbol: str, interval: str) -> str:
        """Save raw data as JSON."""
        raw_dir = Path(self.config.paths["data"]) / "raw"
        filename = f"{symbol}_{interval}_raw.json"
        filepath = raw_dir / filename
        
        # Convert timestamp to string for JSON serialization
        df_json = df.copy()
        df_json['timestamp'] = df_json['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        with open(filepath, 'w') as f:
            json.dump(df_json.to_dict('records'), f, indent=2)
        
        self.logger.info(f"Saved raw data to {filepath}")
        return str(filepath)
    
    def save_bronze_data(self, df: pd.DataFrame, symbol: str, interval: str) -> str:
        """Save processed data as bronze parquet."""
        bronze_dir = Path(self.config.paths["data"]) / "bronze"
        filename = f"{symbol}_{interval}_bronze.parquet"
        filepath = bronze_dir / filename
        
        save_parquet(df, str(filepath))
        
        self.logger.info(f"Saved bronze data to {filepath}")
        return str(filepath)
    
    def fetch_all_data(self) -> None:
        """Fetch data for all symbols and intervals."""
        symbols = self.config.data["symbols"]
        intervals = self.config.data["intervals"]
        start_date = self.config.data["start_date"]
        end_date = self.config.data["end_date"]
        
        for symbol in symbols:
            for interval in intervals:
                try:
                    # Fetch data
                    df = self.fetch_klines(symbol, interval, start_date, end_date)
                    
                    if not df.empty:
                        # Save raw data
                        self.save_raw_data(df, symbol, interval)
                        
                        # Save bronze data
                        self.save_bronze_data(df, symbol, interval)
                    
                except Exception as e:
                    self.logger.error(f"Failed to fetch {symbol} {interval}: {e}")
                    continue


def main():
    """Main function for data fetching."""
    from .utils import load_config, setup_logging, set_deterministic_seed
    
    # Load configuration
    config = load_config("config/settings.yaml")
    
    # Setup logging
    logger = setup_logging(config)
    
    # Set deterministic seed
    set_deterministic_seed(config.app["seed"])
    
    # Fetch data
    fetcher = BinanceDataFetcher(config)
    fetcher.fetch_all_data()


if __name__ == "__main__":
    main()