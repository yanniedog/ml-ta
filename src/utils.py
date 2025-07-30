"""
Utility functions for technical analysis application.
"""
import functools
import hashlib
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel
from numba import jit
from sklearn.preprocessing import StandardScaler


class Config(BaseModel):
    """Configuration model for the application."""
    app: Dict[str, Any]
    data: Dict[str, Any]
    binance: Dict[str, Any]
    indicators: Dict[str, Any]
    model: Dict[str, Any]
    walkforward: Dict[str, Any]
    backtest: Dict[str, Any]
    features: Dict[str, Any]
    paths: Dict[str, str]
    logging: Dict[str, str]
    shap: Dict[str, Any]


def setup_logging(config: Config) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = Path(config.paths["logs"])
    log_dir.mkdir(exist_ok=True)
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, config.logging["level"].upper()))

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(config.logging["format"])

    # Create file handler
    file_handler = logging.FileHandler(log_dir / config.logging["file"])
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def set_deterministic_seed(seed: int) -> None:
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(**config_dict)


def save_config(config: Config, output_path: str) -> None:
    """Save configuration to YAML file."""
    with open(output_path, 'w') as f:
        yaml.dump(config.dict(), f, default_flow_style=False)


def get_config_hash(config: Config) -> str:
    """Generate hash of configuration for reproducibility."""
    config_str = json.dumps(config.dict(), sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def timing_decorator(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper


def save_parquet(df: pd.DataFrame, path: str, compression: str = "snappy") -> None:
    """Save DataFrame to parquet file with compression."""
    df.to_parquet(path, compression=compression, index=False)


def load_parquet(path: str) -> pd.DataFrame:
    """Load DataFrame from parquet file."""
    return pd.read_parquet(path)


def ensure_directory(path: str) -> None:
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def format_timestamp() -> str:
    """Format current timestamp for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate returns for given periods."""
    return prices.pct_change(periods=periods)


def calculate_log_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate log returns for given periods."""
    return np.log(prices / prices.shift(periods))


def calculate_volatility(returns: pd.Series, window: int = 252) -> pd.Series:
    """Calculate rolling volatility."""
    return returns.rolling(window=window).std() * np.sqrt(252)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio."""
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown."""
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return drawdown.min()


def calculate_calmar_ratio(returns: pd.Series, max_drawdown: float) -> float:
    """Calculate Calmar ratio."""
    annual_return = returns.mean() * 252
    return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0


def calculate_profit_factor(trades: pd.DataFrame) -> float:
    """Calculate profit factor from trades."""
    if trades.empty:
        return 0.0
    
    winning_trades = trades[trades['pnl'] > 0]['pnl'].sum()
    losing_trades = abs(trades[trades['pnl'] < 0]['pnl'].sum())
    
    return winning_trades / losing_trades if losing_trades > 0 else float('inf')


def calculate_hit_rate(trades: pd.DataFrame) -> float:
    """Calculate hit rate from trades."""
    if trades.empty:
        return 0.0
    
    winning_trades = len(trades[trades['pnl'] > 0])
    total_trades = len(trades)
    
    return winning_trades / total_trades


def calculate_turnover(trades: pd.DataFrame, initial_capital: float) -> float:
    """Calculate turnover rate."""
    if trades.empty:
        return 0.0
    
    total_volume = trades['notional'].sum()
    return total_volume / initial_capital


def format_bps(value: float) -> str:
    """Format value in basis points."""
    return f"{value * 10000:.1f} bps"


def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value * 100:.2f}%"


def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Get logger from the class if it exists
        logger = None
        if args and hasattr(args[0], 'logger'):
            logger = args[0].logger
        
        execution_time = end_time - start_time
        if logger:
            logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        else:
            print(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    return wrapper


def cache_results(max_size: int = 128):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = (func.__name__, str(args), str(sorted(kwargs.items())))
            
            if key in cache:
                return cache[key]
            
            result = func(*args, **kwargs)
            
            # Simple LRU cache implementation
            if len(cache) >= max_size:
                # Remove oldest item (simple implementation)
                cache.pop(next(iter(cache)))
            
            cache[key] = result
            return result
        
        return wrapper
    return decorator


@jit(nopython=True)
def fast_rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling mean using Numba."""
    n = len(arr)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        result[i] = np.mean(arr[i - window + 1:i + 1])
    
    return result


@jit(nopython=True)
def fast_rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling standard deviation using Numba."""
    n = len(arr)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        result[i] = np.std(arr[i - window + 1:i + 1])
    
    return result


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage."""
    optimized_df = df.copy()
    
    for col in optimized_df.columns:
        col_type = optimized_df[col].dtype
        
        if col_type != 'object':
            c_min = optimized_df[col].min()
            c_max = optimized_df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    optimized_df[col] = optimized_df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    optimized_df[col] = optimized_df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    optimized_df[col] = optimized_df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    optimized_df[col] = optimized_df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    optimized_df[col] = optimized_df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    optimized_df[col] = optimized_df[col].astype(np.float32)
                else:
                    optimized_df[col] = optimized_df[col].astype(np.float64)
    
    return optimized_df


def parallel_process_dataframe(df: pd.DataFrame, func: Callable, 
                             n_jobs: int = -1, **kwargs) -> pd.DataFrame:
    """Process DataFrame in parallel using joblib."""
    try:
        from joblib import Parallel, delayed
    except ImportError:
        # Fallback to sequential processing
        return func(df, **kwargs)
    
    # Split DataFrame into chunks
    chunk_size = len(df) // n_jobs if n_jobs > 0 else len(df)
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Process chunks in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(func)(chunk, **kwargs) for chunk in chunks
    )
    
    # Combine results
    return pd.concat(results, ignore_index=True)


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data quality and return statistics."""
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'data_types': df.dtypes.to_dict(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'datetime_columns': len(df.select_dtypes(include=['datetime']).columns)
    }
    
    # Check for potential issues
    issues = []
    
    # Check for high missing values
    high_missing = {col: count for col, count in quality_report['missing_values'].items() 
                   if count > len(df) * 0.5}
    if high_missing:
        issues.append(f"High missing values in columns: {list(high_missing.keys())}")
    
    # Check for duplicate rows
    if quality_report['duplicate_rows'] > 0:
        issues.append(f"Found {quality_report['duplicate_rows']} duplicate rows")
    
    # Check for memory usage
    if quality_report['memory_usage_mb'] > 1000:  # 1GB
        issues.append(f"High memory usage: {quality_report['memory_usage_mb']:.2f} MB")
    
    quality_report['issues'] = issues
    quality_report['is_clean'] = len(issues) == 0
    
    return quality_report