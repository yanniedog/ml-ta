#!/usr/bin/env python3
"""
Performance optimization module for ML trading system.
"""

import logging
import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from functools import wraps
import joblib
from numba import jit
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name: str) -> None:
        """Start timing a process."""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing and return duration."""
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            self.metrics[name] = duration
            del self.start_times[name]
            return duration
        return 0.0
    
    def get_memory_usage(self) -> Dict:
        """Get current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
    
    def get_metrics(self) -> Dict:
        """Get all performance metrics."""
        return {
            'timing': self.metrics,
            'memory': self.get_memory_usage()
        }


def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = PerformanceMonitor()
        func_name = func.__name__
        
        monitor.start_timer(func_name)
        result = func(*args, **kwargs)
        duration = monitor.end_timer(func_name)
        
        memory_usage = monitor.get_memory_usage()
        logger.info(f"{func_name} completed in {duration:.3f}s, memory: {memory_usage['rss_mb']:.1f}MB")
        
        return result
    return wrapper


class MemoryOptimizer:
    """Optimize DataFrame memory usage."""
    
    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by downcasting numeric types."""
        start_mem = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        
        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        reduction = (start_mem - end_mem) / start_mem * 100
        
        logger.info(f"Memory optimization: {start_mem:.2f}MB -> {end_mem:.2f}MB ({reduction:.1f}% reduction)")
        
        return df


class ParallelProcessor:
    """Parallel processing utilities."""
    
    def __init__(self, n_jobs: int = -1):
        self.n_jobs = n_jobs
    
    @performance_monitor
    def process_features_parallel(self, df: pd.DataFrame, feature_func: Callable, 
                                chunk_size: int = 1000) -> pd.DataFrame:
        """Process features in parallel using joblib."""
        # Split data into chunks
        chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
        
        # Process chunks in parallel
        results = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(feature_func)(chunk) for chunk in chunks
        )
        
        # Combine results
        return pd.concat(results, ignore_index=True)
    
    @performance_monitor
    def calculate_indicators_parallel(self, df: pd.DataFrame, indicator_funcs: List[Callable]) -> pd.DataFrame:
        """Calculate multiple indicators in parallel."""
        def calculate_chunk_indicators(chunk):
            result = chunk.copy()
            for func in indicator_funcs:
                result = func(result)
            return result
        
        return self.process_features_parallel(df, calculate_chunk_indicators)


class CacheManager:
    """Simple caching system for expensive computations."""
    
    def __init__(self):
        self.cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
    
    def get_cached_result(self, key: str, compute_func: Callable, *args, **kwargs):
        """Get cached result or compute if not available."""
        if key in self.cache:
            self.cache_stats['hits'] += 1
            return self.cache[key]
        
        self.cache_stats['misses'] += 1
        result = compute_func(*args, **kwargs)
        self.cache[key] = result
        return result
    
    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
        self.cache_stats = {'hits': 0, 'misses': 0}
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total if total > 0 else 0
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }


@jit(nopython=True)
def fast_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling mean calculation using Numba."""
    n = len(values)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        result[i] = np.mean(values[i - window + 1:i + 1])
    
    return result


@jit(nopython=True)
def fast_rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling standard deviation calculation using Numba."""
    n = len(values)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        result[i] = np.std(values[i - window + 1:i + 1])
    
    return result


class PerformanceOptimizer:
    """Main performance optimization class."""
    
    def __init__(self, config):
        self.config = config
        self.monitor = PerformanceMonitor()
        self.memory_optimizer = MemoryOptimizer()
        self.parallel_processor = ParallelProcessor()
        self.cache_manager = CacheManager()
        self.logger = logging.getLogger(__name__)
    
    def optimize_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all optimizations to a DataFrame."""
        self.monitor.start_timer('total_optimization')
        
        # Memory optimization
        df = self.memory_optimizer.optimize_dataframe_memory(df)
        
        # Log optimization results
        duration = self.monitor.end_timer('total_optimization')
        memory_usage = self.monitor.get_memory_usage()
        
        self.logger.info(f"Pipeline optimization completed in {duration:.3f}s")
        self.logger.info(f"Memory usage: {memory_usage['rss_mb']:.1f}MB")
        
        return df
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report."""
        return {
            'timing': self.monitor.get_metrics(),
            'cache': self.cache_manager.get_cache_stats(),
            'memory': self.monitor.get_memory_usage()
        }