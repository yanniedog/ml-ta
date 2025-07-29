"""
Technical indicators module with leak-safe, vectorized implementations.
"""
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .utils import Config


class TechnicalIndicators:
    """Technical indicators implementation with leak-safe calculations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def sma(self, prices: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return prices.rolling(window=window).mean()
    
    def ema(self, prices: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average."""
        return prices.ewm(span=window).mean()
    
    def macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD (Moving Average Convergence Divergence)."""
        ema_fast = self.ema(prices, fast)
        ema_slow = self.ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
    
    def rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
        """Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return pd.DataFrame({
            'k_percent': k_percent,
            'd_percent': d_percent
        })
    
    def bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """Bollinger Bands."""
        sma = self.sma(prices, window)
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        bandwidth = (upper_band - lower_band) / sma
        percent_b = (prices - lower_band) / (upper_band - lower_band)
        
        return pd.DataFrame({
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band,
            'bandwidth': bandwidth,
            'percent_b': percent_b
        })
    
    def _parabolic_sar_core(self, high: np.ndarray, low: np.ndarray, 
                           acceleration: float, maximum: float) -> Tuple[np.ndarray, np.ndarray]:
        """Core Parabolic SAR calculation."""
        n = len(high)
        sar = np.full(n, np.nan)
        ep = np.full(n, np.nan)
        
        # Initialize
        sar[0] = low[0]
        ep[0] = high[0]
        af = acceleration
        long = True
        
        for i in range(1, n):
            if long:
                sar[i] = sar[i-1] + af * (ep[i-1] - sar[i-1])
                sar[i] = min(sar[i], low[i-1], low[i-2] if i > 1 else low[i-1])
                
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af = min(af + acceleration, maximum)
                else:
                    ep[i] = ep[i-1]
                
                if low[i] < sar[i]:
                    long = False
                    sar[i] = ep[i-1]
                    ep[i] = low[i]
                    af = acceleration
            else:
                sar[i] = sar[i-1] + af * (ep[i-1] - sar[i-1])
                sar[i] = max(sar[i], high[i-1], high[i-2] if i > 1 else high[i-1])
                
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af = min(af + acceleration, maximum)
                else:
                    ep[i] = ep[i-1]
                
                if high[i] > sar[i]:
                    long = True
                    sar[i] = ep[i-1]
                    ep[i] = high[i]
                    af = acceleration
        
        return sar, ep
    
    def parabolic_sar(self, high: pd.Series, low: pd.Series, 
                     acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
        """Parabolic SAR."""
        sar, ep = self._parabolic_sar_core(high.values, low.values, acceleration, maximum)
        return pd.Series(sar, index=high.index)
    
    def ichimoku(self, high: pd.Series, low: pd.Series, 
                 tenkan: int = 9, kijun: int = 26, senkou_span_b: int = 52) -> pd.DataFrame:
        """Ichimoku Cloud."""
        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=tenkan).max()
        tenkan_low = low.rolling(window=tenkan).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=kijun).max()
        kijun_low = low.rolling(window=kijun).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b_high = high.rolling(window=senkou_span_b).max()
        senkou_span_b_low = low.rolling(window=senkou_span_b).min()
        senkou_span_b = ((senkou_span_b_high + senkou_span_b_low) / 2).shift(kijun)
        
        # Chikou Span (Lagging Span)
        chikou_span = pd.Series(index=high.index, dtype=float)
        if len(high) > kijun:
            chikou_span.iloc[:-kijun] = high.iloc[kijun:].values
        
        return pd.DataFrame({
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        })
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    def cci(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """Commodity Channel Index."""
        typical_price = (high + low + close) / 3
        sma_tp = self.sma(typical_price, window)
        mean_deviation = abs(typical_price - sma_tp).rolling(window=window).mean()
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci
    
    def roc(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Rate of Change."""
        return ((prices - prices.shift(window)) / prices.shift(window)) * 100
    
    def williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R."""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
        return williams_r
    
    def keltner_channels(self, high: pd.Series, low: pd.Series, close: pd.Series,
                         ema_window: int = 20, atr_window: int = 10, multiplier: float = 2) -> pd.DataFrame:
        """Keltner Channels."""
        ema = self.ema(close, ema_window)
        atr = self.atr(high, low, close, atr_window)
        upper_channel = ema + (multiplier * atr)
        lower_channel = ema - (multiplier * atr)
        width = (upper_channel - lower_channel) / ema
        
        return pd.DataFrame({
            'upper': upper_channel,
            'middle': ema,
            'lower': lower_channel,
            'width': width
        })
    
    def _supertrend_core(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                         atr_window: int, factor: float) -> Tuple[np.ndarray, np.ndarray]:
        """Core Supertrend calculation."""
        n = len(high)
        supertrend = np.full(n, np.nan)
        direction = np.full(n, np.nan)
        
        # Calculate ATR
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.full(n, np.nan)
        for i in range(atr_window, n):
            atr[i] = np.mean(tr[i-atr_window+1:i+1])
        
        # Calculate Supertrend
        basic_upper = np.full(n, np.nan)
        basic_lower = np.full(n, np.nan)
        final_upper = np.full(n, np.nan)
        final_lower = np.full(n, np.nan)
        
        for i in range(atr_window, n):
            basic_upper[i] = (high[i] + low[i]) / 2 + factor * atr[i]
            basic_lower[i] = (high[i] + low[i]) / 2 - factor * atr[i]
            
            if i == atr_window:
                final_upper[i] = basic_upper[i]
                final_lower[i] = basic_lower[i]
            else:
                final_upper[i] = basic_upper[i] if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1] else final_upper[i-1]
                final_lower[i] = basic_lower[i] if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1] else final_lower[i-1]
        
        # Determine direction and supertrend
        for i in range(atr_window, n):
            if i == atr_window:
                direction[i] = 1
                supertrend[i] = final_lower[i]
            else:
                if direction[i-1] == 1 and close[i] <= final_upper[i]:
                    direction[i] = 1
                    supertrend[i] = final_lower[i]
                elif direction[i-1] == 1 and close[i] > final_upper[i]:
                    direction[i] = -1
                    supertrend[i] = final_upper[i]
                elif direction[i-1] == -1 and close[i] >= final_lower[i]:
                    direction[i] = -1
                    supertrend[i] = final_upper[i]
                elif direction[i-1] == -1 and close[i] < final_lower[i]:
                    direction[i] = 1
                    supertrend[i] = final_lower[i]
        
        return supertrend, direction
    
    def supertrend(self, high: pd.Series, low: pd.Series, close: pd.Series,
                   atr_window: int = 10, factor: float = 3) -> pd.DataFrame:
        """Supertrend indicator."""
        supertrend, direction = self._supertrend_core(high.values, low.values, close.values, atr_window, factor)
        return pd.DataFrame({
            'supertrend': pd.Series(supertrend, index=high.index),
            'direction': pd.Series(direction, index=high.index)
        })
    
    def dpo(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Detrended Price Oscillator."""
        sma = self.sma(prices, window)
        dpo = prices - sma.shift(window // 2 + 1)
        return dpo
    
    def trix(self, prices: pd.Series, window: int = 15, signal: int = 9) -> pd.DataFrame:
        """TRIX oscillator."""
        ema1 = self.ema(prices, window)
        ema2 = self.ema(ema1, window)
        ema3 = self.ema(ema2, window)
        trix = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100
        signal_line = self.ema(trix, signal)
        
        return pd.DataFrame({
            'trix': trix,
            'signal': signal_line
        })
    
    def momentum(self, prices: pd.Series, window: int = 10) -> pd.Series:
        """Momentum indicator."""
        return prices - prices.shift(window)
    
    def adx(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.DataFrame:
        """Average Directional Index."""
        # Calculate +DM and -DM
        high_diff = high.diff()
        low_diff = low.diff()
        
        plus_dm = pd.Series(0.0, index=high.index, dtype=float)
        minus_dm = pd.Series(0.0, index=high.index, dtype=float)
        
        plus_dm[(high_diff > low_diff) & (high_diff > 0)] = high_diff[(high_diff > low_diff) & (high_diff > 0)]
        minus_dm[(low_diff > high_diff) & (low_diff > 0)] = low_diff[(low_diff > high_diff) & (low_diff > 0)]
        
        # Calculate TR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate smoothed values
        tr_smooth = tr.rolling(window=window).sum()
        plus_dm_smooth = plus_dm.rolling(window=window).sum()
        minus_dm_smooth = minus_dm.rolling(window=window).sum()
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = self.sma(dx, window)
        
        return pd.DataFrame({
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        })
    
    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        obv = pd.Series(0.0, index=close.index, dtype=float)
        obv.iloc[0] = float(volume.iloc[0])
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
        """Money Flow Index."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = pd.Series(0.0, index=typical_price.index, dtype=float)
        negative_flow = pd.Series(0.0, index=typical_price.index, dtype=float)
        
        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = float(money_flow.iloc[i])
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                negative_flow.iloc[i] = float(money_flow.iloc[i])
        
        positive_mf = positive_flow.rolling(window=window).sum()
        negative_mf = negative_flow.rolling(window=window).sum()
        
        mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
        return mfi
    
    def accumulation_distribution(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Accumulation/Distribution Line."""
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        money_flow_volume = money_flow_multiplier * volume
        ad_line = money_flow_volume.cumsum()
        return ad_line
    
    def chaikin_money_flow(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
        """Chaikin Money Flow."""
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        money_flow_volume = money_flow_multiplier * volume
        cmf = money_flow_volume.rolling(window=window).sum() / volume.rolling(window=window).sum()
        return cmf
    
    def vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, 
             reset_daily: bool = True) -> pd.Series:
        """Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        price_volume = typical_price * volume
        
        if reset_daily and hasattr(typical_price.index, 'date'):
            # Reset daily only if index has date attribute
            vwap = pd.Series(index=typical_price.index, dtype=float)
            for date in typical_price.index.date.unique():
                mask = typical_price.index.date == date
                date_pv = price_volume[mask]
                date_vol = volume[mask]
                vwap[mask] = date_pv.cumsum() / date_vol.cumsum()
        else:
            # Anchored VWAP
            vwap = price_volume.cumsum() / volume.cumsum()
        
        return vwap
    
    def chaikin_oscillator(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                          fast: int = 3, slow: int = 10) -> pd.Series:
        """Chaikin Oscillator."""
        ad_line = self.accumulation_distribution(high, low, close, volume)
        fast_ema = self.ema(ad_line, fast)
        slow_ema = self.ema(ad_line, slow)
        chaikin_osc = fast_ema - slow_ema
        return chaikin_osc
    
    def ease_of_movement(self, high: pd.Series, low: pd.Series, volume: pd.Series, window: int = 14) -> pd.DataFrame:
        """Ease of Movement."""
        distance_moved = (high + low) / 2 - (high.shift(1) + low.shift(1)) / 2
        box_ratio = volume / (high - low)
        emv = distance_moved / box_ratio
        emv_sma = self.sma(emv, window)
        
        return pd.DataFrame({
            'emv': emv,
            'emv_sma': emv_sma
        })
    
    def force_index(self, close: pd.Series, volume: pd.Series, window: int = 13) -> pd.Series:
        """Force Index."""
        force = close.diff() * volume
        force_ema = self.ema(force, window)
        return force_ema
    
    def volume_roc(self, volume: pd.Series, window: int = 14) -> pd.Series:
        """Volume Rate of Change."""
        return ((volume - volume.shift(window)) / volume.shift(window)) * 100
    
    def nvi(self, close: pd.Series, volume: pd.Series, ma_window: int = 255) -> pd.Series:
        """Negative Volume Index."""
        nvi = pd.Series(1000.0, index=close.index, dtype=float)
        
        for i in range(1, len(close)):
            if volume.iloc[i] < volume.iloc[i-1]:
                nvi.iloc[i] = float(nvi.iloc[i-1] * (1 + (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]))
            else:
                nvi.iloc[i] = nvi.iloc[i-1]
        
        nvi_ma = self.sma(nvi, ma_window)
        return pd.Series(nvi, index=close.index)
    
    def mfv(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
        """Money Flow Volume."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        # Determine if money flow is positive or negative
        signed_volume = pd.Series(0.0, index=volume.index, dtype=float)
        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                signed_volume.iloc[i] = float(volume.iloc[i])
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                signed_volume.iloc[i] = float(-volume.iloc[i])
        
        mfv = signed_volume.rolling(window=window).sum()
        return mfv
    
    def close_std(self, close: pd.Series, window: int = 20) -> pd.Series:
        """Rolling standard deviation of close prices."""
        return close.rolling(window=window).std()
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for the given DataFrame."""
        self.logger.info("Calculating all technical indicators")
        
        result_df = df.copy()
        
        # Extract OHLCV data
        high = df['high']
        low = df['low']
        close = df['close']
        open_price = df['open']
        volume = df['volume']
        
        # SMA indicators
        for window in self.config.indicators["sma"]:
            result_df[f'sma_{window}'] = self.sma(close, window)
        
        # EMA indicators
        for window in self.config.indicators["ema"]:
            result_df[f'ema_{window}'] = self.ema(close, window)
        
        # MACD
        macd_params = self.config.indicators["macd"]
        macd_result = self.macd(close, macd_params[0], macd_params[1], macd_params[2]).add_prefix('macd_')
        result_df = pd.concat([result_df, macd_result], axis=1)
        
        # RSI
        rsi_params = self.config.indicators["rsi"]
        result_df['rsi'] = self.rsi(close, rsi_params[0])
        
        # Stochastic
        stoch_params = self.config.indicators["stochastic"]
        stoch_result = self.stochastic(high, low, close, stoch_params[0], stoch_params[1]).add_prefix('stoch_')
        result_df = pd.concat([result_df, stoch_result], axis=1)
        
        # Bollinger Bands
        bb_params = self.config.indicators["bollinger"]
        bb_result = self.bollinger_bands(close, bb_params[0], bb_params[1]).add_prefix('bb_')
        result_df = pd.concat([result_df, bb_result], axis=1)
        
        # Parabolic SAR
        psar_params = self.config.indicators["parabolic_sar"]
        result_df['parabolic_sar'] = self.parabolic_sar(high, low, psar_params[0], psar_params[1])
        
        # Ichimoku
        ichimoku_params = self.config.indicators["ichimoku"]
        ichimoku_result = self.ichimoku(high, low, ichimoku_params[0], ichimoku_params[1], ichimoku_params[2]).add_prefix('ichimoku_')
        result_df = pd.concat([result_df, ichimoku_result], axis=1)
        
        # ATR
        atr_params = self.config.indicators["atr"]
        result_df['atr'] = self.atr(high, low, close, atr_params[0])
        
        # CCI
        cci_params = self.config.indicators["cci"]
        result_df['cci'] = self.cci(high, low, close, cci_params[0])
        
        # ROC
        roc_params = self.config.indicators["roc"]
        result_df['roc'] = self.roc(close, roc_params[0])
        
        # Williams %R
        williams_params = self.config.indicators["williams_r"]
        result_df['williams_r'] = self.williams_r(high, low, close, williams_params[0])
        
        # Keltner Channels
        keltner_params = self.config.indicators["keltner"]
        keltner_result = self.keltner_channels(high, low, close, keltner_params[0], keltner_params[1], keltner_params[2]).add_prefix('keltner_')
        result_df = pd.concat([result_df, keltner_result], axis=1)
        
        # Supertrend
        supertrend_params = self.config.indicators["supertrend"]
        supertrend_result = self.supertrend(high, low, close, supertrend_params[0], supertrend_params[1]).add_prefix('supertrend_')
        result_df = pd.concat([result_df, supertrend_result], axis=1)
        
        # DPO
        dpo_params = self.config.indicators["dpo"]
        result_df['dpo'] = self.dpo(close, dpo_params[0])
        
        # TRIX
        trix_params = self.config.indicators["trix"]
        trix_result = self.trix(close, trix_params[0], trix_params[1]).add_prefix('trix_')
        result_df = pd.concat([result_df, trix_result], axis=1)
        
        # Momentum
        momentum_params = self.config.indicators["momentum"]
        result_df['momentum'] = self.momentum(close, momentum_params[0])
        
        # ADX
        adx_params = self.config.indicators["adx"]
        adx_result = self.adx(high, low, close, adx_params[0]).add_prefix('adx_')
        result_df = pd.concat([result_df, adx_result], axis=1)
        
        # OBV
        result_df['obv'] = self.obv(close, volume)
        
        # MFI
        mfi_params = self.config.indicators["mfi"]
        result_df['mfi'] = self.mfi(high, low, close, volume, mfi_params[0])
        
        # Accumulation/Distribution Line
        result_df['ad_line'] = self.accumulation_distribution(high, low, close, volume)
        
        # Chaikin Money Flow
        cmf_params = self.config.indicators["chaikin_money_flow"]
        result_df['chaikin_money_flow'] = self.chaikin_money_flow(high, low, close, volume, cmf_params[0])
        
        # VWAP
        vwap_params = self.config.indicators["vwap"]
        result_df['vwap'] = self.vwap(high, low, close, volume, reset_daily=True)
        
        # Chaikin Oscillator
        co_params = self.config.indicators["chaikin_oscillator"]
        result_df['chaikin_oscillator'] = self.chaikin_oscillator(high, low, close, volume, co_params[0], co_params[1])
        
        # Ease of Movement
        eom_params = self.config.indicators["ease_of_movement"]
        eom_result = self.ease_of_movement(high, low, volume, eom_params[0]).add_prefix('eom_')
        result_df = pd.concat([result_df, eom_result], axis=1)
        
        # Force Index
        fi_params = self.config.indicators["force_index"]
        result_df['force_index'] = self.force_index(close, volume, fi_params[0])
        
        # Volume ROC
        vroc_params = self.config.indicators["volume_roc"]
        result_df['volume_roc'] = self.volume_roc(volume, vroc_params[0])
        
        # NVI
        nvi_params = self.config.indicators["nvi"]
        result_df['nvi'] = self.nvi(close, volume, nvi_params[0])
        
        # MFV
        mfv_params = self.config.indicators["mfv"]
        result_df['mfv'] = self.mfv(high, low, close, volume, mfv_params[0])
        
        # Close Standard Deviation
        close_std_params = self.config.indicators["close_std"]
        result_df['close_std'] = self.close_std(close, close_std_params[0])
        
        # Drop duplicate columns, keeping the last occurrence (the prefixed one)
        result_df = result_df.loc[:, ~result_df.columns.duplicated(keep='last')]
        self.logger.info(f"Calculated {len(result_df.columns) - len(df.columns)} technical indicators")
        return result_df