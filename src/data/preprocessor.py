# src/data/preprocessor.py

import pandas as pd
import numpy as np


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a comprehensive set of technical features for the given DataFrame.
    """
    # Ensure the DataFrame has the necessary columns (Open, High, Low, Close, Volume)
    df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
        errors="ignore",
    )

    # --- Basic Features ---
    df["returns"] = df["Close"].pct_change()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()  # 50-day Simple Moving Average

    # --- Momentum Indicators ---
    df["RSI_14"] = calculate_rsi(df["Close"], 14)
    df["STOCHk_14_3_3"] = calculate_stochastic_k(
        df["High"], df["Low"], df["Close"], 14, 3
    )

    # --- Trend Indicators ---
    df["MACDh_12_26_9"] = calculate_macd_histogram(df["Close"], 12, 26, 9)
    df["ADX_14"] = calculate_adx(df["High"], df["Low"], df["Close"], 14)

    # --- Volatility Indicators ---
    df["BBP_20_2"] = calculate_bollinger_bands_percentage(df["Close"], 20, 2)
    df["ATR_14"] = calculate_atr(df["High"], df["Low"], df["Close"], 14)

    # --- Volume Indicators ---
    df["OBV"] = calculate_obv(df["Close"], df["Volume"])

    # Add missing volume features
    if 'Volume' in df.columns:
        df['volume_sma_5'] = df['Volume'].rolling(window=5).mean()
        df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
        df['volume_sma_ratio'] = df['Volume'] / df['volume_sma_20']
        df['volume_ema_ratio'] = df['Volume'] / df['Volume'].ewm(span=10).mean()
    else:
        # Default values if no volume data
        df['volume_sma_5'] = 1.0
        df['volume_sma_20'] = 1.0
        df['volume_sma_ratio'] = 1.0
        df['volume_ema_ratio'] = 1.0

    # Fill any NaN values that might have been created
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.fillna(0, inplace=True)

    return df


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_stochastic_k(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14, smooth_k: int = 3
) -> pd.Series:
    """Calculate Stochastic %K"""
    low_min = low.rolling(window=window).min()
    high_max = high.rolling(window=window).max()
    stoch_k = 100 * (close - low_min) / (high_max - low_min)
    return stoch_k.rolling(window=smooth_k).mean()


def calculate_macd_histogram(
    close: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> pd.Series:
    """Calculate MACD Histogram"""
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd - signal


def calculate_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Calculate Average Directional Index (ADX)"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    dm_plus = high.diff()
    dm_minus = -low.diff()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)

    atr = true_range.ewm(alpha=1/window, adjust=False).mean()
    di_plus = 100 * (dm_plus.ewm(alpha=1/window, adjust=False).mean() / atr)
    di_minus = 100 * (dm_minus.ewm(alpha=1/window, adjust=False).mean() / atr)

    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    return dx.ewm(alpha=1/window, adjust=False).mean()


def calculate_bollinger_bands_percentage(
    close: pd.Series, window: int = 20, std_dev: int = 2
) -> pd.Series:
    """Calculate Bollinger Bands Percentage"""
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return (close - lower_band) / (upper_band - lower_band)


def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(span=window, adjust=False).mean()


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume (OBV)"""
    # OBV is the cumulative sum of volume based on price direction
    price_direction = np.sign(close.diff())
    obv = (price_direction * volume).cumsum()
    return obv