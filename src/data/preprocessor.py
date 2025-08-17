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
    # RSI (Relative Strength Index)
    df["RSI_14"] = calculate_rsi(df["Close"], 14)

    # Stochastic %K
    df["STOCHk_14_3_3"] = calculate_stochastic_k(
        df["High"], df["Low"], df["Close"], 14, 3
    )

    # --- Trend Indicators ---
    # MACD Histogram
    df["MACDh_12_26_9"] = calculate_macd_histogram(df["Close"], 12, 26, 9)

    # ADX (Average Directional Index)
    df["ADX_14"] = calculate_adx(df["High"], df["Low"], df["Close"], 14)

    # --- Volatility Indicators ---
    # Bollinger Bands Percentage
    df["BBP_20_2"] = calculate_bollinger_bands_percentage(df["Close"], 20, 2)

    # ATR (Average True Range)
    df["ATR_14"] = calculate_atr(df["High"], df["Low"], df["Close"], 14)

    # --- Volume Indicators ---
    # OBV (On-Balance Volume)
    df["OBV"] = calculate_obv(df["Close"], df["Volume"])

    # Our observation space requires no NaN values.
    df.dropna(inplace=True)

    # Normalize some indicators that don't have a natural 0-100 range
    if "OBV" in df.columns:
        df["OBV"] = df["OBV"].pct_change()

    df.dropna(inplace=True)

    return df


def calculate_rsi(close_prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_stochastic_k(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_window: int = 14,
    d_window: int = 3,
) -> pd.Series:
    """Calculate Stochastic %K"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    return k_percent.rolling(window=d_window).mean()


def calculate_macd_histogram(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.Series:
    """Calculate MACD Histogram"""
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line - signal_line


def calculate_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Calculate Average Directional Index"""
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    dm_plus = high.diff()
    dm_minus = low.diff() * -1

    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)

    # Smoothed averages
    atr = true_range.rolling(window=window).mean()
    di_plus = 100 * (dm_plus.rolling(window=window).mean() / atr)
    di_minus = 100 * (dm_minus.rolling(window=window).mean() / atr)

    # ADX calculation
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    return dx.rolling(window=window).mean()


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
    return true_range.rolling(window=window).mean()


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume"""
    obv = np.where(
        close > close.shift(), volume, np.where(close < close.shift(), -volume, 0)
    )
    return pd.Series(obv, index=close.index).cumsum()
