import yfinance as yf
import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TimeMachine")


def build_historical_dataset(symbol="BTC-USD", interval="1h", period="2y"):
    logger.info(f"Initiating Time Machine. Fetching {period} of {interval} candles for {symbol}...")

    # 1. Download Historical Data
    df = yf.download(symbol, interval=interval, period=period, progress=False)

    if df.empty:
        logger.error("Failed to download data. The timeline is empty.")
        return

    # Fix for newer versions of yfinance returning MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Feature Engineering
    # CRITICAL: These must mathematically match `calculate_state` in async_engine.py
    logger.info("Calculating neural network observation states...")

    # 10-period moving average
    df['moving_avg'] = df['Close'].rolling(window=10).mean()
    # Distance from moving average (normalized)
    df['dist_from_ma'] = (df['Close'] - df['moving_avg']) / df['moving_avg']
    # Percentage returns for calculating the reward later
    df['Returns'] = df['Close'].pct_change()

    # Drop the first 10 rows because the moving average needs 10 hours to warm up
    df.dropna(inplace=True)

    # 3. Clean and isolate the exact columns the AI will see
    dataset = df[['Close', 'Volume', 'dist_from_ma', 'Returns']].copy()

    # 4. Save to Vault
    os.makedirs("data", exist_ok=True)
    filepath = f"data/historical_{symbol.replace('-', '').lower()}.csv"
    dataset.to_csv(filepath)

    logger.info(f"Successfully locked {len(dataset)} hours of combat data into {filepath}")
    return filepath


if __name__ == "__main__":
    build_historical_dataset()