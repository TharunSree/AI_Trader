# src/data/yfinance_loader.py

import requests
import pandas as pd
import logging
from datetime import datetime
import time

logger = logging.getLogger("rl_trading_backend")


class YFinanceLoader:
    def __init__(self, tickers: list, start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        logger.info(
            f"YFinanceLoader initialized for tickers {tickers} from {start_date} to {end_date}"
        )

    def _convert_date_to_timestamp(self, date_str: str) -> int:
        """Convert date string to Unix timestamp."""
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp())

    def load_data(self) -> pd.DataFrame:
        try:
            ticker_symbol = self.tickers[0]

            # Convert dates to timestamps
            start_timestamp = self._convert_date_to_timestamp(self.start_date)
            end_timestamp = self._convert_date_to_timestamp(self.end_date)

            # Yahoo Finance API endpoint
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker_symbol}"

            params = {
                "period1": start_timestamp,
                "period2": end_timestamp,
                "interval": "1d",
                "includePrePost": "true",
                "events": "div%2Csplit",
            }

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()

            if "chart" not in data or not data["chart"]["result"]:
                logger.error(
                    f"No data found for ticker {ticker_symbol} in the given date range."
                )
                return pd.DataFrame()

            result = data["chart"]["result"][0]

            # Extract timestamps and convert to dates
            timestamps = result["timestamp"]
            dates = [datetime.fromtimestamp(ts) for ts in timestamps]

            # Extract OHLCV data
            quotes = result["indicators"]["quote"][0]

            # Create DataFrame
            df = pd.DataFrame(
                {
                    "Open": quotes["open"],
                    "High": quotes["high"],
                    "Low": quotes["low"],
                    "Close": quotes["close"],
                    "Volume": quotes["volume"],
                },
                index=dates,
            )

            # Handle adjusted close if available
            if "adjclose" in result["indicators"]:
                df["Adj Close"] = result["indicators"]["adjclose"][0]["adjclose"]
            else:
                df["Adj Close"] = df["Close"]

            # Drop rows with NaN values
            df = df.dropna()

            if df.empty:
                logger.error(
                    f"No valid data found for ticker {ticker_symbol} after cleaning."
                )
                return pd.DataFrame()

            # Auto-adjust prices (similar to yfinance auto_adjust=True)
            if "Adj Close" in df.columns:
                adjustment_factor = df["Adj Close"] / df["Close"]
                df["Open"] = df["Open"] * adjustment_factor
                df["High"] = df["High"] * adjustment_factor
                df["Low"] = df["Low"] * adjustment_factor
                df["Close"] = df["Adj Close"]
                df = df.drop("Adj Close", axis=1)

            logger.info(
                f"Successfully loaded {len(df)} rows of data for {ticker_symbol}."
            )
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error occurred while loading data: {e}")
            return pd.DataFrame()
        except KeyError as e:
            logger.error(f"Data format error - missing key: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"An error occurred while loading data: {e}")
            return pd.DataFrame()
