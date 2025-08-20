import logging
import pandas as pd
from src.data.yfinance_loader import YFinanceLoader

logger = logging.getLogger("rl_trading_backend")


class Scanner:
    def __init__(self, stock_universe: list = None, config: dict = None):
        self.universe = (
            stock_universe if stock_universe is not None else self._get_sp500_tickers()
        )

        if config is None:
            self.config = {
                "min_price": 20.0,
                "max_price": 1000.0,
                "min_volume_avg_20d": 1_000_000,
                "min_volatility_pct_1d": 1.5,  # Loosened from 2.0 to allow more opportunities
            }
        else:
            self.config = config

        logger.info(f"Scanner initialized for a universe of {len(self.universe)} stocks.")
        logger.info(f"Using filter configuration: {self.config}")

    def _get_sp500_tickers(self) -> list:
        try:
            logger.info("Fetching S&P 500 component tickers from Wikipedia...")
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            table = pd.read_html(url)
            sp500_df = table[0]
            tickers = sp500_df["Symbol"].str.replace(".", "-", regex=False).tolist()
            logger.info(f"Successfully fetched {len(tickers)} S&P 500 tickers.")
            return tickers
        except Exception as e:
            logger.error(f"Could not fetch S&P 500 tickers: {e}. Falling back to a default list.")
            return ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "JPM", "V", "NVDA", "JNJ", "UNH"]

    def scan_for_opportunities(self, buying_power: float) -> list:
        hot_list = []
        logger.info(f"Starting scan on {len(self.universe)} tickers...")

        for ticker in self.universe:
            try:
                # Use a recent but fixed date range for scanning consistency
                loader = YFinanceLoader(
                    [ticker], start_date="2024-01-01", end_date="2025-12-31"
                )
                df = loader.load_data()

                if df.empty or len(df) < 30:
                    continue

                if self._apply_filters(df, ticker, buying_power):
                    hot_list.append(ticker)
                    logger.info(f"Ticker {ticker} passed filters and was added to the hot list.")
            except Exception as e:
                logger.warning(f"Could not process ticker {ticker}: {e}")
                continue

        logger.info(f"Scan complete. Found {len(hot_list)} opportunities: {hot_list}")
        return hot_list

    def get_active_tickers(self, buying_power: float) -> list:
        """Alias for scan_for_opportunities for backward compatibility"""
        return self.scan_for_opportunities(buying_power)

    def _apply_filters(self, df: pd.DataFrame, ticker: str, buying_power: float) -> bool:
        try:
            last_price = df["Close"].iloc[-1].item()

            if last_price > (buying_power * 0.5):
                return False

            if not (self.config["min_price"] < last_price < self.config["max_price"]):
                return False

            avg_volume = df["Volume"].rolling(window=20).mean().iloc[-1].item()
            if avg_volume < self.config["min_volume_avg_20d"]:
                return False

            close_today = df["Close"].iloc[-1].item()
            close_yesterday = df["Close"].iloc[-2].item()
            price_change_pct = ((close_today / close_yesterday) - 1) * 100
            if abs(price_change_pct) < self.config["min_volatility_pct_1d"]:
                return False

            return True
        except (IndexError, TypeError, ValueError) as e:
            logger.warning(f"Not enough data or data format issue for {ticker} to apply filters. Error: {e}")
            return False

    def get_filtered_universe(self, buying_power: float, max_tickers: int = 50) -> list:
        """Get a filtered subset of the universe for faster scanning"""
        filtered = []
        for ticker in self.universe[:max_tickers]:  # Limit to first N tickers
            try:
                loader = YFinanceLoader([ticker], start_date="2024-11-01", end_date="2025-01-31")
                df = loader.load_data()

                if not df.empty and len(df) >= 20:
                    last_price = df["Close"].iloc[-1]
                    if self.config["min_price"] <= last_price <= self.config["max_price"]:
                        if last_price <= (buying_power * 0.5):
                            filtered.append(ticker)

            except Exception as e:
                logger.warning(f"Error filtering {ticker}: {e}")
                continue

        logger.info(f"Filtered universe to {len(filtered)} tickers from {max_tickers}")
        return filtered