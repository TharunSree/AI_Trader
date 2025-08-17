# src/execution/broker.py

import alpaca_trade_api as tradeapi
import configparser
import logging
from alpaca_trade_api.rest import TimeFrame
import pandas as pd
from django.conf import settings

logger = logging.getLogger("rl_trading_backend")


class Broker:
    def __init__(self):
        try:
            # --- UPDATED: Read credentials directly from Django settings ---
            api_key = settings.API_KEY
            secret_key = settings.SECRET_KEY_ALPACA
            base_url_setting = settings.BASE_URL

            # --- NEW: Handle "paper" and "live" shorthand ---
            if base_url_setting == 'paper':
                base_url = 'https://paper-api.alpaca.markets'
            elif base_url_setting == 'live':
                base_url = 'https://api.alpaca.markets'
            else:
                base_url = base_url_setting # Use the value directly if it's a full URL

            self.api = tradeapi.REST(
                key_id=api_key,
                secret_key=secret_key,
                base_url=base_url,
                api_version='v2'
            )
            account_status = self.api.get_account().status
            logger.info(f"Connected to Alpaca. Endpoint: {base_url}, Account Status: {account_status}")
        except Exception as e:
            # --- NEW: Improved logging with full traceback ---
            logger.exception("Failed to connect to Alpaca. Check API credentials and endpoint in settings.")
            # Re-raise the exception so the calling task knows something went wrong
            raise e

    def get_buying_power(self) -> float:
        try:
            return float(self.api.get_account().buying_power)
        except Exception as e:
            logger.error(f"Failed to get buying power: {e}")
            return 0.0

    def get_position_value(self, symbol: str) -> float:
        try:
            position = self.api.get_position(symbol)
            return float(position.market_value)
        except tradeapi.rest.APIError:
            return 0.0
        except Exception as e:
            logger.error(f"Error getting position value for {symbol}: {e}")
            return 0.0

    def place_market_order(
        self, symbol: str, side: str, notional_value: float = None, qty: int = None
    ):
        try:
            if notional_value:
                self.api.submit_order(
                    symbol=symbol,
                    notional=notional_value,
                    side=side,
                    type="market",
                    time_in_force="day",
                )
                logger.info(
                    f"Market Order Submitted: {side.upper()} ${notional_value:.2f} of {symbol}"
                )
            elif qty:
                self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type="market",
                    time_in_force="day",
                )
                logger.info(
                    f"Market Order Submitted: {side.upper()} {qty} shares of {symbol}"
                )
            else:
                logger.error("Order must have either 'qty' or 'notional_value'.")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            return False
