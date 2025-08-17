# src/execution/broker.py

import alpaca_trade_api as tradeapi
import configparser
import logging
from alpaca_trade_api.rest import TimeFrame
import pandas as pd

logger = logging.getLogger("rl_trading_backend")


class Broker:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read("config.ini")

        self.api = tradeapi.REST(
            key_id=config["alpaca"]["API_KEY"],
            secret_key=config["alpaca"]["SECRET_KEY"],
            base_url=config["alpaca"]["BASE_URL"],
            api_version="v2",
        )
        try:
            account_status = self.api.get_account().status
            logger.info(f"Connected to Alpaca. Account Status: {account_status}")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise

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
