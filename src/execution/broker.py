# src/execution/broker.py

import alpaca_trade_api as tradeapi
import logging
from alpaca_trade_api.rest import APIError
from django.conf import settings

logger = logging.getLogger("rl_trading_backend")


class Broker:
    """
    Handles all interactions with the Alpaca trading API using the `alpaca-trade-api` library.
    """
    def __init__(self):
        try:
            # --- Read credentials directly from Django settings ---
            api_key = settings.API_KEY
            secret_key = settings.SECRET_KEY_ALPACA
            base_url_setting = settings.BASE_URL

            # --- Handle "paper" and "live" shorthand ---
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
            self.account = self.api.get_account()

        except Exception as e:
            logger.exception("Failed to connect to Alpaca. Check API credentials and endpoint in settings.")
            self.api = None
            self.account = None
            raise e

    def get_account(self):
        """Returns the Alpaca account object."""
        if not self.api: return None
        try:
            # Refresh account details
            self.account = self.api.get_account()
            return self.account
        except APIError as e:
            logger.error(f"Failed to get account details: {e}")
            return None

    def get_equity(self) -> float:
        """Returns the current total portfolio equity."""
        account = self.get_account()
        return float(account.equity) if account else 0.0

    def get_buying_power(self) -> float:
        """Returns the available buying power."""
        account = self.get_account()
        return float(account.buying_power) if account else 0.0

    def get_positions(self) -> list:
        """Returns a list of all open positions."""
        if not self.api: return []
        try:
            return self.api.list_positions()
        except APIError as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_net_exposure(self) -> float:
        """Calculates the total market value of all positions."""
        positions = self.get_positions()
        return sum(abs(float(p.market_value)) for p in positions)


    def get_position_value(self, symbol: str) -> float:
        try:
            position = self.api.get_position(symbol)
            return float(position.market_value)
        except tradeapi.rest.APIError: # Position does not exist
            return 0.0
        except Exception as e:
            logger.error(f"Error getting position value for {symbol}: {e}")
            return 0.0

    def place_market_order(self, symbol: str, side: str, notional_value: float = None, qty: float = None) -> bool:
        """
        Submits a market order. Handles both notional and quantity orders.
        """
        try:
            if notional_value and qty:
                logger.error("Order cannot have both 'qty' and 'notional_value'.")
                return False

            if notional_value:
                self.api.submit_order(
                    symbol=symbol,
                    notional=notional_value,
                    side=side,
                    type="market",
                    time_in_force="day",
                )
                logger.info(
                    f"Market Order Submitted: {side.upper()} ${notional_value:,.2f} of {symbol}"
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
        except APIError as e:
            logger.error(f"APIError placing order for {symbol}: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error placing order for {symbol}: {e}", exc_info=True)
            return False