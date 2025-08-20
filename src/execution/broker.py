# src/execution/broker.py
import logging
import time
from alpaca_trade_api.rest import APIError, REST
from django.conf import settings

logger = logging.getLogger("rl_trading_backend")


class Broker:
    """
    Lightweight Alpaca broker wrapper.
    Compatible with existing settings:
      settings.API_KEY
      settings.SECRET_KEY_ALPACA (loaded from env SECRET_KEY)
      settings.BASE_URL (already a full URL chosen in settings.py)
    """

    def __init__(self):
        api_key = getattr(settings, 'ALPACA_API_KEY', None) or getattr(settings, 'API_KEY', None)
        secret_key = (
            getattr(settings, 'ALPACA_SECRET_KEY', None)
            or getattr(settings, 'SECRET_KEY_ALPACA', None)
            or getattr(settings, 'SECRET_KEY', None)   # legacy / discouraged
        )
        raw_base = getattr(settings, 'BASE_URL', 'paper')

        if not api_key or not secret_key:
            raise ValueError("Missing Alpaca credentials (API_KEY / SECRET_KEY).")

        base_url = self._normalize_base_url(raw_base)

        logger.info(f"Broker initializing (endpoint={base_url})")
        try:
            self.api = REST(key_id=api_key, secret_key=secret_key, base_url=base_url, api_version='v2')
            self.account = self.api.get_account()
            logger.info(
                f"Connected to Alpaca: status={self.account.status} equity={self.account.equity} buying_power={self.account.buying_power}"
            )
        except APIError as e:
            # Distinguish auth quickly
            msg = str(e).lower()
            if 'not authorized' in msg or 'access key verification failed' in msg or 'forbidden' in msg:
                logger.error(f"Alpaca authorization failed: {e}", exc_info=True)
            else:
                logger.error(f"Alpaca API error during init: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected broker init error: {e}", exc_info=True)
            raise

    @staticmethod
    def _normalize_base_url(value: str) -> str:
        """
        Accept:
          - full URLs (returned unchanged)
          - 'paper' / 'live' tokens (map to official URLs)
          - anything else starting with http(s) is passed through
        """
        if not value:
            return 'https://paper-api.alpaca.markets'
        v = value.strip()
        lower = v.lower()
        if lower.startswith('http://') or lower.startswith('https://'):
            return v
        if lower == 'paper':
            return 'https://paper-api.alpaca.markets'
        if lower == 'live':
            return 'https://api.alpaca.markets'
        # Fallback: if user put something unexpected, assume they meant a full URL but forgot scheme
        if 'alpaca' in lower and not lower.startswith('http'):
            return 'https://' + v
        return v

    def _refresh_account(self):
        if not self.api:
            return None
        try:
            self.account = self.api.get_account()
            return self.account
        except APIError as e:
            logger.error(f"Failed to refresh account: {e}")
            return None

    def get_account(self):
        return self._refresh_account()

    def get_equity(self) -> float:
        acct = self._refresh_account()
        try:
            return float(acct.equity) if acct else 0.0
        except Exception:
            return 0.0

    def get_buying_power(self) -> float:
        acct = self._refresh_account()
        try:
            return float(acct.buying_power) if acct else 0.0
        except Exception:
            return 0.0

    def get_positions(self) -> list:
        if not self.api:
            return []
        try:
            return self.api.list_positions()
        except APIError as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def place_market_order(self, symbol: str, side: str, notional_value: float) -> bool:
        """
        Market order by converting notional to whole-share quantity (buy) or full position (sell).
        """
        if not self.api:
            logger.error("Broker API not initialized.")
            return False
        side_l = side.lower()
        try:
            if side_l == 'buy':
                last_price = self.api.get_latest_trade(symbol).price
                qty = int(notional_value // last_price)
                if qty <= 0:
                    logger.warning(
                        f"Notional ${notional_value:.2f} insufficient for one share of {symbol} at ${last_price:.2f}."
                    )
                    return False
            elif side_l == 'sell':
                try:
                    position = self.api.get_position(symbol)
                except APIError as e:
                    if 'position does not exist' in str(e).lower() or 'position not found' in str(e).lower():
                        logger.warning(f"No position to sell for {symbol}.")
                        return False
                    raise
                qty = int(abs(float(position.qty)))
                if qty <= 0:
                    logger.warning(f"Computed zero quantity to sell for {symbol}.")
                    return False
            else:
                logger.error(f"Invalid order side: {side}")
                return False

            logger.info(f"Submitting {side_l.upper()} order: {symbol} qty={qty}")
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side_l,
                type="market",
                time_in_force="day",
            )

            # Simple confirmation loop
            for _ in range(5):
                time.sleep(1)
                chk = self.api.get_order(order.id)
                if chk.status in ('accepted', 'new', 'pending_new', 'partially_filled', 'filled'):
                    logger.info(f"Order {order.id} status={chk.status}")
                    return True
                if chk.status in ('rejected', 'canceled', 'expired'):
                    logger.error(f"Order {order.id} failed status={chk.status} symbol={symbol}")
                    return False
            logger.warning(f"Order {order.id} not confirmed after polling.")
            return False

        except APIError as e:
            msg = str(e).lower()
            if 'not authorized' in msg:
                logger.error(f"Authorization error placing order: {e}")
            else:
                logger.error(f"APIError placing order for {symbol}: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error placing order for {symbol}: {e}", exc_info=True)
            return False