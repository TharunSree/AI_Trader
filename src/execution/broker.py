import logging
import time
from alpaca_trade_api.rest import APIError, REST
from django.conf import settings

logger = logging.getLogger("rl_trading_backend")


class Broker:
    def __init__(self):
        api_key = getattr(settings, 'ALPACA_API_KEY', None) or getattr(settings, 'API_KEY', None)
        secret_key = (
                getattr(settings, 'ALPACA_SECRET_KEY', None)
                or getattr(settings, 'SECRET_KEY_ALPACA', None)
                or getattr(settings, 'SECRET_KEY', None)
        )
        raw_base = getattr(settings, 'BASE_URL', 'paper')
        if not api_key or not secret_key:
            raise ValueError("Missing Alpaca credentials (API_KEY / SECRET_KEY).")
        base_url = self._normalize_base_url(raw_base)
        logger.info(f"Broker initializing (endpoint={base_url})")
        try:
            self.api = REST(key_id=api_key, secret_key=secret_key, base_url=base_url, api_version='v2')
            acct = self.api.get_account()
            logger.info(f"Connected: status={acct.status} equity={acct.equity} buying_power={acct.buying_power}")
        except Exception as e:
            logger.error(f"Broker init failed: {e}", exc_info=True)
            raise

    @staticmethod
    def _normalize_base_url(value: str) -> str:
        if not value:
            return 'https://paper-api.alpaca.markets'
        v = value.strip()
        l = v.lower()
        if l.startswith(('http://', 'https://')):
            return v
        if l == 'paper':
            return 'https://paper-api.alpaca.markets'
        if l == 'live':
            return 'https://api.alpaca.markets'
        if 'alpaca' in l and not l.startswith('http'):
            return 'https://' + v
        return v

    def _refresh_account(self):
        try:
            return self.api.get_account()
        except Exception:
            return None

    def get_equity(self) -> float:
        acct = self._refresh_account()
        return float(acct.equity) if acct else 0.0

    def get_buying_power(self) -> float:
        acct = self._refresh_account()
        return float(acct.buying_power) if acct else 0.0

    def get_positions(self) -> list:
        try:
            return self.api.list_positions()
        except APIError as e:
            logger.error(f"Get positions failed: {e}")
            return []

    def place_market_order(
            self,
            symbol: str,
            side: str,
            notional_value: float | None = None,
            qty: int | None = None,
            wait_fill: bool = True,
            timeout_sec: int = 30,
            poll_interval: float = 1.0
    ):
        """
        Place a market order.
        For buys prefer notional (Alpaca will compute qty).
        For sells must supply qty (defaults to full position).
        Returns (filled: bool, order_obj)
        """
        side_l = side.lower()
        try:
            if side_l == 'buy':
                if qty is None and notional_value is None:
                    raise ValueError("Buy requires notional_value or qty.")
                submit_args = dict(symbol=symbol, side='buy', type='market', time_in_force='day')
                if qty is not None:
                    submit_args['qty'] = qty
                else:
                    submit_args['notional'] = round(float(notional_value), 2)
            elif side_l == 'sell':
                if qty is None:
                    # fetch full position
                    try:
                        pos = self.api.get_position(symbol)
                        qty = int(abs(float(pos.qty)))
                    except APIError as e:
                        if 'position does not exist' in str(e).lower():
                            logger.warning(f"No position to sell for {symbol}.")
                            return False, None
                        raise
                if qty <= 0:
                    logger.warning(f"Computed zero qty sell for {symbol}.")
                    return False, None
                submit_args = dict(symbol=symbol, side='sell', type='market', time_in_force='day', qty=qty)
            else:
                logger.warning(f"Invalid side {side}")
                return False, None

            logger.info(f"Submitting {side_l.upper()} order {submit_args}")
            order = self.api.submit_order(**submit_args)

            if not wait_fill:
                return True, order  # accepted

            deadline = time.time() + timeout_sec
            last_status = None
            while time.time() < deadline:
                time.sleep(poll_interval)
                o = self.api.get_order(order.id)
                last_status = o.status
                if last_status == 'filled':
                    logger.info(f"Order {o.id} filled qty={o.filled_qty} avg_price={o.filled_avg_price}")
                    return True, o
                if last_status in ('canceled', 'rejected', 'expired'):
                    logger.error(f"Order {o.id} failed status={last_status}")
                    return False, o
            # Timed out; return whatever partial fill (if any)
            o = self.api.get_order(order.id)
            if o.filled_qty and float(o.filled_qty) > 0:
                logger.warning(f"Order {o.id} timeout; partial fill {o.filled_qty}")
                return True, o
            logger.warning(f"Order {o.id} not filled after timeout (status={o.status}).")
            return False, o

        except APIError as e:
            logger.error(f"APIError placing order {symbol} side={side}: {e}", exc_info=True)
            return False, None
        except Exception as e:
            logger.error(f"Unexpected order error {symbol}: {e}", exc_info=True)
            return False, None
