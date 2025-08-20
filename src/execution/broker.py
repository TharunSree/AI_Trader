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
        qty: float | None = None,
        wait_fill: bool = True,
        timeout_sec: int = 180,
        poll_interval: float = 2.0
    ):
        """
        Place market order using qty-based approach for better reliability.
        Returns (filled: bool, order_obj)
        """
        side_l = side.lower()
        try:
            if side_l == 'buy':
                if qty is None and notional_value is None:
                    raise ValueError("Buy requires notional_value or qty.")

                # Always calculate quantity from notional for consistency
                if qty is None:
                    try:
                        # Get current market price
                        latest_trade = self.api.get_latest_trade(symbol)
                        current_price = float(latest_trade.price)
                        qty = float(notional_value) / current_price
                        logger.info(f"Calculated qty {qty:.6f} for {symbol} at ${current_price:.2f}")
                    except Exception as e:
                        logger.error(f"Failed to get price for {symbol}: {e}")
                        return False, None

                submit_args = dict(
                    symbol=symbol,
                    side='buy',
                    type='market',
                    time_in_force='day',
                    qty=str(qty)  # Always use qty, not notional
                )

            elif side_l == 'sell':
                if qty is None:
                    try:
                        pos = self.api.get_position(symbol)
                        qty = abs(float(pos.qty))
                    except APIError as e:
                        if 'position does not exist' in str(e).lower():
                            logger.warning(f"No position to sell for {symbol}.")
                            return False, None
                        raise

                if qty <= 0:
                    logger.warning(f"Zero qty to sell for {symbol}.")
                    return False, None

                submit_args = dict(
                    symbol=symbol,
                    side='sell',
                    type='market',
                    time_in_force='day',
                    qty=str(qty)
                )
            else:
                logger.warning(f"Invalid side {side}")
                return False, None

            logger.info(f"Submitting {side_l.upper()} order: {submit_args}")
            order = self.api.submit_order(**submit_args)
            logger.info(f"Order submitted: {order.id} status={order.status} qty={getattr(order, 'qty', 'N/A')}")

            if not wait_fill:
                return True, order

            # Enhanced polling with status progression tracking
            deadline = time.time() + timeout_sec
            poll_count = 0
            last_status = None

            while time.time() < deadline:
                time.sleep(poll_interval)
                poll_count += 1

                try:
                    current_order = self.api.get_order(order.id)
                    status = current_order.status
                    filled_qty = float(current_order.filled_qty or 0)

                    if status != last_status:
                        logger.info(f"Poll {poll_count}: Order {current_order.id} status changed: {last_status} -> {status} (filled_qty={filled_qty})")
                        last_status = status

                    # Success states
                    if status == 'filled':
                        logger.info(f"✅ Order {current_order.id} FILLED: qty={current_order.filled_qty} avg_price=${current_order.filled_avg_price}")
                        return True, current_order

                    # Failure states
                    elif status in ('canceled', 'rejected', 'expired'):
                        logger.error(f"❌ Order {current_order.id} FAILED: status={status}")
                        return False, current_order

                    # Processing states - continue waiting
                    elif status in ('new', 'accepted', 'pending_new', 'pending_cancel', 'pending_replace'):
                        continue

                    # Partial fills
                    elif status == 'partially_filled':
                        logger.info(f"🔄 Order {current_order.id} partially filled: {filled_qty}")
                        continue

                    else:
                        logger.warning(f"⚠️ Order {current_order.id} unknown status: {status}")
                        continue

                except Exception as e:
                    logger.error(f"Error polling order {order.id}: {e}")
                    time.sleep(1)

            # Timeout handling
            try:
                final_order = self.api.get_order(order.id)
                filled_qty = float(final_order.filled_qty or 0)

                logger.warning(f"⏰ Order {final_order.id} timeout after {timeout_sec}s. Status: {final_order.status}, filled_qty: {filled_qty}")

                # Accept partial fills
                if filled_qty > 0:
                    logger.info(f"✅ Order {final_order.id} partial fill accepted: {filled_qty}")
                    return True, final_order
                else:
                    logger.error(f"❌ Order {final_order.id} no fill after timeout")
                    return False, final_order

            except Exception as e:
                logger.error(f"Error in final order check: {e}")
                return False, order

        except APIError as e:
            logger.error(f"APIError placing {side} order for {symbol}: {e}", exc_info=True)
            return False, None
        except Exception as e:
            logger.error(f"Unexpected error placing {side} order for {symbol}: {e}", exc_info=True)
            return False, None