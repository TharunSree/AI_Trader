import logging
import time
from alpaca_trade_api.rest import APIError, REST
from django.conf import settings
from datetime import datetime
import pytz

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

    def _get_market_times_ist(self) -> dict:
        """Get market times in both ET and IST"""
        et_tz = pytz.timezone('US/Eastern')
        ist_tz = pytz.timezone('Asia/Kolkata')

        now_et = datetime.now(et_tz)
        now_ist = datetime.now(ist_tz)

        return {
            'et_time': now_et.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'ist_time': now_ist.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'et_hour': now_et.hour,
            'et_minute': now_et.minute,
            'et_weekday': now_et.weekday(),
            'ist_hour': now_ist.hour,
            'ist_minute': now_ist.minute,
            'ist_weekday': now_ist.weekday()
        }

    def _is_market_open(self) -> bool:
        """Check if US market is currently open (with IST logging)"""
        try:
            clock = self.api.get_clock()
            is_open = clock.is_open

            times = self._get_market_times_ist()

            logger.info(f"🕐 Market Status: {'OPEN' if is_open else 'CLOSED'}")
            logger.info(f"🇺🇸 US ET: {times['et_time']}")
            logger.info(f"🇮🇳 India IST: {times['ist_time']}")

            if not is_open:
                # Calculate when market opens next in IST
                if times['et_weekday'] >= 5:  # Weekend
                    logger.info("📅 Market closed: Weekend")
                elif times['et_hour'] < 9 or (times['et_hour'] == 9 and times['et_minute'] < 30):
                    logger.info(f"🌅 Market opens at 9:30 AM ET (8:00 PM IST)")
                else:
                    logger.info(f"🌙 Market closed until tomorrow 9:30 AM ET (8:00 PM IST)")

            return is_open

        except Exception as e:
            logger.warning(f"Could not check market status: {e}")

            # Fallback: manual calculation
            times = self._get_market_times_ist()
            et_hour = times['et_hour']
            et_minute = times['et_minute']
            et_weekday = times['et_weekday']

            # Market hours: Mon-Fri 9:30 AM - 4:00 PM ET
            if et_weekday >= 5:  # Weekend
                return False
            if et_hour < 9 or (et_hour == 9 and et_minute < 30) or et_hour >= 16:
                return False
            return True

    def place_market_order(
            self,
            symbol: str,
            side: str,
            notional_value: float | None = None,
            qty: float | None = None,
            wait_fill: bool = True,
            timeout_sec: int = 300,  # 5 minutes default
            poll_interval: float = 3.0
    ):
        """
        Place market order with India timezone awareness.
        Returns (filled: bool, order_obj)
        """
        # Check market status with IST logging
        market_open = self._is_market_open()

        if not market_open:
            logger.warning(
                f"⚠️ US Market CLOSED - orders placed during Indian daytime won't fill until US market opens")
            logger.warning(f"💡 US Market Hours: 9:30 AM - 4:00 PM ET = 8:00 PM - 2:30 AM IST")
            timeout_sec = 30  # Very short timeout when market closed
            wait_fill = False  # Don't wait for fills when market closed

        side_l = side.lower()
        try:
            if side_l == 'buy':
                if qty is None and notional_value is None:
                    raise ValueError("Buy requires notional_value or qty.")

                if qty is None:
                    try:
                        # Get current market price
                        latest_trade = self.api.get_latest_trade(symbol)
                        current_price = float(latest_trade.price)
                        qty = float(notional_value) / current_price

                        # Round to 6 decimal places for fractional shares
                        qty = round(qty, 6)
                        logger.info(
                            f"Calculated qty {qty} for {symbol} at ${current_price:.2f} (notional=${notional_value})")
                    except Exception as e:
                        logger.error(f"Failed to get price for {symbol}: {e}")
                        return False, None

                # Check if symbol supports fractional shares
                try:
                    asset = self.api.get_asset(symbol)
                    if not asset.fractionable and qty < 1.0:
                        logger.warning(f"{symbol} doesn't support fractional shares. Rounding up to 1 share.")
                        qty = 1.0
                except Exception as e:
                    logger.warning(f"Could not check fractional support for {symbol}: {e}")

                submit_args = dict(
                    symbol=symbol,
                    side='buy',
                    type='market',
                    time_in_force='day',
                    qty=qty
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
                    qty=qty
                )
            else:
                logger.warning(f"Invalid side {side}")
                return False, None

            logger.info(f"🚀 Submitting {side_l.upper()} order: {submit_args}")
            order = self.api.submit_order(**submit_args)
            logger.info(f"📋 Order submitted: {order.id} status={order.status} qty={getattr(order, 'qty', 'N/A')}")

            if not wait_fill:
                logger.info(f"📤 Order submitted successfully (not waiting for fill due to market status)")
                return True, order

            # Enhanced polling with market-aware logic
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
                        logger.info(
                            f"📊 Poll {poll_count}: Order {current_order.id} status: {last_status} -> {status} (filled_qty={filled_qty})")
                        last_status = status

                    # Success states
                    if status == 'filled':
                        logger.info(
                            f"✅ Order {current_order.id} FILLED: qty={current_order.filled_qty} avg_price=${current_order.filled_avg_price}")
                        return True, current_order

                    # Failure states
                    elif status in ('canceled', 'rejected', 'expired'):
                        logger.error(f"❌ Order {current_order.id} FAILED: status={status}")
                        return False, current_order

                    # Market closed but order pending - treat as success
                    elif not market_open and status in ('new', 'accepted'):
                        if poll_count >= 3:  # Give it a few polls then accept
                            logger.info(
                                f"📈 Order {current_order.id} pending (US market closed) - will fill when market opens")
                            return True, current_order

                    # Processing states - continue waiting
                    elif status in ('new', 'accepted', 'pending_new', 'pending_cancel', 'pending_replace'):
                        continue

                    # Partial fills
                    elif status == 'partially_filled':
                        logger.info(f"📊 Order {current_order.id} partially filled: {filled_qty}")
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

                logger.warning(
                    f"⏰ Order {final_order.id} timeout after {timeout_sec}s. Status: {final_order.status}, filled_qty: {filled_qty}")

                # Accept partial fills or pending orders when market closed
                if filled_qty > 0:
                    logger.info(f"✅ Order {final_order.id} partial fill accepted: {filled_qty}")
                    return True, final_order
                elif not market_open and final_order.status in ('new', 'accepted'):
                    logger.info(f"📈 Order {final_order.id} pending (US market closed)")
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