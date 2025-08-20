import logging
import time
from alpaca_trade_api.rest import APIError, REST
from django.conf import settings
from datetime import datetime, timedelta
import pytz
from alpaca_trade_api.common import URL
import math

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

    def _get_last_close_price(self, symbol: str) -> float:
        """Get the last closing price for gap protection"""
        try:
            # Get last day's bar data
            from alpaca_trade_api.rest import TimeFrame
            bars = self.api.get_bars(
                symbol,
                TimeFrame.Day,
                limit=1,
                asof=datetime.now().strftime('%Y-%m-%d')
            )

            if bars and len(bars) > 0:
                # bars is a list of Bar objects
                last_bar = bars[0]
                return float(last_bar.c)  # closing price
            else:
                logger.warning(f"No bar data found for {symbol}")
                return None

        except Exception as e:
            logger.warning(f"Could not get bars for {symbol}: {e}")

        try:
            # Fallback: get latest trade price
            latest_trade = self.api.get_latest_trade(symbol)
            return float(latest_trade.price)
        except Exception as e:
            logger.warning(f"Could not get latest trade for {symbol}: {e}")
            return None

    def _check_gap_protection(self, symbol: str, side: str, max_gap_percent: float = 5.0) -> tuple:
        """
        Check if current price gaps too much from last close.
        Returns (is_safe: bool, current_price: float, last_close: float, gap_percent: float)
        """
        try:
            # Get current market price
            latest_trade = self.api.get_latest_trade(symbol)
            current_price = float(latest_trade.price)

            # Get last close price
            last_close = self._get_last_close_price(symbol)

            if last_close is None:
                logger.warning(
                    f"⚠️ Gap protection: Could not get last close for {symbol}, proceeding without protection")
                return True, current_price, None, 0.0

            # Calculate gap percentage
            gap_percent = abs((current_price - last_close) / last_close) * 100

            # Check if gap is within acceptable range
            is_safe = gap_percent <= max_gap_percent

            gap_direction = "UP" if current_price > last_close else "DOWN"

            if not is_safe:
                logger.warning(f"🚨 GAP PROTECTION TRIGGERED: {symbol}")
                logger.warning(f"📊 Last Close: ${last_close:.2f} | Current: ${current_price:.2f}")
                logger.warning(f"📈 Gap: {gap_percent:.2f}% {gap_direction} (max allowed: {max_gap_percent}%)")
                logger.warning(f"❌ {side.upper()} order blocked due to excessive gap")
            else:
                logger.info(
                    f"✅ Gap check passed: {symbol} gap {gap_percent:.2f}% {gap_direction} (limit: {max_gap_percent}%)")

            return is_safe, current_price, last_close, gap_percent

        except Exception as e:
            logger.error(f"Error in gap protection for {symbol}: {e}")
            # On error, allow trade but warn
            try:
                latest_trade = self.api.get_latest_trade(symbol)
                current_price = float(latest_trade.price)
                return True, current_price, None, 0.0
            except:
                return True, 0.0, None, 0.0

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

    def get_next_market_open_minutes(self) -> int:
        """Get minutes until next market open for sleep timer"""
        try:
            clock = self.api.get_clock()
            if clock.is_open:
                return 0

            next_open = clock.next_open.replace(tzinfo=pytz.timezone('US/Eastern'))
            now_et = datetime.now(pytz.timezone('US/Eastern'))

            time_diff = next_open - now_et
            minutes_until_open = int(time_diff.total_seconds() / 60)

            logger.info(f"⏰ Market opens in {minutes_until_open} minutes ({time_diff})")
            return max(0, minutes_until_open)

        except Exception as e:
            logger.warning(f"Could not calculate market open time: {e}")
            # Fallback: assume market opens in 12 hours
            return 12 * 60

    def place_market_order(
            self,
            symbol: str,
            side: str,
            notional_value: float | None = None,
            qty: float | None = None,
            wait_fill: bool = True,
            timeout_sec: int = 300,  # 5 minutes default
            poll_interval: float = 3.0,
            enable_gap_protection: bool = True,
            max_gap_percent: float = 5.0
    ):
        """
        Place market order with gap protection and India timezone awareness.
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

        # Gap protection check
        if enable_gap_protection and market_open:
            is_safe, current_price, last_close, gap_percent = self._check_gap_protection(symbol, side, max_gap_percent)
            if not is_safe:
                logger.error(f"❌ Trade blocked: {symbol} gap {gap_percent:.2f}% exceeds limit {max_gap_percent}%")
                return False, None
        else:
            try:
                latest_trade = self.api.get_latest_trade(symbol)
                current_price = float(latest_trade.price)
            except:
                current_price = 0.0

        side_l = side.lower()
        try:
            if side_l == 'buy':
                if qty is None and notional_value is None:
                    raise ValueError("Buy requires notional_value or qty.")

                if qty is None:
                    if current_price <= 0:
                        logger.error(f"Invalid price {current_price} for {symbol}")
                        return False, None

                    qty = float(notional_value) / current_price
                    qty = round(qty, 6)  # Round to 6 decimal places for fractional shares
                    logger.info(
                        f"Calculated qty {qty} for {symbol} at ${current_price:.2f} (notional=${notional_value})")

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

    def close_position(self, symbol: str) -> bool:
        """
        Close a position entirely
        """
        try:
            position = self.api.get_position(symbol)
            qty = abs(float(position.qty))

            if qty <= 0:
                logger.info(f"No position to close for {symbol}")
                return True

            side = 'sell' if float(position.qty) > 0 else 'buy'
            logger.info(f"Closing position: {side} {qty} shares of {symbol}")

            filled, order = self.place_market_order(symbol=symbol, side=side, qty=qty)
            return filled

        except APIError as e:
            if 'position does not exist' in str(e).lower():
                logger.info(f"No position exists for {symbol}")
                return True
            logger.error(f"Error closing position for {symbol}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error closing position for {symbol}: {e}")
            return False

    def close_all_positions(self) -> dict:
        """
        Close all open positions
        Returns dict with results
        """
        results = {'success': [], 'failed': []}

        try:
            positions = self.get_positions()

            if not positions:
                logger.info("No positions to close")
                return results

            logger.info(f"Closing {len(positions)} positions")

            for position in positions:
                symbol = position.symbol
                try:
                    if self.close_position(symbol):
                        results['success'].append(symbol)
                        logger.info(f"✅ Closed position: {symbol}")
                    else:
                        results['failed'].append(symbol)
                        logger.error(f"❌ Failed to close position: {symbol}")

                    time.sleep(1)  # Rate limiting

                except Exception as e:
                    logger.error(f"Error closing {symbol}: {e}")
                    results['failed'].append(symbol)

        except Exception as e:
            logger.error(f"Error in close_all_positions: {e}")

        logger.info(f"Position closure complete: {len(results['success'])} closed, {len(results['failed'])} failed")
        return results

    def get_account_summary(self) -> dict:
        """
        Get comprehensive account information
        """
        try:
            account = self._refresh_account()
            positions = self.get_positions()

            return {
                'equity': float(account.equity) if account else 0.0,
                'buying_power': float(account.buying_power) if account else 0.0,
                'cash': float(account.cash) if account else 0.0,
                'portfolio_value': float(account.portfolio_value) if account else 0.0,
                'day_trade_count': int(account.daytrade_count) if account else 0,
                'position_count': len(positions),
                'positions_value': sum(float(p.market_value) for p in positions),
                'unrealized_pl': sum(float(p.unrealized_pl) for p in positions),
                'account_status': account.status if account else 'unknown'
            }
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return {
                'equity': 0.0, 'buying_power': 0.0, 'cash': 0.0,
                'portfolio_value': 0.0, 'day_trade_count': 0,
                'position_count': 0, 'positions_value': 0.0,
                'unrealized_pl': 0.0, 'account_status': 'error'
            }

    def has_position(self, symbol: str) -> bool:
        """
        Check if we have a position in the given symbol
        """
        try:
            position = self.api.get_position(symbol)
            return abs(float(position.qty)) > 0
        except APIError as e:
            if 'position does not exist' in str(e).lower():
                return False
            logger.error(f"Error checking position for {symbol}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking position for {symbol}: {e}")
            return False

    def get_position_value(self, symbol: str) -> float:
        """
        Get the market value of a position
        """
        try:
            position = self.api.get_position(symbol)
            return float(position.market_value)
        except APIError as e:
            if 'position does not exist' in str(e).lower():
                return 0.0
            logger.error(f"Error getting position value for {symbol}: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Unexpected error getting position value for {symbol}: {e}")
            return 0.0
