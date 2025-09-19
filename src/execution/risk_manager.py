import logging
from alpaca_trade_api.rest import APIError
from src.data.preprocessor import calculate_atr
import pandas as pd
from datetime import datetime

logger = logging.getLogger("rl_trading_backend")


class RiskManager:
    def __init__(self, broker, base_trade_usd=500.0, max_trade_usd=1000.0, **kwargs):
        self.broker = broker
        # These are now defaults, as position sizing is dynamic
        self.base_trade_usd = base_trade_usd
        self.max_trade_usd = max_trade_usd

        # Strategy parameters
        self.max_daily_trades = kwargs.get('max_daily_trades', 15)
        self.profit_take_multiplier = 3.0  # e.g. 3 * ATR
        self.stop_loss_multiplier = 1.5  # e.g. 1.5 * ATR

        # Portfolio level risk
        self.max_drawdown_pct = kwargs.get('max_drawdown_pct', 0.10)
        self.max_position_pct = kwargs.get('max_position_pct', 0.20)

        # State tracking
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.peak_equity = self.broker.get_equity()
        self.kill_switch_active = False

    def reset_daily_counters(self):
        """Reset daily counters if it's a new day."""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trade_count = 0
            self.last_trade_date = today
            logger.info(f"Daily trade counter reset for {today}.")

    def update_peak_equity(self):
        """Update the peak equity to calculate drawdown."""
        current_equity = self.broker.get_equity()
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

    def check_drawdown(self):
        """Check for excessive drawdown and activate kill switch if necessary."""
        self.update_peak_equity()
        current_equity = self.broker.get_equity()
        drawdown = (self.peak_equity - current_equity) / self.peak_equity

        if drawdown > self.max_drawdown_pct:
            self.kill_switch_active = True
            logger.critical(
                f"KILL SWITCH ACTIVATED: Drawdown of {drawdown:.2%} exceeds limit of {self.max_drawdown_pct:.2%}")

    def get_risk_status(self):
        """Returns a dictionary with the current risk status."""
        self.reset_daily_counters()
        self.check_drawdown()
        current_equity = self.broker.get_equity()
        drawdown = (self.peak_equity - current_equity) / self.peak_equity

        return {
            "daily_trades_used": self.daily_trade_count,
            "drawdown_pct": drawdown * 100,
            "kill_switch_active": self.kill_switch_active,
            "peak_equity": self.peak_equity
        }

    def check_daily_trade_limit(self):
        self.reset_daily_counters()
        return self.daily_trade_count < self.max_daily_trades

    def check_buy_cooldown(self, ticker):
        # A more sophisticated implementation would track last buy time per ticker
        return True

    def check_trade(self, ticker, action, confidence):
        # This method is now simpler as sizing is handled in TradingSession
        if self.kill_switch_active:
            logger.warning("Trade blocked: Kill switch is active.")
            return False, 0

        if not self.check_daily_trade_limit():
            logger.warning("Trade blocked: Daily trade limit reached.")
            return False, 0

        # Can add more checks here (e.g., max exposure per ticker)

        if action == 1:  # BUY
            return True, self.base_trade_usd  # Return a placeholder value
        return True, 0  # Allow sells

    def should_take_profit_or_stop_loss(self, symbol):
        """
        Determines if a position should be closed based on dynamic take-profit and stop-loss levels.
        """
        try:
            position = self.broker.api.get_position(symbol)
            entry_price = float(position.avg_entry_price)
            current_price = float(position.current_price)

            # Get historical data to calculate ATR
            bars = self.broker.api.get_bars(symbol, "1D", limit=20)
            if not bars or len(bars) < 14:
                return False, None

            df = pd.DataFrame(
                [(bar.h, bar.l, bar.c) for bar in bars],
                columns=['High', 'Low', 'Close']
            )

            atr = calculate_atr(df['High'], df['Low'], df['Close'], 14).iloc[-1]

            if atr == 0: return False, None

            take_profit_price = entry_price + (self.profit_take_multiplier * atr)
            stop_loss_price = entry_price - (self.stop_loss_multiplier * atr)

            logger.info(
                f"{symbol} | Entry: {entry_price:.2f} | Current: {current_price:.2f} | TP: {take_profit_price:.2f} | SL: {stop_loss_price:.2f}")

            if current_price >= take_profit_price:
                return True, "PROFIT_TAKE"

            if current_price <= stop_loss_price:
                return True, "STOP_LOSS"

        except APIError:
            # Position likely doesn't exist anymore
            return False, None
        except Exception as e:
            logger.error(f"Error checking TP/SL for {symbol}: {e}")

        return False, None