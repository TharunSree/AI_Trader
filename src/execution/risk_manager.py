import logging
from datetime import datetime, timedelta
from collections import defaultdict
from src.execution.broker import Broker

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Enhanced risk management with daily trade limits, position sizing, and profit controls.
    """

    def __init__(self, broker: Broker, base_trade_usd: float = 500.0, max_trade_usd: float = 1000.0,
                 max_drawdown_pct: float = 0.10, max_position_pct: float = 0.15,
                 max_daily_trades: int = 10, profit_take_pct: float = 0.02,
                 stop_loss_pct: float = 0.01):
        self.broker = broker
        self.base_trade_usd = base_trade_usd
        self.max_trade_usd = max_trade_usd
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_pct = max_position_pct
        self.max_daily_trades = max_daily_trades
        self.profit_take_pct = profit_take_pct
        self.stop_loss_pct = stop_loss_pct

        # Track daily trades
        self.daily_trades = defaultdict(int)
        self.current_date = datetime.now().date()

        # Track position entry times for cooldown
        self.position_entry_times = {}
        self.buy_cooldown_minutes = 30

        try:
            self.start_of_session_equity = self.broker.get_equity()
            if self.start_of_session_equity == 0:
                raise ValueError("Could not fetch starting equity from broker.")
            self.kill_switch_threshold = self.start_of_session_equity * (1 - self.max_drawdown_pct)
            logger.info(
                f"🛡️  Enhanced RiskManager initialized with starting equity ${self.start_of_session_equity:,.2f}")
            logger.info(f"⚠️  Kill switch threshold: ${self.kill_switch_threshold:,.2f}")
            logger.info(f"📈 Profit take: {self.profit_take_pct * 100}%, Stop loss: {self.stop_loss_pct * 100}%")
            logger.info(f"🔢 Max daily trades: {self.max_daily_trades}")
        except Exception as e:
            logger.error(f"FATAL: Could not initialize RiskManager. Error: {e}")
            raise

    def check_daily_trade_limit(self) -> bool:
        """Check if we've hit the daily trade limit"""
        today = datetime.now().date()
        if today != self.current_date:
            # Reset daily counter for new day
            self.daily_trades.clear()
            self.current_date = today

        return self.daily_trades[today] < self.max_daily_trades

    def check_buy_cooldown(self, symbol: str) -> bool:
        """Check if enough time has passed since last buy for this symbol"""
        if symbol not in self.position_entry_times:
            return True

        last_entry = self.position_entry_times[symbol]
        cooldown_end = last_entry + timedelta(minutes=self.buy_cooldown_minutes)
        return datetime.now() >= cooldown_end

    def should_take_profit_or_stop_loss(self, symbol: str) -> tuple[bool, str]:
        """Check if we should automatically close position due to profit/loss thresholds"""
        try:
            positions = self.broker.get_positions()
            for pos in positions:
                if pos.symbol == symbol:
                    unrealized_pl = float(pos.unrealized_pl)
                    market_value = abs(float(pos.market_value))

                    if market_value > 0:
                        pl_pct = unrealized_pl / market_value

                        if pl_pct >= self.profit_take_pct:
                            logger.info(f"🎯 Taking profit on {symbol}: {pl_pct * 100:.1f}% gain")
                            return True, "PROFIT_TAKE"
                        elif pl_pct <= -self.stop_loss_pct:
                            logger.warning(f"🛑 Stop loss triggered on {symbol}: {pl_pct * 100:.1f}% loss")
                            return True, "STOP_LOSS"
            return False, ""
        except Exception as e:
            logger.error(f"Error checking profit/loss for {symbol}: {e}")
            return False, ""

    def check_trade(self, symbol: str, action: int, confidence: float) -> tuple[bool, float]:
        """Enhanced trade checking with all risk controls"""
        if action not in [1, 2]:
            return False, 0.0

        # Check daily trade limit
        if not self.check_daily_trade_limit():
            logger.warning(f"🚫 TRADE REJECTED: Daily trade limit ({self.max_daily_trades}) reached")
            return False, 0.0

        current_equity = self.broker.get_equity()
        buying_power = self.broker.get_buying_power()

        # Kill switch check
        if current_equity < self.kill_switch_threshold:
            logger.critical(
                f"🔴 KILL SWITCH: Equity ${current_equity:,.2f} below threshold ${self.kill_switch_threshold:,.2f}")
            return False, 0.0

        # Enhanced position sizing based on confidence
        base_size = min(self.base_trade_usd, current_equity * 0.05)  # Max 5% per trade
        confidence_multiplier = 0.5 + (confidence * 1.5)  # 0.5x to 2.0x based on confidence
        notional_value = base_size * confidence_multiplier
        notional_value = min(notional_value, self.max_trade_usd)

        if action == 1:  # BUY
            # Check buy cooldown
            if not self.check_buy_cooldown(symbol):
                logger.info(f"⏰ Buy cooldown active for {symbol}")
                return False, 0.0

            # Buying power check
            if notional_value > buying_power * 0.95:  # Leave 5% buffer
                logger.warning(f"💰 TRADE REJECTED: Insufficient buying power")
                return False, 0.0

            # Position concentration check
            max_position_size = current_equity * self.max_position_pct
            if notional_value > max_position_size:
                notional_value = max_position_size
                logger.info(f"📏 Position size limited to ${notional_value:,.2f}")

        # Approve trade and update tracking
        today = datetime.now().date()
        self.daily_trades[today] += 1

        if action == 1:
            self.position_entry_times[symbol] = datetime.now()

        logger.info(
            f"✅ Trade approved: {symbol} {['HOLD', 'BUY', 'SELL'][action]} ${notional_value:,.2f} (confidence: {confidence:.2f})")
        return True, notional_value

    def get_risk_status(self) -> dict:
        """Get current risk management status"""
        today = datetime.now().date()
        current_equity = self.broker.get_equity()

        return {
            "daily_trades_used": self.daily_trades[today],
            "daily_trades_remaining": self.max_daily_trades - self.daily_trades[today],
            "current_equity": current_equity,
            "drawdown_pct": ((self.start_of_session_equity - current_equity) / self.start_of_session_equity) * 100,
            "kill_switch_active": current_equity < self.kill_switch_threshold
        }
