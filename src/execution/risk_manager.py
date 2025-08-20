import logging
from src.execution.broker import Broker

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Manages risk based on live data from the broker.
    """

    def __init__(self, broker: Broker, base_trade_usd: float = 50.0, max_trade_usd: float = 200.0,
                 max_drawdown_pct: float = 0.15, max_position_pct: float = 0.20):  # Lowered from 0.25 to 0.20
        self.broker = broker
        self.base_trade_usd = base_trade_usd
        self.max_trade_usd = max_trade_usd
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_pct = max_position_pct

        try:
            self.start_of_session_equity = self.broker.get_equity()
            if self.start_of_session_equity == 0:
                raise ValueError("Could not fetch starting equity from broker.")
            self.kill_switch_threshold = self.start_of_session_equity * (1 - self.max_drawdown_pct)
            logger.info(f"RiskManager initialized with starting equity of ${self.start_of_session_equity:,.2f}.")
            logger.info(f"Kill switch drawdown threshold set at ${self.kill_switch_threshold:,.2f}.")
        except Exception as e:
            logger.error(f"FATAL: Could not initialize RiskManager. Error: {e}")
            raise

    def check_trade(self, symbol: str, action: int, confidence: float) -> tuple[bool, float]:
        if action not in [1, 2]:
            return False, 0.0

        current_equity = self.broker.get_equity()
        buying_power = self.broker.get_buying_power()

        if action == 1 and current_equity < self.kill_switch_threshold:
            logger.critical(f"TRADE REJECTED (KILL SWITCH): Equity ${current_equity:,.2f} is below threshold.")
            return False, 0.0

        notional_value = self.base_trade_usd + (self.max_trade_usd - self.base_trade_usd) * confidence
        notional_value = min(notional_value, self.max_trade_usd)

        if action == 1:
            if notional_value > buying_power:
                logger.warning(
                    f"TRADE REJECTED (FUNDS): Value ${notional_value:,.2f} exceeds buying power ${buying_power:,.2f}.")
                return False, 0.0

            max_position_size = current_equity * self.max_position_pct
            if notional_value > max_position_size:
                logger.warning(
                    f"TRADE REJECTED (CONCENTRATION): Value ${notional_value:,.2f} exceeds max position size ${max_position_size:,.2f}.")
                return False, 0.0

        logger.info(
            f"Trade approved for {symbol} ({['HOLD', 'BUY', 'SELL'][action]}) with notional value ${notional_value:,.2f}")
        return True, notional_value
