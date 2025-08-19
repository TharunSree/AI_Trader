# src/execution/risk_manager.py
import logging
from .broker import Broker  # Assuming broker.py is in the same directory

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Manages risk for the trading bot by checking trades and monitoring the portfolio
    based on live data from the broker.
    """

    def __init__(self, broker: Broker, base_trade_usd: float = 50.0, max_trade_usd: float = 200.0,
                 max_drawdown_pct: float = 0.15, max_position_pct: float = 0.25):
        self.broker = broker
        self.base_trade_usd = base_trade_usd
        self.max_trade_usd = max_trade_usd
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_pct = max_position_pct

        try:
            # Set initial equity from the live account at startup
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
        """
        Checks if a trade is approved based on live broker account data.
        Returns (is_approved, notional_value).
        """
        # Action must be BUY (1) or SELL (2)
        if action not in [1, 2]:
            return False, 0.0

        # --- Fetch LIVE account state ---
        current_equity = self.broker.get_equity()
        buying_power = self.broker.get_buying_power()

        # --- RULE 1: Global Kill Switch (for BUY orders only) ---
        if action == 1 and current_equity < self.kill_switch_threshold:
            logger.critical(
                f"TRADE REJECTED (KILL SWITCH): Equity ${current_equity:,.2f} is below total drawdown threshold of ${self.kill_switch_threshold:,.2f}.")
            return False, 0.0

        # --- Calculate Proposed Trade Size ---
        notional_value = self.base_trade_usd + (self.max_trade_usd - self.base_trade_usd) * confidence
        notional_value = min(notional_value, self.max_trade_usd)

        # --- Portfolio-Level Checks (for BUY orders only) ---
        if action == 1:
            # --- RULE 2: Check Available Buying Power ---
            if notional_value > buying_power:
                logger.warning(
                    f"TRADE REJECTED (INSUFFICIENT FUNDS): Proposed trade of ${notional_value:,.2f} for {symbol} exceeds buying power of ${buying_power:,.2f}.")
                return False, 0.0

            # --- RULE 3: Position Concentration Limit ---
            # A single position should not exceed a certain percentage of our total equity.
            max_position_size = current_equity * self.max_position_pct
            if notional_value > max_position_size:
                logger.warning(
                    f"TRADE REJECTED (CONCENTRATION): Proposed trade of ${notional_value:,.2f} for {symbol} exceeds max position size of ${max_position_size:,.2f}.")
                return False, 0.0

        # If all checks pass for a BUY, or if the action is a SELL, the trade is approved.
        logger.info(
            f"Trade approved for {symbol} ({['HOLD', 'BUY', 'SELL'][action]}) with notional value of ${notional_value:,.2f}")
        return True, notional_value