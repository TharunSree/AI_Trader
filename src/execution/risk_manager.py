# src/execution/risk_manager.py

import logging
from src.execution.broker import Broker

logger = logging.getLogger('rl_trading_backend')


class RiskManager:
    def __init__(self, broker: Broker, base_trade_usd: float = 5.00, max_trade_usd: float = 20.00):
        self.broker = broker
        self.base_trade_usd = base_trade_usd
        self.max_trade_usd = max_trade_usd
        logger.info(
            f"RiskManager initialized for dynamic sizing between ${base_trade_usd:.2f} and ${max_trade_usd:.2f}")

    def check_trade(self, symbol: str, action: int, confidence: float = 0.5) -> (bool, float):
        """
        Performs pre-trade risk checks with dynamic sizing based on confidence.
        Confidence is the probability of the chosen action.
        """
        if action == 0:  # HOLD
            return False, 0.0

        if action == 1:  # BUY
            # Dynamic position sizing
            raw_trade_size = self.base_trade_usd + (self.max_trade_usd - self.base_trade_usd) * confidence

            # --- THIS IS THE FIX ---
            # Round the calculated trade size to 2 decimal places for currency.
            trade_size = round(raw_trade_size, 2)

            if self.broker.get_buying_power() < trade_size:
                logger.warning("Not enough buying power for dynamically sized order.")
                return False, 0.0

            logger.info(f"BUY action approved by Risk Manager for ${trade_size:.2f} (Confidence: {confidence:.2f}).")
            return True, trade_size

        elif action == 2:  # SELL
            position_value = self.broker.get_position_value(symbol)
            if position_value < 1.0:
                return False, 0.0

            # Alpaca position value is already rounded, but it's good practice to be sure.
            trade_size = round(position_value, 2)

            logger.info(f"SELL action approved for entire position of ${trade_size:.2f}.")
            return True, trade_size

        return False, 0.0