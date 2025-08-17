# src/execution/risk_manager.py

import logging
from src.execution.broker import Broker

logger = logging.getLogger("rl_trading_backend")


class RiskManager:
    def __init__(
        self, broker: Broker, base_trade_usd: float = 5.00, max_trade_usd: float = 20.00
    ):
        self.broker = broker
        self.base_trade_usd = base_trade_usd
        self.max_trade_usd = max_trade_usd
        logger.info(
            f"RiskManager initialized for dynamic sizing between ${base_trade_usd:.2f} and ${max_trade_usd:.2f}"
        )

    def check_trade(
        self, symbol: str, action: int, confidence: float = 0.5
    ) -> (bool, float):
        if action == 0:  # HOLD
            return False, 0.0

        if action == 1:  # BUY
            trade_size = (
                self.base_trade_usd
                + (self.max_trade_usd - self.base_trade_usd) * confidence
            )

            if self.broker.get_buying_power() < trade_size:
                logger.warning("Not enough buying power for dynamically sized order.")
                return False, 0.0

            logger.info(
                f"BUY action approved by Risk Manager for ${trade_size:.2f} (Confidence: {confidence:.2f})."
            )
            return True, trade_size

        elif action == 2:  # SELL
            position_value = self.broker.get_position_value(symbol)
            if position_value < 1.0:
                return False, 0.0

            logger.info(
                f"SELL action approved for entire position of ${position_value:.2f}."
            )
            return True, position_value

        return False, 0.0
