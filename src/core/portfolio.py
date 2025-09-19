# src/core/portfolio.py

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("rl_trading_backend")


class Portfolio:
    """
    Manages the state of a trading account, including cash, positions, and equity.
    """

    def __init__(self, initial_cash=100000.0, transaction_cost=0.001):
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost

        self.cash = initial_cash
        self.positions = np.zeros(1)  # Assuming single asset for now
        self.holdings_value = 0.0
        self.total_equity = initial_cash
        self.trade_log = []

    def reset(self):
        """ Resets the portfolio to its initial state. """
        self.cash = self.initial_cash
        self.positions.fill(0)
        self.holdings_value = 0.0
        self.total_equity = self.initial_cash
        self.trade_log = []

    def update(self, current_prices):
        """ Updates the value of holdings and total equity based on new prices. """
        if not isinstance(current_prices, np.ndarray):
            current_prices = np.array([current_prices])

        self.holdings_value = (self.positions * current_prices).sum()
        self.total_equity = self.cash + self.holdings_value

    def get_total_equity(self) -> float:
        """
        FIX: Ensures the returned equity is always a float, not a tensor.
        """
        if hasattr(self.total_equity, 'item'):
            # If it's a tensor, extract the scalar value
            return self.total_equity.item()
        return float(self.total_equity)

    def execute_trade(self, action: int, quantity: int, price: float):
        """
        Executes a trade and updates portfolio state.
        Action: 0=HOLD, 1=BUY, 2=SELL
        """
        if action == 1:  # BUY
            cost = quantity * price
            commission = cost * self.transaction_cost
            total_cost = cost + commission

            if self.cash >= total_cost:
                self.cash -= total_cost
                self.positions[0] += quantity
                self._log_trade("BUY", quantity, price, commission)
            else:
                logger.warning("Not enough cash to execute buy order.")

        elif action == 2:  # SELL
            if self.positions[0] >= quantity:
                proceeds = quantity * price
                commission = proceeds * self.transaction_cost
                total_proceeds = proceeds - commission

                self.cash += total_proceeds
                self.positions[0] -= quantity
                self._log_trade("SELL", quantity, price, commission)
            else:
                logger.warning("Not enough shares to execute sell order.")

        self.update(price)

    def _log_trade(self, side, quantity, price, commission):
        self.trade_log.append({
            'side': side,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'equity': self.total_equity
        })