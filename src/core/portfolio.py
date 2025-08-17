# src/core/portfolio.py

import logging

logger = logging.getLogger("rl_trading_backend")


class Portfolio:
    def __init__(
        self, initial_cash: float = 100_000, transaction_cost_pct: float = 0.001
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        # --- UPDATED: Position dictionary now stores more detail ---
        self.positions = {}  # Holds {'quantity': X, 'entry_price': Y}
        self.transaction_cost_pct = transaction_cost_pct
        self.equity = initial_cash

    def buy(self, symbol: str, quantity: float, price: float):
        trade_value = quantity * price
        cost = trade_value * self.transaction_cost_pct
        total_cost = trade_value + cost

        if self.cash >= total_cost:
            self.cash -= total_cost

            # Update position by averaging the entry price
            if symbol in self.positions:
                current_qty = self.positions[symbol]["quantity"]
                current_value = current_qty * self.positions[symbol]["entry_price"]

                new_total_qty = current_qty + quantity
                new_total_value = current_value + trade_value

                self.positions[symbol]["entry_price"] = new_total_value / new_total_qty
                self.positions[symbol]["quantity"] = new_total_qty
            else:
                # Add new position
                self.positions[symbol] = {"quantity": quantity, "entry_price": price}

    def sell(self, symbol: str, quantity: float, price: float):
        if symbol in self.positions and self.positions[symbol]["quantity"] >= quantity:
            trade_value = quantity * price
            cost = trade_value * self.transaction_cost_pct
            total_proceeds = trade_value - cost

            self.cash += total_proceeds
            self.positions[symbol]["quantity"] -= quantity

            # Remove position if all shares are sold
            if (
                self.positions[symbol]["quantity"] < 1e-9
            ):  # Use a small threshold for float precision
                del self.positions[symbol]

    def get_equity(self, current_prices: dict) -> float:
        positions_value = 0
        for symbol, data in self.positions.items():
            positions_value += data["quantity"] * current_prices.get(symbol, 0)

        self.equity = self.cash + positions_value
        return self.equity
