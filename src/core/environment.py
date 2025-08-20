import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from src.core.portfolio import Portfolio
import random


class TradingEnv(gym.Env):
    """
    A stock trading environment for reinforcement learning.
    This version includes AGGRESSIVE reward shaping to force the agent
    to learn both buy and sell signals.
    """

    def __init__(self, df, observation_columns, window_size, initial_cash, transaction_cost_pct, slippage_pct):
        super(TradingEnv, self).__init__()
        self.df = df
        self.observation_columns = observation_columns
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.action_space = spaces.Discrete(3)
        num_features = len(self.observation_columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size * num_features,), dtype=np.float32,
        )
        self.portfolio = None
        self.current_step = 0
        self.trade_log = []

    def reset(self, seed=None, options=None, start_at_beginning=False):
        """Resets the environment."""
        super().reset(seed=seed)
        if start_at_beginning:
            self.current_step = self.window_size
        else:
            self.current_step = random.randint(self.window_size, len(self.df) - 2)
        self.portfolio = Portfolio(self.initial_cash, self.transaction_cost_pct)
        self.trade_log = []
        return self._get_observation(), self.get_info()

    # src/core/environment.py (modified reward shaping section inside step)
    def step(self, action):
        realized_pnl = self._execute_trade(action)
        self.current_step += 1
        current_prices = self._get_current_prices()
        current_equity = self.portfolio.get_equity(current_prices)

        # Initialize reward
        step_reward = 0.0

        # --- Realized PnL shaping (normalized) ---
        if realized_pnl > 0:
            # Strong reward for profitable close
            step_reward += 5.0 * (realized_pnl / self.initial_cash)
        elif realized_pnl < 0:
            # Penalty for realized loss
            step_reward += 3.0 * (realized_pnl / self.initial_cash)  # realized_pnl is negative

        # Track / store previous equity to reward new highs
        if not hasattr(self, "_peak_equity"):
            self._peak_equity = self.initial_cash
        if current_equity > self._peak_equity:
            # Mild bonus for pushing to a new equity high (prevents too-early profit taking)
            equity_gain = (current_equity - self._peak_equity) / self.initial_cash
            step_reward += 0.5 * equity_gain
            self._peak_equity = current_equity

        # Holding penalties / unrealized shaping
        if "SPY" in self.portfolio.positions:
            # Time cost
            step_reward -= 1e-5
            pos = self.portfolio.positions["SPY"]
            entry_price = pos["entry_price"]
            quantity = pos["quantity"]
            current_price = current_prices.get("SPY", 0)
            unrealized_pnl = (current_price - entry_price) * quantity

            # Penalize unrealized losses (scaled)
            if unrealized_pnl < 0:
                step_reward += 0.2 * (unrealized_pnl / self.initial_cash)
            else:
                # Slight positive shaping for unrealized gains (encourage waiting for larger profits)
                step_reward += 0.05 * (unrealized_pnl / self.initial_cash)

        done = current_equity <= self.initial_cash * 0.5 or self.current_step >= len(self.df) - 1
        return self._get_observation(), float(step_reward), done, False, self.get_info()

    def get_info(self):
        """Returns a dictionary of the current state of the environment."""
        return {
            "timestamp": self.df['timestamp'].iloc[self.current_step],
            "equity": self.portfolio.get_equity(self._get_current_prices()),
            "trade_log": self.trade_log,
        }

    def _get_observation(self):
        start = self.current_step - self.window_size
        end = self.current_step
        obs_df = self.df.iloc[start:end][self.observation_columns]
        return obs_df.values.flatten().astype(np.float32)

    def _get_current_prices(self):
        return {"SPY": self.df["Close"].iloc[self.current_step].item()}

    def _execute_trade(self, action):
        symbol = "SPY"
        current_price = self.df["Close"].iloc[self.current_step].item()
        timestamp = self.df['timestamp'].iloc[self.current_step]
        buy_price = current_price * (1 + self.slippage_pct)
        sell_price = current_price * (1 - self.slippage_pct)
        realized_pnl = 0

        if action == 1:  # Buy
            if "SPY" not in self.portfolio.positions:
                trade_value = self.portfolio.cash * 0.95
                if trade_value > 10:
                    quantity = trade_value / buy_price
                    self.portfolio.buy(symbol, quantity, buy_price)
                    self.trade_log.append(
                        {'timestamp': timestamp, 'action': 'BUY', 'price': buy_price, 'quantity': quantity,
                         'symbol': symbol})
        elif action == 2:  # Sell
            if symbol in self.portfolio.positions:
                quantity = self.portfolio.positions[symbol]["quantity"]
                entry_price = self.portfolio.positions[symbol]["entry_price"]
                realized_pnl = (sell_price - entry_price) * quantity
                self.portfolio.sell(symbol, quantity, sell_price)
                self.trade_log.append(
                    {'timestamp': timestamp, 'action': 'SELL', 'price': sell_price, 'quantity': quantity,
                     'symbol': symbol})

        return realized_pnl
