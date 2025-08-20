import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from src.core.portfolio import Portfolio
import random


class TradingEnv(gym.Env):
    """
    A stock trading environment for reinforcement learning.
    This version includes enhanced reward shaping to encourage profitable selling.
    """

    def __init__(self, df, observation_columns, window_size, initial_cash, transaction_cost_pct, slippage_pct):
        super(TradingEnv, self).__init__()

        self.df = df
        self.observation_columns = observation_columns
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct

        self.action_space = spaces.Discrete(3)  # 0:Hold, 1:Buy, 2:Sell

        num_features = len(self.observation_columns)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size * num_features,),
            dtype=np.float32,
        )
        self.portfolio = None
        self.current_step = 0
        self.trade_log = []

    def reset(self, seed=None, options=None, start_at_beginning=False):
        """
        Resets the environment.
        Args:
            start_at_beginning (bool): If True, starts from the first possible step.
                                       If False, starts at a random step.
        """
        super().reset(seed=seed)

        if start_at_beginning:
            self.current_step = self.window_size
        else:
            self.current_step = random.randint(self.window_size, len(self.df) - 2)

        self.portfolio = Portfolio(self.initial_cash, self.transaction_cost_pct)
        self.trade_log = []
        return self._get_observation(), self.get_info()

    def step(self, action):
        """Execute one time step within the environment."""
        realized_pnl = self._execute_trade(action)
        self.current_step += 1

        current_prices = self._get_current_prices()
        current_equity = self.portfolio.get_equity(current_prices)
        step_reward = 0

        # Enhanced Reward Shaping
        if realized_pnl > 0:
            step_reward += realized_pnl * 1.5
        elif realized_pnl < 0:
            step_reward += realized_pnl * 2.0

        unrealized_pnl = 0
        if "SPY" in self.portfolio.positions:
            entry_price = self.portfolio.positions["SPY"]["entry_price"]
            quantity = self.portfolio.positions["SPY"]["quantity"]
            current_price = current_prices.get("SPY", 0)
            unrealized_pnl = (current_price - entry_price) * quantity
            if unrealized_pnl < 0:
                step_reward += unrealized_pnl * 0.1

        if action == 0 and not self.portfolio.positions:
            step_reward -= self.initial_cash * 1e-7

        done = current_equity <= self.initial_cash * 0.5 or self.current_step >= len(self.df) - 1

        return self._get_observation(), step_reward, done, False, self.get_info()

    def get_info(self):
        """Returns a dictionary of the current state of the environment."""
        return {
            "timestamp": self.df['timestamp'].iloc[self.current_step],
            "equity": self.portfolio.get_equity(self._get_current_prices()),
            "trade_log": self.trade_log,
        }

    def _get_observation(self):
        """Get the observation for the current time step."""
        start = self.current_step - self.window_size
        end = self.current_step
        obs_df = self.df.iloc[start:end][self.observation_columns]
        return obs_df.values.flatten().astype(np.float32)

    def _get_current_prices(self):
        """Get the current price from the dataframe."""
        return {"SPY": self.df["Close"].iloc[self.current_step].item()}

    def _execute_trade(self, action):
        """Executes a trade based on the chosen action."""
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
                        {'timestamp': timestamp, 'action': 'BUY', 'price': buy_price, 'quantity': quantity})
        elif action == 2:  # Sell
            if symbol in self.portfolio.positions:
                quantity = self.portfolio.positions[symbol]["quantity"]
                entry_price = self.portfolio.positions[symbol]["entry_price"]
                realized_pnl = (sell_price - entry_price) * quantity
                self.portfolio.sell(symbol, quantity, sell_price)
                self.trade_log.append(
                    {'timestamp': timestamp, 'action': 'SELL', 'price': sell_price, 'quantity': quantity})

        return realized_pnl
