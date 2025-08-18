import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from src.core.portfolio import Portfolio
import random


class TradingEnv(gym.Env):
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
        super().reset(seed=seed)
        if start_at_beginning:
            self.current_step = self.window_size
        else:
            self.current_step = random.randint(self.window_size, len(self.df) - 2)

        self.portfolio = Portfolio(self.initial_cash, self.transaction_cost_pct)
        self.trade_log = []
        return self._get_observation(), {}

    def step(self, action):
        prev_equity = self.portfolio.get_equity(self._get_current_prices())

        trade_profit_loss = self._execute_trade(action)

        self.current_step += 1

        current_equity = self.portfolio.get_equity(self._get_current_prices())
        step_reward = current_equity - prev_equity

        if trade_profit_loss > 0:
            step_reward += trade_profit_loss  # Add realized profit to reward
        elif trade_profit_loss < 0:
            step_reward += trade_profit_loss  # Add realized loss (penalty)

        if action == 0 and not self.portfolio.positions:
            step_reward -= self.initial_cash * 1e-5  # Tiny penalty for inaction

        done = current_equity <= 0 or self.current_step >= len(self.df) - 1

        return self._get_observation(), step_reward, done, False, {}

    def _get_observation(self):
        start = self.current_step - self.window_size
        end = self.current_step
        obs_df = self.df.iloc[start:end][self.observation_columns]
        return obs_df.values.flatten().astype(np.float32)

    def _get_current_prices(self):
        prices = {}
        prices["SPY"] = self.df["Close"].iloc[self.current_step].item()
        return prices

    def _execute_trade(self, action):
        symbol = "SPY"
        current_price = self.df["Close"].iloc[self.current_step].item()
        timestamp = self.df.index[self.current_step]

        price_with_slippage = current_price
        if action == 1:  # Buy
            price_with_slippage = current_price * (1 + self.slippage_pct)
        elif action == 2:  # Sell
            price_with_slippage = current_price * (1 - self.slippage_pct)

        if action == 1:  # Buy
            trade_value = self.portfolio.cash * 0.95
            if trade_value > 10:
                quantity = trade_value / price_with_slippage
                self.portfolio.buy(symbol, quantity, price_with_slippage)
                self.trade_log.append({'timestamp': timestamp, 'action': 'BUY', 'symbol': symbol, 'quantity': quantity,
                                       'price': price_with_slippage})
            return 0

        elif action == 2:  # Sell
            if symbol in self.portfolio.positions:
                quantity = self.portfolio.positions[symbol]["quantity"]
                entry_price = self.portfolio.positions[symbol]["entry_price"]
                profit_loss = (price_with_slippage - entry_price) * quantity
                self.portfolio.sell(symbol, quantity, price_with_slippage)
                self.trade_log.append({'timestamp': timestamp, 'action': 'SELL', 'symbol': symbol, 'quantity': quantity,
                                       'price': price_with_slippage})
                return profit_loss

        return 0