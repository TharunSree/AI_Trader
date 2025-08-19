# src/core/environment.py
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
            low=-np.inf, high=np.inf,
            shape=(window_size * num_features,), dtype=np.float32,
        )

        self.portfolio = None
        self.current_step = 0
        self.trade_log = []

    def reset(self, seed=None, options=None, start_at_beginning=False):
        super().reset(seed=seed)
        if start_at_beginning:
            self.current_step = self.window_size
        else:
            # Start at a random point in the first 80% of the data to ensure test data remains unseen
            max_start = int((len(self.df) - 2) * 0.8)
            self.current_step = random.randint(self.window_size, max_start)

        self.portfolio = Portfolio(self.initial_cash, self.transaction_cost_pct)
        self.trade_log = []
        return self._get_observation(), {}

    def step(self, action):
        prev_equity = self.portfolio.get_equity(self._get_current_prices())
        trade_profit_loss = self._execute_trade(action)
        self.current_step += 1

        current_equity = self.portfolio.get_equity(self._get_current_prices())
        step_reward = current_equity - prev_equity

        # --- REWARD SHAPING ---
        # 1. Heavily reward realized profits
        if trade_profit_loss > 0:
            step_reward += trade_profit_loss * 0.5  # Add 50% of realized profit to reward

        # 2. Penalize realized losses more
        elif trade_profit_loss < 0:
            step_reward += trade_profit_loss * 1.5  # Penalize 150% of realized loss

        # 3. Small penalty for holding cash and doing nothing
        if action == 0 and not self.portfolio.positions:
            step_reward -= self.initial_cash * 1e-7

        # 4. Include unrealized P&L to give a continuous signal
        unrealized_pnl = 0
        if self.portfolio.positions:
            current_price = self._get_current_prices().get("SPY", 0)
            entry_price = self.portfolio.positions["SPY"]["entry_price"]
            quantity = self.portfolio.positions["SPY"]["quantity"]
            unrealized_pnl = (current_price - entry_price) * quantity

        step_reward += unrealized_pnl * 0.05  # Add 5% of unrealized P&L to reward

        done = current_equity <= self.initial_cash * 0.5 or self.current_step >= len(self.df) - 1

        return self._get_observation(), step_reward, done, False, {}

    def _get_observation(self):
        start = self.current_step - self.window_size
        end = self.current_step
        obs_df = self.df.iloc[start:end][self.observation_columns]
        return obs_df.values.flatten().astype(np.float32)

    def _get_current_prices(self):
        prices = {"SPY": self.df["Close"].iloc[self.current_step].item()}
        return prices

    def _execute_trade(self, action):
        symbol = "SPY"
        current_price = self.df["Close"].iloc[self.current_step].item()
        timestamp = self.df.index[self.current_step]

        price_with_slippage = current_price
        if action == 1:  # Buy
            price_with_slippage *= (1 + self.slippage_pct)
        elif action == 2:  # Sell
            price_with_slippage *= (1 - self.slippage_pct)

        if action == 1:  # Buy
            # Buy with 95% of available cash
            trade_value = self.portfolio.cash * 0.95
            if trade_value > 10:  # Minimum trade
                quantity = trade_value / price_with_slippage
                self.portfolio.buy(symbol, quantity, price_with_slippage)
                self.trade_log.append(
                    {'timestamp': timestamp, 'action': 'BUY', 'quantity': quantity, 'price': price_with_slippage})
            return 0

        elif action == 2:  # Sell
            if symbol in self.portfolio.positions:
                quantity = self.portfolio.positions[symbol]["quantity"]
                entry_price = self.portfolio.positions[symbol]["entry_price"]
                profit_loss = (price_with_slippage - entry_price) * quantity
                self.portfolio.sell(symbol, quantity, price_with_slippage)
                self.trade_log.append(
                    {'timestamp': timestamp, 'action': 'SELL', 'quantity': quantity, 'price': price_with_slippage})
                return profit_loss
        return 0
