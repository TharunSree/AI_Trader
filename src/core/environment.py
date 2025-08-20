import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from src.core.portfolio import Portfolio
import random

class TradingEnv(gym.Env):
    """
    Actions: 0=Hold, 1=Enter/Increase Long (if flat buy 95% cash), 2=Close Long (if any)
    Single-asset simplified environment.
    """
    metadata = {"render.modes": []}

    def __init__(self, df, observation_columns, window_size, initial_cash,
                 transaction_cost_pct=0.001, slippage_pct=0.0005,
                 reward_clip=5.0):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.observation_columns = observation_columns
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.reward_clip = reward_clip

        num_features = len(observation_columns)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size * num_features,), dtype=np.float32
        )

        self.portfolio = None
        self.current_step = 0
        self.trade_log = []
        self._peak_equity = initial_cash
        self._last_equity = initial_cash

    def reset(self, seed=None, options=None, start_at_beginning=False):
        super().reset(seed=seed)
        if start_at_beginning:
            self.current_step = self.window_size
        else:
            self.current_step = random.randint(self.window_size, len(self.df) - 2)
        self.portfolio = Portfolio(self.initial_cash, self.transaction_cost_pct)
        self.trade_log = []
        self._peak_equity = self.initial_cash
        self._last_equity = self.initial_cash
        obs = self._get_observation()
        return obs, self.get_info()

    def step(self, action):
        realized_pnl = self._execute_trade(action)
        self.current_step += 1

        prices = self._get_current_prices()
        equity = self.portfolio.get_equity(prices)

        # --- Reward Shaping ---
        reward = 0.0
        norm = self.initial_cash

        # Realized PnL (scaled & asymmetric)
        if realized_pnl > 0:
            reward += 4.0 * (realized_pnl / norm)
        elif realized_pnl < 0:
            reward += 2.0 * (realized_pnl / norm)  # realized_pnl negative

        # Equity high bonus (discourages premature small profit exits)
        if equity > self._peak_equity:
            reward += 0.3 * ((equity - self._peak_equity) / norm)
            self._peak_equity = equity

        # Mild penalty for equity drawdown step-to-step (stability focus)
        equity_delta = (equity - self._last_equity) / norm
        if equity_delta < 0:
            reward += 0.05 * equity_delta
        self._last_equity = equity

        # Position-based shaping
        if "SPY" in self.portfolio.positions:
            pos = self.portfolio.positions["SPY"]
            entry_price = pos["entry_price"]
            qty = pos["quantity"]
            cur_price = prices["SPY"]
            unrealized = (cur_price - entry_price) * qty

            # Time cost while capital tied
            reward -= 0.00002

            # Unrealized losses penalized
            if unrealized < 0:
                reward += 0.4 * (unrealized / norm)
            else:
                # Small positive shaping for holding winning trades
                reward += 0.05 * (unrealized / norm)

            # Soft stop-loss: force close if loss exceeds 10%
            if unrealized / norm < -0.10:
                self._force_close(prices)

        else:
            # Idle cash penalty (encourage deployment)
            reward -= 0.00001

        # Clip extreme reward spikes
        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        done = (
            equity <= self.initial_cash * 0.4 or
            self.current_step >= len(self.df) - 1
        )
        truncated = False

        return self._get_observation(), reward, done, truncated, self.get_info()

    def get_info(self):
        return {
            "timestamp": self.df['timestamp'].iloc[self.current_step],
            "equity": self.portfolio.get_equity(self._get_current_prices()),
            "trade_log": self.trade_log,
        }

    def _get_observation(self):
        start = self.current_step - self.window_size
        window_df = self.df.iloc[start:self.current_step][self.observation_columns]
        return window_df.values.flatten().astype(np.float32)

    def _get_current_prices(self):
        return {"SPY": float(self.df["Close"].iloc[self.current_step])}

    def _execute_trade(self, action):
        symbol = "SPY"
        price = float(self.df["Close"].iloc[self.current_step])
        ts = self.df['timestamp'].iloc[self.current_step]
        buy_price = price * (1 + self.slippage_pct)
        sell_price = price * (1 - self.slippage_pct)
        realized = 0.0

        if action == 1:
            if symbol not in self.portfolio.positions:
                capital = self.portfolio.cash * 0.95
                if capital > 10:
                    qty = capital / buy_price
                    self.portfolio.buy(symbol, qty, buy_price)
                    self.trade_log.append({
                        "timestamp": ts, "action": "BUY", "price": buy_price,
                        "quantity": qty, "symbol": symbol
                    })
        elif action == 2:
            if symbol in self.portfolio.positions:
                qty = self.portfolio.positions[symbol]["quantity"]
                entry_price = self.portfolio.positions[symbol]["entry_price"]
                realized = (sell_price - entry_price) * qty
                self.portfolio.sell(symbol, qty, sell_price)
                self.trade_log.append({
                    "timestamp": ts, "action": "SELL", "price": sell_price,
                    "quantity": qty, "symbol": symbol
                })
        return realized

    def _force_close(self, prices):
        symbol = "SPY"
        if symbol in self.portfolio.positions:
            qty = self.portfolio.positions[symbol]["quantity"]
            sell_price = prices[symbol] * (1 - self.slippage_pct)
            ts = self.df['timestamp'].iloc[self.current_step]
            self.portfolio.sell(symbol, qty, sell_price)
            self.trade_log.append({
                "timestamp": ts, "action": "SELL", "price": sell_price,
                "quantity": qty, "symbol": symbol, "forced": True
            })