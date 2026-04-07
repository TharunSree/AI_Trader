import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple, Any, Dict

logger = logging.getLogger("TheMatrix")

class TradingEnvironment(gym.Env):
    """
    A custom simulation environment for the AI to practice trading.
    Modernized to support 4D State (Price, Dist_from_MA, Volume, Sentiment)
    and robust continuous reward scaling.
    """
    
    metadata = {"render_modes": ["human"]}

    def __init__(
        self, 
        df: pd.DataFrame = None, 
        observation_columns: list = None,
        window_size: int = 1,
        initial_balance: float = 100_000.0,
        fee_rate: float = 0.0015,
        slippage: float = 0.000,
        data_path: str = "data/historical_btcusd.csv",
        **kwargs
    ):
        super().__init__()

        # Handle explicit dataframes (Meta Training) vs legacy csv loading
        if df is not None:
            self.df = df.copy()
            self.df.reset_index(drop=True, inplace=True)
        else:
            self.df = pd.read_csv(data_path)
            
        # Set instance attributes
        if 'initial_cash' in kwargs:
            initial_balance = kwargs['initial_cash']
            
        self.initial_balance = initial_balance
        self.observation_columns = observation_columns if observation_columns else ['Close', 'Volume']
        self.window_size = int(window_size)
        self.fee_rate = float(fee_rate)
        self.slippage = float(slippage)

        # 2. Action Space (Continuous action between -1.0 (Strong Sell) and 1.0 (Strong Buy))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 3. Observation Space based directly on window size and feature columns requested
        obs_dim = len(self.observation_columns) * self.window_size
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.current_step: int = self.window_size
        self.balance: float = initial_balance
        self.crypto_held: float = 0.0
        self.net_worth: float = initial_balance
        
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the simulation to the beginning of the timeline."""
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.crypto_held = 0.0
        self.net_worth = self.initial_balance

        return self._next_observation(), {}

    def _next_observation(self) -> np.ndarray:
        """Returns the fully padded multidimensional array for the neural network input."""
        end_idx = self.current_step + 1
        start_idx = end_idx - self.window_size
        
        # Verify requested columns exist
        cols = [c for c in self.observation_columns if c in self.df.columns]
        
        if len(cols) < len(self.observation_columns):
            # Dynamic missing value padding matrix
            full_window = np.zeros((self.window_size, len(self.observation_columns)))
            for i, col in enumerate(self.observation_columns):
                if col in self.df.columns:
                    full_window[:, i] = self.df[col].iloc[start_idx:end_idx].values
            obs = full_window.flatten()
        else:
            window_data = self.df[cols].iloc[start_idx:end_idx].values
            obs = window_data.flatten()
            
        # Nan stripping for clean tensors
        obs = np.nan_to_num(obs)
        return obs.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advances the simulation by one hour based on the AI's decision."""
        current_price = float(self.df.iloc[self.current_step]['Close'])
        prev_net_worth = self.net_worth
        decision = action[0]

        # 1. Dynamic Trade Sizing based on Confidence (Pseudo-Kelly Criterion)
        # Bounding the max risk to 15% of portfolio even at 1.0 absolute certainty.
        
        # Robust dimensionality squeeze for Tensor batch payloads
        decision_scalar = float(np.squeeze(action))
        
        action_confidence = max(0.0, abs(decision_scalar))
        trade_size_usd = self.net_worth * (action_confidence * 0.15)
        trade_qty = trade_size_usd / current_price

        # 2. Strict Friction Injection (Taker + Slippage Penalty)
        FEE_RATE = self.fee_rate + self.slippage 

        if decision > 0.4:  # BUY
            gross_cost = trade_qty * current_price
            fee = gross_cost * FEE_RATE
            total_deduction = gross_cost + fee
            
            if self.balance >= total_deduction:
                self.balance -= total_deduction
                self.crypto_held += trade_qty
                
        elif decision < -0.4:  # SELL
            if self.crypto_held >= trade_qty:
                gross_revenue = trade_qty * current_price
                fee = gross_revenue * FEE_RATE
                net_revenue = gross_revenue - fee
                
                self.balance += net_revenue
                self.crypto_held -= trade_qty

        # Calculate new net worth (penalized exactly by the spread loss)
        self.net_worth = self.balance + (self.crypto_held * current_price)

        # Reward = Percentage growth of the portfolio relative to original balance 
        # Scaled up for better gradient signal. Keeps rewards stable vs exploding bounds.
        reward = ((self.net_worth - prev_net_worth) / prev_net_worth) * 100.0

        self.current_step += 1
        
        done = self.net_worth <= 0 or self.current_step >= len(self.df) - 1

        info = {
            "net_worth": self.net_worth,
            "balance": self.balance,
            "crypto_held": self.crypto_held
        }

        return self._next_observation(), reward, done, False, info