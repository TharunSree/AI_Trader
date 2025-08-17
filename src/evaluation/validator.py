# src/evaluation/validator.py

import numpy as np
import torch
from src.core.environment import TradingEnv
from src.models.ppo_agent import PPOAgent


class Validator:
    def __init__(self, validation_df, env_config):
        self.validation_df = validation_df
        self.env_config = env_config

    def evaluate(self, agent: PPOAgent) -> dict:
        """
        Evaluates the agent on the validation dataset.
        Returns a dictionary of performance metrics.
        """
        # Create a new environment with the validation data
        env = TradingEnv(
            df=self.validation_df,
            observation_columns=self.env_config["features"],
            window_size=self.env_config["window"],
            initial_cash=100_000,
            transaction_cost_pct=0.001,
            slippage_pct=0.0005,
        )

        obs, _ = env.reset()
        done = False
        daily_equity = [env.initial_cash]

        while not done:
            state = torch.FloatTensor(obs).to(agent.device)
            with torch.no_grad():
                action_probs = agent.actor(state)
                action = torch.argmax(action_probs).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            daily_equity.append(env.portfolio.equity)
            done = terminated or truncated

        # --- Calculate Performance Metrics ---
        daily_returns = (pd.Series(daily_equity).pct_change()).dropna()

        total_return_pct = (daily_equity[-1] / daily_equity[0] - 1) * 100

        # Sharpe Ratio: Measures risk-adjusted return. Higher is better.
        # Assumes 252 trading days in a year.
        if daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        return {"total_return_pct": total_return_pct, "sharpe_ratio": sharpe_ratio}
