# src/core/engine.py

import pandas as pd
from src.core.environment import TradingEnv
from src.core.performance import (
    calculate_cagr,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
)
import logging

logger = logging.getLogger("rl_trading_backend")


class BacktestEngine:
    """
    Orchestrates the backtesting process by running the agent through the environment
    and generating performance reports.
    """

    def __init__(self, agent, environment: TradingEnv):
        self.agent = agent
        self.env = environment
        self.history = []

    def run(self, start_at_beginning: bool = False) -> dict:
        """
        Runs the backtest from start to finish.
        """
        logger.info("Starting backtest...")
        obs, info = self.env.reset(start_at_beginning=start_at_beginning)

        self.history.append(info)

        while True:
            action = self.agent.predict(obs)  # The agent decides on an action
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.history.append(info)

            if terminated or truncated:
                logger.info("Backtest finished.")
                break

        return self.generate_report()

    def generate_report(self) -> dict:
        """
        Generates a performance report from the backtest history.
        """
        # FIX: Check if history has more than one entry (initial state + at least one step)
        if len(self.history) < 2:
            logger.error("Backtest did not run long enough to generate a report. History is empty or contains only the initial state.")
            # Return a default/error report
            return {
                "start_equity": 0,
                "end_equity": 0,
                "total_return_pct": 0,
                "cagr": 0,
                "sharpe_ratio": 0,
                "max_drawdown_pct": 0,
                "trade_count": 0,
                "error": "Backtest failed to produce sufficient history for a report."
            }


        equity_curve = pd.Series(
            [info["equity"] for info in self.history],
            index=[info["timestamp"] for info in self.history],
        )

        report = {
            "start_equity": equity_curve.iloc[0],
            "end_equity": equity_curve.iloc[-1],
            "total_return_pct": (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
            * 100,
            "cagr": calculate_cagr(equity_curve) * 100,
            "sharpe_ratio": calculate_sharpe_ratio(equity_curve),
            "max_drawdown_pct": calculate_max_drawdown(equity_curve) * 100,
            "trade_count": len(self.history[-1]["trade_log"]),
        }

        logger.info(f"--- Backtest Report ---")
        for key, value in report.items():
            if isinstance(value, float):
                logger.info(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                 logger.info(f"{key.replace('_', ' ').title()}: {value}")


        return report