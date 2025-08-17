# run_backtest.py

from pathlib import Path
import numpy as np
from src.data.csv_loader import CSVLoader
from src.data.preprocessor import calculate_features
from src.core.environment import TradingEnv
from src.core.engine import BacktestEngine
from src.utils.logger import setup_logging
import logging


# This is a placeholder agent. We will build a real RL agent in the next stage.
class RandomAgent:
    """A dummy agent that takes random actions."""

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation):
        """Returns a random action."""
        return self.action_space.sample()


def main():
    """Main function to run the backtest."""
    setup_logging()

    # --- THIS IS THE CORRECTED LINE ---
    log = logging.getLogger("rl_trading_backend")

    log.info("Setting up backtest...")

    # 1. Load and process data
    loader = CSVLoader(csv_path=Path("data/sample_spy.csv"))
    raw_df = loader.load_data()
    featured_df = calculate_features(raw_df)

    # 2. Set up the environment
    OBSERVATION_COLUMNS = ["Close", "Volume", "returns", "SMA_5"]
    env = TradingEnv(
        df=featured_df,
        observation_columns=OBSERVATION_COLUMNS,
        window_size=5,  # Increased window_size to match SMA_5
        initial_cash=100_000,
        transaction_cost_pct=0.001,
        slippage_pct=0.0005,
    )

    # 3. Set up the agent
    agent = RandomAgent(env.action_space)

    # 4. Run the backtest engine
    engine = BacktestEngine(agent=agent, environment=env)
    report = engine.run()

    log.info("Backtest complete. Final report generated.")


if __name__ == "__main__":
    main()
