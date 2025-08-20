import logging
from pathlib import Path
import torch
import os
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trader_project.settings')
django.setup()

from src.data.yfinance_loader import YFinanceLoader
from src.data.preprocessor import calculate_features
from src.core.environment import TradingEnv
from src.core.engine import BacktestEngine
from src.models.ppo_agent import PPOAgent

logger = logging.getLogger("rl_trading_backend")


class EvaluationSession:
    """
    Handles the process of evaluating a trained agent over a specific historical period.
    """

    def __init__(self, config: dict):
        self.config = config
        logger.info(f"Initializing evaluation session with config: {self.config}")

    def run(self) -> dict:
        """
        Executes the full evaluation backtest.
        """
        logger.info("--- Starting Evaluation Session ---")

        # 1. Load the trained agent and its configuration
        model_path = Path("saved_models") / self.config['model_file']
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        agent, model_config = PPOAgent.load_with_config(model_path)
        agent.actor.eval()  # Set agent to evaluation mode
        logger.info(f"Loaded model {self.config['model_file']} with training config: {model_config}")

        # 2. Load the historical data for the evaluation period
        logger.info(f"Loading data from {self.config['start_date']} to {self.config['end_date']}...")
        loader = YFinanceLoader(
            ['SPY'],
            start_date=self.config['start_date'],
            end_date=self.config['end_date']
        )
        raw_df = loader.load_data()
        if raw_df.empty:
            raise ValueError("No data loaded for the specified evaluation period.")

        featured_df = calculate_features(raw_df)
        logger.info(f"Data loaded and features calculated. Shape: {featured_df.shape}")

        # 3. Set up the environment using the model's specific parameters
        env = TradingEnv(
            df=featured_df,
            observation_columns=model_config['features'],
            window_size=model_config['window'],
            initial_cash=100_000,  # Standard initial cash for evaluation consistency
            transaction_cost_pct=0.001,
            slippage_pct=0.0005,
        )

        # 4. Run the backtest from the very beginning of the loaded data
        engine = BacktestEngine(agent=agent, environment=env)

        # --- THIS IS THE FIX ---
        # We now explicitly tell the engine to run the test from the start.
        report = engine.run(start_at_beginning=True)

        logger.info("--- Evaluation Session Complete ---")
        return report