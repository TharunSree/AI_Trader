# src/sessions/training_session.py

import logging
import time
from pathlib import Path

from src.data.preprocessor import calculate_features
from src.core.environment import TradingEnv
from src.models.ppo_agent import PPOAgent
from src.models.trainer import Trainer
from src.data.yfinance_loader import YFinanceLoader

logger = logging.getLogger('rl_trading_backend')


class TrainingSession:
    """Encapsulates the entire process for a single training run."""

    def __init__(self, config: dict):
        self.config = config

    def run(self, progress_callback=None):
        """Executes the training session."""

        # 1. Load Data
        logger.info("Loading training data...")
        loader = YFinanceLoader(
            tickers=[self.config['ticker']],
            start_date=self.config['start_date'],
            end_date=self.config['end_date']
        )
        raw_df = loader.load_data()
        if raw_df.empty:
            raise ValueError("Data loading failed, DataFrame is empty.")

        featured_df = calculate_features(raw_df)

        # 2. Set up Environment
        env = TradingEnv(
            df=featured_df,
            observation_columns=self.config['observation_columns'],
            window_size=self.config.get('window_size', 10),
            initial_cash=self.config['initial_cash'],
            transaction_cost_pct=0.001,
            slippage_pct=0.0005
        )

        # 3. Set up Agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, lr=0.001)

        # 4. Set up and run Trainer
        trainer_config = {
            "num_episodes": self.config['num_episodes'],
            "target_equity": self.config['target_equity'],
            "patience_episodes": 50
        }
        trainer = Trainer(agent, env, trainer_config)

        # Pass the progress callback to the trainer
        result = trainer.train(progress_callback=progress_callback)

        # 5. Save the final model
        model_name = f"agent_{self.config['ticker']}_{int(time.time())}.pth"
        save_path = Path(f"saved_models/{model_name}")
        agent.save(save_path)
        logger.info(f"Training session complete. Result: {result}. Model saved to {save_path}")

        return result