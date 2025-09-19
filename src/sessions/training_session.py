import logging
from pathlib import Path
import django
import os

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trader_project.settings')
django.setup()

from src.data.yfinance_loader import YFinanceLoader
from src.data.preprocessor import calculate_features
from src.core.environment import TradingEnv
from src.models.ppo_agent import PPOAgent
from src.models.trainer import Trainer
from src.strategies import STRATEGY_PLAYBOOK  # Import the playbook

logger = logging.getLogger('rl_trading_backend')


class TrainingSession:
    """Encapsulates the entire process for a single training run."""

    def __init__(self, config: dict):
        self.base_config = config
        self.strategy_config = self._load_strategy(config.get('strategy'))

        # Merge the two configs, with strategy-specific values overriding base values
        self.config = {**self.base_config, **self.strategy_config}

        logger.info(f"Initializing training session with final merged config: {self.config}")

    def _load_strategy(self, strategy_name: str) -> dict:
        """
        Loads the feature set and hyperparameters for a given strategy name.
        """
        if not strategy_name:
            raise ValueError("A strategy name must be provided in the config.")

        logger.info(f"Loading configuration for strategy: {strategy_name}")

        # Find the feature set and hyperparameters based on the name
        feature_set_name = None
        hyperparams_name = None

        for key, features in STRATEGY_PLAYBOOK["feature_sets"].items():
            if key == strategy_name:
                feature_set_name = key
                break

        for key, params in STRATEGY_PLAYBOOK["hyperparameters"].items():
            if key == strategy_name:
                hyperparams_name = key
                break

        if not feature_set_name or not hyperparams_name:
            raise ValueError(f"Strategy '{strategy_name}' not found in the STRATEGY_PLAYBOOK.")

        return {
            "features": STRATEGY_PLAYBOOK["feature_sets"][feature_set_name],
            "params": STRATEGY_PLAYBOOK["hyperparameters"][hyperparams_name],
            "window": STRATEGY_PLAYBOOK.get("windows", {}).get(feature_set_name, 10)  # Default window
        }

    def run(self, progress_callback=None):
        """Executes the training session and returns the trained agent."""

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

        # Set up Environment
        env = TradingEnv(
            df=featured_df,
            observation_columns=self.config['features'],  # This will now work
            window_size=self.config.get('window', 10),
            initial_cash=self.config['initial_cash'],
        )

        # Set up Agent
        state_dim = len(self.config['features']) * self.config['window']
        action_dim = env.action_space.n
        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, lr=self.config['params']['lr'])

        # Set up and run Trainer
        trainer_config = {
            "num_episodes": self.config.get('num_episodes', 500),
            "gamma": self.config['params']['gamma'],
        }
        trainer = Trainer(agent, env, trainer_config)

        trained_agent = trainer.run(progress_callback=progress_callback)

        return trained_agent