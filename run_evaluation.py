# run_evaluation.py
import logging
from pathlib import Path
from src.data.csv_loader import CSVLoader
from src.data.preprocessor import calculate_features
from src.core.environment import TradingEnv
from src.core.engine import BacktestEngine
from src.models.ppo_agent import PPOAgent
from src.utils.logger import setup_logging


def main():
    """Main function to evaluate the trained agent."""
    setup_logging()
    log = logging.getLogger("rl_trading_backend")

    log.info("Setting up evaluation...")

    # 1. Load and process data (same as training)
    loader = CSVLoader(csv_path=Path("data/sample_spy.csv"))
    raw_df = loader.load_data()
    featured_df = calculate_features(raw_df)

    # 2. Set up the environment
    OBSERVATION_COLUMNS = ["Close", "Volume", "returns", "SMA_5"]
    env = TradingEnv(
        df=featured_df,
        observation_columns=OBSERVATION_COLUMNS,
        window_size=5,
        initial_cash=100_000,
        transaction_cost_pct=0.001,
        slippage_pct=0.0005,
    )

    # 3. Set up the AGENT and LOAD THE TRAINED MODEL
    state_dim = len(OBSERVATION_COLUMNS) * 5
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
    model_path = Path("saved_models/ppo_final.pth")

    if not model_path.exists():
        log.error(f"Model file not found at {model_path}. Please run training first.")
        return

    log.info(f"Loading trained model from {model_path}...")
    agent.load(model_path)

    # 4. Run the backtest engine for evaluation
    engine = BacktestEngine(agent=agent, environment=env)
    report = engine.run()

    log.info("Evaluation complete.")


if __name__ == "__main__":
    main()
