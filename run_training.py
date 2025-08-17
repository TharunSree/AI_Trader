# run_training.py

import logging
from pathlib import Path
from src.data.preprocessor import calculate_features
from src.core.environment import TradingEnv
from src.models.ppo_agent import PPOAgent
from src.models.trainer import Trainer
from src.utils.logger import setup_logging
from src.data.yfinance_loader import YFinanceLoader


def main():
    """Main function to run the training pipeline."""
    setup_logging()
    log = logging.getLogger("rl_trading_backend")

    TICKER = ["SPY"]
    START_DATE = "2015-01-01"
    END_DATE = "2023-12-31"

    log.info(
        f"Starting PERFECT EYES training for {TICKER[0]} from {START_DATE} to {END_DATE}"
    )

    loader = YFinanceLoader(tickers=TICKER, start_date=START_DATE, end_date=END_DATE)
    raw_df = loader.load_data()

    if raw_df.empty:
        return

    featured_df = calculate_features(raw_df)

    # --- THIS IS THE MOST IMPORTANT CHANGE ---
    # We must update the list of columns the agent will "see".
    OBSERVATION_COLUMNS = [
        "returns",
        "SMA_50",
        "RSI_14",
        "STOCHk_14_3_3",
        "MACDh_12_26_9",
        "ADX_14",
        "BBP_20_2",
        "ATR_14",
        "OBV",
    ]

    env = TradingEnv(
        df=featured_df,
        observation_columns=OBSERVATION_COLUMNS,
        window_size=10,  # Increased window size to capture more complex patterns
        initial_cash=100_000,
        transaction_cost_pct=0.001,
        slippage_pct=0.0005,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, lr=0.001)

    trainer = Trainer(agent=agent, env=env, num_episodes=50, gamma=0.99)

    trainer.train()

    agent.save(Path("saved_models/ppo_agent_perfect_eyes.pth"))
    log.info(
        "PERFECT EYES agent model saved to saved_models/ppo_agent_perfect_eyes.pth"
    )


if __name__ == "__main__":
    main()
