# run_backtest.py
import logging
from pathlib import Path

from src.core.environment import TradingEnvironment
from src.core.engine import BacktestEngine
from src.models.ppo_agent import PPOAgent
from src.utils.logger import setup_logging

def main():
    """Main function to run the deterministic evaluation on pristine unseen validation data."""
    setup_logging()
    log = logging.getLogger("rl_trading_backend")

    log.info("Setting up Matrix Evaluation Protocol...")

    # 1. Ensure out-of-sample data exists
    data_path = "data/validation_btcusd.csv"
    if not Path(data_path).exists():
        log.error(f"Cannot find out-of-sample data at {data_path}. Please split historical first.")
        return

    # 2. Set up the 4D Environment (Price, MA, Volume, Sentiment)
    log.info("Loading validation matrix...")
    env = TradingEnvironment(
        data_path=data_path,
        initial_balance=100_000.0
    )

    # 3. Set up the Brain
    log.info("Initializing PPO Neural Core (state_dim=4)...")
    agent = PPOAgent(state_dim=4, action_dim=1)
    
    # Load the champion weights safely
    try:
        agent.load_weights("best_model.pth")
    except FileNotFoundError:
        log.error("Could not find 'best_model.pth'. Make sure training has saved a champion model.")
        return

    # 4. Run the deterministic backtest engine (no backpropagation!)
    # We must patch the BacktestEngine temporarily to support our environment loop structure
    log.info("Executing evaluation timeline...")
    engine = BacktestEngine(agent=agent, environment=env)
    
    # We override the run method of engine temporarily to map step properly since our env is modernized
    obs, info = env.reset()
    engine.history.append({"equity": env.net_worth, "timestamp": env.current_step, "trade_log": []})
    
    done = False
    while not done:
        action = agent.predict(obs) # Deterministic forward pass
        obs, reward, terminated, truncated, _ = env.step([action])
        
        engine.history.append({"equity": env.net_worth, "timestamp": env.current_step, "trade_log": [action] if abs(action)>0.4 else []})
        
        done = terminated or truncated

    report = engine.generate_report()
    log.info("Evaluation complete.")

if __name__ == "__main__":
    main()
