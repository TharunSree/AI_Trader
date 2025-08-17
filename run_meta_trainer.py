# run_meta_trainer.py

import logging
import pandas as pd
import torch
from pathlib import Path
from itertools import product
from tqdm import tqdm
import time

from src.strategies import STRATEGY_PLAYBOOK
from src.data.preprocessor import calculate_features
from src.data.yfinance_loader import YFinanceLoader
from src.core.environment import TradingEnv
from src.models.ppo_agent import PPOAgent
from src.models.trainer import Trainer
from src.utils.logger import setup_logging
from src.evaluation.validator import Validator


def main():
    setup_logging()
    log = logging.getLogger("rl_trading_backend")

    # --- Configuration ---
    TICKER = ["SPY"]
    TRAIN_START = "2015-01-01"
    TRAIN_END = "2021-12-31"
    VALIDATION_START = "2022-01-01"
    VALIDATION_END = "2023-12-31"
    PRINCIPAL = 100_000
    TARGET_EQUITY = PRINCIPAL * 2  # Target a 100% return

    # --- Data Loading ---
    log.info("Loading and preparing training and validation data...")
    train_df = calculate_features(
        YFinanceLoader(TICKER, TRAIN_START, TRAIN_END).load_data()
    )
    validation_df = calculate_features(
        YFinanceLoader(TICKER, VALIDATION_START, VALIDATION_END).load_data()
    )

    all_combinations = list(
        product(
            STRATEGY_PLAYBOOK["feature_sets"].keys(),
            STRATEGY_PLAYBOOK["hyperparameters"].keys(),
            STRATEGY_PLAYBOOK["window_sizes"],
        )
    )

    # --- PHASE 1: Find all strategies that can reach the target ---
    log.info(
        f"--- STARTING PHASE 1: Finding Profitable Strategies (Target: ${TARGET_EQUITY:,.2f}) ---"
    )
    successful_strategies = []

    for i, (feat_key, param_key, window) in enumerate(
        tqdm(all_combinations, desc="Phase 1: Finding Strategies")
    ):
        log.info(
            f"Testing Strategy #{i + 1}: Features='{feat_key}', Params='{param_key}', Window={window}"
        )

        features = STRATEGY_PLAYBOOK["feature_sets"][feat_key]
        params = STRATEGY_PLAYBOOK["hyperparameters"][param_key]

        train_env = TradingEnv(train_df, features, window, PRINCIPAL, 0.001, 0.0005)

        state_dim = train_env.observation_space.shape[0]
        action_dim = train_env.action_space.n
        agent = PPOAgent(state_dim, action_dim, lr=params["lr"])

        trainer_config = {
            "num_episodes": 500,
            "gamma": params["gamma"],
            "target_equity": TARGET_EQUITY,
            "patience_episodes": 100,
            "min_reward_improvement": 100.0,
        }
        trainer = Trainer(agent, train_env, trainer_config)
        result = trainer.train()

        if result == "TARGET_REACHED":
            log.info(f"SUCCESS: Strategy #{i} reached the target during training.")
            strategy_info = {
                "id": i,
                "feat_key": feat_key,
                "param_key": param_key,
                "window": window,
                "agent": agent,
            }
            successful_strategies.append(strategy_info)

    if not successful_strategies:
        log.error("PHASE 1 FAILED: No strategies were able to reach the target.")
        return

    # --- PHASE 2: Find the fastest strategy among the winners ---
    log.info(
        f"--- STARTING PHASE 2: Finding Fastest Strategy ({len(successful_strategies)} candidates) ---"
    )
    speed_results = []

    for strategy in tqdm(successful_strategies, desc="Phase 2: Speed Testing"):
        validator_env_config = {
            "features": STRATEGY_PLAYBOOK["feature_sets"][strategy["feat_key"]],
            "window": strategy["window"],
        }
        validator = Validator(validation_df, validator_env_config)

        # We need to modify validator to return steps_to_target
        # For now, we'll use Sharpe Ratio as a proxy for efficiency
        performance = validator.evaluate(strategy["agent"])

        speed_results.append(
            {
                "strategy_id": strategy["id"],
                "features": strategy["feat_key"],
                "params": strategy["param_key"],
                "return_pct": performance["total_return_pct"],
                "sharpe_ratio": performance["sharpe_ratio"],
            }
        )

    if not speed_results:
        log.error("PHASE 2 FAILED: Could not evaluate successful strategies.")
        return

    results_df = pd.DataFrame(speed_results)
    log.info("--- Meta-Training Complete ---")
    log.info("Performance of successful strategies on validation data:")
    sorted_results = results_df.sort_values(by="sharpe_ratio", ascending=False)
    log.info("\n" + sorted_results.to_string())

    # Find the best agent and save it
    best_strategy_id = sorted_results.iloc[0]["strategy_id"]
    for s in successful_strategies:
        if s["id"] == best_strategy_id:
            best_agent = s["agent"]
            best_agent.save(Path("saved_models/best_agent_by_sharpe.pth"))
            log.info(
                f"Saved the best overall agent (Strategy #{best_strategy_id}) to 'best_agent_by_sharpe.pth'"
            )
            break


if __name__ == "__main__":
    main()
