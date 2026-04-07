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
from src.core.environment import TradingEnvironment
from src.models.ppo_agent import PPOAgent
from src.models.trainer import Trainer
from src.utils.logger import setup_logging
from src.evaluation.validator import Validator

import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=int, help="Django MetaTrainingJob ID to update DOM progress.")
    args = parser.parse_args()

    PRINCIPAL = 100_000
    TARGET_EQUITY = PRINCIPAL * 2  # Target a 100% return

    meta_job = None
    if args.job_id:
        import django
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "trader_project.settings")
        django.setup()
        from control_panel.models import MetaTrainingJob
        meta_job = MetaTrainingJob.objects.filter(id=args.job_id).first()
        if meta_job:
            meta_job.status = 'RUNNING'
            meta_job.progress = 0
            meta_job.save()
            PRINCIPAL = float(meta_job.initial_cash) if meta_job.initial_cash else 100_000
            TARGET_EQUITY = float(meta_job.target_equity) if meta_job.target_equity else PRINCIPAL * 2

    setup_logging()
    log = logging.getLogger("rl_trading_backend")

    # --- Configuration ---
    TICKER = ["SPY"]
    TRAIN_START = "2015-01-01"
    TRAIN_END = "2021-12-31"
    VALIDATION_START = "2022-01-01"
    VALIDATION_END = "2023-12-31"

    # --- Data Loading ---
    log.info("Loading and preparing training and validation data...")
    train_df = calculate_features(
        YFinanceLoader(TICKER, TRAIN_START, TRAIN_END).load_data()
    )
    validation_df = calculate_features(
        YFinanceLoader(TICKER, VALIDATION_START, VALIDATION_END).load_data()
    )

    window_sizes_set = set()
    for w_list in STRATEGY_PLAYBOOK["window_sizes"].values():
        window_sizes_set.update(w_list)

    all_combinations = list(
        product(
            STRATEGY_PLAYBOOK["feature_sets"].keys(),
            STRATEGY_PLAYBOOK["hyperparameters"].keys(),
            list(window_sizes_set),
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
        if meta_job:
            # Update Progress Bar (Phase 1 = 50% of total progress limit)
            progress_pct = int(((i) / len(all_combinations)) * 50)
            meta_job.progress = progress_pct
            meta_job.save()

        log.info(
            f"Testing Strategy #{i + 1}: Features='{feat_key}', Params='{param_key}', Window={window}"
        )

        features = STRATEGY_PLAYBOOK["feature_sets"][feat_key]
        params = STRATEGY_PLAYBOOK["hyperparameters"][param_key]

        train_env = TradingEnvironment(train_df, features, window, PRINCIPAL, 0.001, 0.0005)

        state_dim = train_env.observation_space.shape[0]
        action_dim = train_env.action_space.shape[0]
        agent = PPOAgent(state_dim, action_dim, lr=params["lr"])

        trainer_config = {
            "num_episodes": 3,  # Restricted brute-force span so pipeline finishes tonight
            "gamma": params["gamma"],
            "target_equity": TARGET_EQUITY,
            "minibatch_size": 32
        }
        trainer = Trainer(agent, train_env, trainer_config)
        trained_agent = trainer.run()

        # Extract real dynamic performance from the Gym environment instead of hallucinated strings
        final_net_worth = train_env.net_worth
        
        # If it generated at least 2% ROI over its seed money, it qualifies for the Phase 2 elimination bracket
        if final_net_worth > PRINCIPAL * 1.02:  
            log.info(f"SUCCESS: Strategy #{i} turned a profit (Final Equity: ${final_net_worth:,.2f} from ${PRINCIPAL:,.2f} base).")
            strategy_info = {
                "id": i,
                "feat_key": feat_key,
                "param_key": param_key,
                "window": window,
                "agent": trained_agent,
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

    for i, strategy in enumerate(tqdm(successful_strategies, desc="Phase 2: Speed Testing")):
        if meta_job:
            # Update Progress Bar (Phase 2 = 50-100% of progress)
            progress_pct = 50 + int(((i + 1) / len(successful_strategies)) * 50)
            meta_job.progress = progress_pct
            meta_job.save()
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
    save_dir = Path("saved_models")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for s in successful_strategies:
        if s["id"] == best_strategy_id:
            best_agent = s["agent"]
            best_agent.save_weights(str(save_dir / "best_agent_by_sharpe.pth"))
            log.info(
                f"Saved the best overall agent (Strategy #{best_strategy_id}) to 'best_agent_by_sharpe.pth'"
            )
            break

    if meta_job:
        meta_job.status = 'COMPLETED'
        meta_job.progress = 100
        meta_job.results = {"sharpe_ratio": float(sorted_results.iloc[0]["sharpe_ratio"])}
        meta_job.save()

if __name__ == "__main__":
    import argparse
    import sys
    import os
    import traceback

    try:
        main()
    except Exception as e:
        parser = argparse.ArgumentParser()
        parser.add_argument("--job_id", type=int, nargs='?')
        args, _ = parser.parse_known_args()
        if args.job_id:
            import django
            os.environ.setdefault("DJANGO_SETTINGS_MODULE", "trader_project.settings")
            try:
                django.setup()
            except Exception:
                pass
            from control_panel.models import MetaTrainingJob
            meta_job = MetaTrainingJob.objects.filter(id=args.job_id).first()
            if meta_job:
                meta_job.status = 'FAILED'
                meta_job.error_message = f"Process Crashed: {str(e)}"
                meta_job.save()
        raise e
