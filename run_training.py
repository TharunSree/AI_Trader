import os
import torch
import logging
import numpy as np
import redis
import json
from dotenv import load_dotenv
from typing import Dict, List

from src.core.environment import TradingEnvironment
from src.models.ppo_agent import PPOAgent
from src.strategies import STRATEGY_PLAYBOOK

load_dotenv()
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
try:
    redis_client = redis.from_url(redis_url, decode_responses=True)
except Exception as e:
    redis_client = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Matrix_Trainer")


import argparse

def train_jarvis(job_id=None):
    job = None
    if job_id:
        import django
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "trader_project.settings")
        django.setup()
        from control_panel.models import TrainingJob
        job = TrainingJob.objects.filter(id=job_id).first()
        if job:
            job.status = 'RUNNING'
            job.save()

    # Determine specific architecture to run
    feat_key = job.feature_set_key if job else "scalping_momentum"
    param_key = job.hyperparameter_key if job else "balanced_growth"
    window = job.window_size if job else 10
    principal = float(job.initial_cash) if job and job.initial_cash else 100_000.0

    features = STRATEGY_PLAYBOOK["feature_sets"].get(feat_key, ["Close", "Volume"])
    params = STRATEGY_PLAYBOOK["hyperparameters"].get(param_key, {"lr": 1e-4, "gamma": 0.99})

    logger.info(f"🟢 MATRIX ONLINE. Hardware: {device.type.upper()}")
    logger.info(f"Targeting Architecture: Features={feat_key}, Params={param_key}, Window={window}")

    # Load authentic financial data
    from src.data.yfinance_loader import YFinanceLoader
    from src.data.preprocessor import calculate_features
    from src.models.trainer import Trainer

    logger.info("Downloading historical market payload...")
    train_df = calculate_features(
        YFinanceLoader(["SPY"], "2015-01-01", "2021-12-31").load_data()
    )

    # Boot the modernized Gym
    env = TradingEnvironment(train_df, features, window, principal, 0.001, 0.0005)
    
    # Auto-Calculate Tensor requirements
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(state_dim, action_dim, lr=params["lr"])

    trainer_config = {
        "num_episodes": 200,
        "gamma": params["gamma"],
        "update_timestep": 2048
    }

    # Custom Database Sync Hook
    def progress_callback(episode, max_episodes, reward):
        if job:
            job.progress = int((episode / max_episodes) * 100)
            if reward > job.best_reward:
                job.best_reward = float(reward)
            job.save()

    logger.info("Initiating Deep Reinforcement Learning Protocol...\n" + "-" * 50)
    
    # Delegate to the newly hardened orchestration module
    trainer = Trainer(agent, env, trainer_config)
    trained_agent = trainer.run(progress_callback=progress_callback)

    final_net_worth = env.net_worth
    roi = ((final_net_worth - principal) / principal) * 100
    
    logger.info(f"✅ Training Terminated | Final Equity: ${final_net_worth:,.2f} | ROI: {roi:+.2f}%")
    
    # Lock champion weights
    safe_name = f"single_job_{job_id}_{feat_key}.pth" if job_id else "single_model.pth"
    trained_agent.save_weights(f"saved_models/{safe_name}")

    if job:
        job.status = 'COMPLETED'
        job.progress = 100
        job.save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', type=int, default=None)
    args, _ = parser.parse_known_args()
    
    try:
        train_jarvis(job_id=args.job_id)
    except Exception as e:
        if args.job_id:
            import django
            import os
            os.environ.setdefault("DJANGO_SETTINGS_MODULE", "trader_project.settings")
            try:
                django.setup()
            except Exception:
                pass
            from control_panel.models import TrainingJob
            job = TrainingJob.objects.filter(id=args.job_id).first()
            if job:
                job.status = 'FAILED'
                job.error_message = f"Process Crashed: {str(e)}"
                job.save()
        raise e