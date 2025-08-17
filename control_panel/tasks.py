# control_panel/tasks.py
import time
import logging
import pandas as pd
from pathlib import Path
from itertools import product

from celery import shared_task, current_app
from celery.contrib.abortable import AbortableTask
from django.utils import timezone

from src.sessions.evaluation_session import EvaluationSession
from src.sessions.trading_session import TradingSession
from trader_project import settings
from .models import TrainingJob, MetaTrainingJob, PaperTrader, EvaluationJob
from src.strategies import STRATEGY_PLAYBOOK
from src.data.preprocessor import calculate_features
from src.data.yfinance_loader import YFinanceLoader
from src.core.environment import TradingEnv
from src.models.ppo_agent import PPOAgent
from src.models.trainer import Trainer
from src.evaluation.validator import Validator

# --- NEW: Import our training session class ---
from src.sessions.training_session import TrainingSession

# Set up logger
logger = logging.getLogger(__name__)


@shared_task(bind=True, base=AbortableTask)
def run_training_job_task(self, job_id):
    """The real task that connects to our AI training engine."""
    job = TrainingJob.objects.get(id=job_id)
    job.status = 'RUNNING'
    job.celery_task_id = self.request.id
    job.start_time = timezone.now()
    job.save()

    # Define a callback function that the trainer can use to update progress
    def progress_callback(progress_percent, latest_reward):
        self.update_state(state='PROGRESS', meta={'progress': progress_percent, 'reward': latest_reward})
        job.progress = progress_percent
        job.best_reward = latest_reward if latest_reward > job.best_reward else job.best_reward
        job.save(update_fields=['progress', 'best_reward'])

    try:
        # --- THIS IS THE INTEGRATION ---
        # 1. Define the configuration for the training session from our Django model
        config = {
            'ticker': 'SPY',
            'start_date': '2015-01-01',
            'end_date': '2023-12-31',
            'observation_columns': [
                'returns', 'SMA_50', 'RSI_14', 'STOCHk_14_3_3', 'MACDh_12_26_9',
                'ADX_14', 'BBP_20_2', 'ATR_14', 'OBV'
            ],
            'window_size': 10,
            'initial_cash': job.initial_cash,
            'num_episodes': job.num_episodes,
            'target_equity': job.target_equity,
        }

        # 2. Create and run the session
        session = TrainingSession(config)
        result = session.run(progress_callback=progress_callback)

        # 3. Update the job status based on the result
        if result == 'TARGET_REACHED' or result == 'MAX_EPISODES_REACHED':
            job.status = 'COMPLETED'
        else:
            job.status = 'STOPPED'  # Could be 'STALLED'

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        job.status = 'FAILED'
        job.error_message = str(e)

    job.end_time = timezone.now()
    job.progress = 100
    job.save()

    return f"Training finished with status: {job.status}"


@shared_task
def stop_celery_task(task_id):
    """Stops a running Celery task."""
    current_app.control.revoke(task_id, terminate=True)


# In control_panel/tasks.py

@shared_task(bind=True, base=AbortableTask)
def run_meta_trainer_task(self, meta_job_id):
    meta_job = MetaTrainingJob.objects.get(id=meta_job_id)
    meta_job.status = 'RUNNING'
    meta_job.celery_task_id = self.request.id
    meta_job.start_time = timezone.now()
    meta_job.save()

    try:
        # 1. --- Load and Prepare Data ---
        TICKER = ['SPY'];
        TRAIN_START = '2015-01-01';
        TRAIN_END = '2021-12-31'
        VALIDATION_START = '2022-01-01';
        VALIDATION_END = '2023-12-31'

        logger.info("META-TRAINER: Loading and preparing data...")
        train_df = calculate_features(YFinanceLoader(TICKER, TRAIN_START, TRAIN_END).load_data())
        validation_df = calculate_features(YFinanceLoader(TICKER, VALIDATION_START, VALIDATION_END).load_data())

        best_sharpe = -float('inf')
        best_strategy_info = {}

        # 2. --- Generate All Strategy Combinations ---
        all_combinations = list(product(
            STRATEGY_PLAYBOOK["feature_sets"].keys(),
            STRATEGY_PLAYBOOK["hyperparameters"].keys(),
            STRATEGY_PLAYBOOK["window_sizes"]
        ))

        # 3. --- Loop Through Each Strategy ---
        for i, (feat_key, param_key, window) in enumerate(all_combinations):
            logger.info(
                f"META-TRAINER: Running experiment {i + 1}/{len(all_combinations)}: {feat_key}, {param_key}, w={window}")

            features = STRATEGY_PLAYBOOK["feature_sets"][feat_key]
            params = STRATEGY_PLAYBOOK["hyperparameters"][param_key]

            # 4. --- Train a Candidate Agent ---
            train_env = TradingEnv(train_df, features, window, meta_job.initial_cash, 0.001, 0.0005)
            state_dim, action_dim = train_env.observation_space.shape[0], train_env.action_space.n
            agent = PPOAgent(state_dim, action_dim, lr=params['lr'])

            trainer_config = {
                "num_episodes": 50, "gamma": params['gamma'], "target_equity": meta_job.target_equity,
                "patience_episodes": 20, "min_reward_improvement": 100.0
            }
            trainer = Trainer(agent, train_env, trainer_config)
            trainer.train()

            # 5. --- Evaluate the Agent on Unseen Data ---
            validator_env_config = {'features': features, 'window': window}
            validator = Validator(validation_df, validator_env_config)
            performance_metrics = validator.evaluate(agent)

            current_sharpe = performance_metrics['sharpe_ratio']

            # 6. --- Check for New Champion ---
            if current_sharpe > best_sharpe:
                best_sharpe = current_sharpe
                best_strategy_info = {
                    "features": feat_key, "params": param_key, "window": window,
                    "sharpe_ratio": best_sharpe, "return_pct": performance_metrics['total_return_pct']
                }
                save_path = settings.BASE_DIR / "saved_models" / "best_agent.pth"
                agent.save(save_path)
                logger.info(f"!!! New best agent found! Sharpe: {best_sharpe:.2f}. Model saved. !!!")

            # 7. --- Update Progress in Database ---
            progress = int(((i + 1) / len(all_combinations)) * 100)
            meta_job.progress = progress
            meta_job.results = best_strategy_info
            meta_job.save(update_fields=['progress', 'results'])

        meta_job.status = 'COMPLETED'

    except Exception as e:
        logger.error(f"Meta-Training job {meta_job.id} failed: {e}")
        meta_job.status = 'FAILED'
        meta_job.error_message = str(e)

    meta_job.end_time = timezone.now()
    meta_job.save()
    return f"Meta-Training finished with status: {meta_job.status}"


@shared_task(bind=True, base=AbortableTask)
def run_paper_trader_task(self, trader_id, model_file):
    """The real task that runs the live paper trading bot."""
    trader = PaperTrader.objects.get(id=trader_id)
    logger.info(f"Starting paper trader with model: {model_file}")

    def should_abort():
        trader.refresh_from_db()
        return self.is_aborted() or trader.status == 'STOPPED'

    try:
        config = {
            "model_file": model_file,
            "interval_minutes": 60,
        }
        session = TradingSession(config, abort_flag_callback=should_abort)
        session.task = self  # Pass the task instance to the session
        session.run()

    except Exception as e:
        logger.error(f"Paper trader task failed: {e}")
        trader.status = 'FAILED'

    if trader.status != 'FAILED':
        trader.status = 'STOPPED'
    trader.save()
    logger.info("Paper trader task was stopped.")
    return "PAPER TRADER STOPPED"


@shared_task(bind=True, base=AbortableTask)
def run_evaluation_task(self, job_id):
    job = EvaluationJob.objects.get(id=job_id)
    job.status = 'RUNNING'
    job.celery_task_id = self.request.id
    job.save()

    try:
        config = {
            'model_file': job.model_file,
            'start_date': job.start_date.strftime('%Y-%m-%d'),
            'end_date': job.end_date.strftime('%Y-%m-%d'),
        }

        session = EvaluationSession(config)
        results = session.run()

        job.results = results
        job.status = 'COMPLETED'

    except Exception as e:
        logger.error(f"Evaluation job {job.id} failed: {e}")
        job.status = 'FAILED'
        job.error_message = str(e)

    job.save()
    return f"Evaluation finished with status: {job.status}"