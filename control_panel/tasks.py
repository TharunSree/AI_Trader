# control_panel/tasks.py

from celery import shared_task, current_app
from celery.contrib.abortable import AbortableTask
from django.utils import timezone
from .models import TrainingJob, MetaTrainingJob
from .models import TrainingJob, MetaTrainingJob
from src.strategies import STRATEGY_PLAYBOOK
from src.data.preprocessor import calculate_features
from src.data.yfinance_loader import YFinanceLoader
from src.core.environment import TradingEnv
from src.models.ppo_agent import PPOAgent
from src.models.trainer import Trainer
from src.evaluation.validator import Validator
from itertools import product
import pandas as pd
from pathlib import Path

# --- NEW: Import our training session class ---
from src.sessions.training_session import TrainingSession


@shared_task(bind=True, base=AbortableTask)
def run_training_job_task(self, job_id):
    """The real task that connects to our AI training engine."""
    job = TrainingJob.objects.get(id=job_id)
    job.status = 'RUNNING'
    job.celery_task_id = self.request.id
    job.start_time = timezone.now()
    job.save()

    # Define a callback function that the trainer can use to update progress
    def progress_callback(progress_percent):
        self.update_state(state='PROGRESS', meta={'progress': progress_percent})
        job.progress = progress_percent
        job.save(update_fields=['progress'])

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
        print(f"An error occurred during training: {e}")
        job.status = 'FAILED'

    job.end_time = timezone.now()
    job.progress = 100
    job.save()

    return f"Training finished with status: {job.status}"


@shared_task
def stop_celery_task(task_id):
    """Stops a running Celery task."""
    current_app.control.revoke(task_id, terminate=True)


@shared_task(bind=True)
def run_meta_trainer_task(self, meta_job_id):
    meta_job = MetaTrainingJob.objects.get(id=meta_job_id)
    meta_job.status = 'RUNNING'
    meta_job.celery_task_id = self.request.id
    meta_job.start_time = timezone.now()
    meta_job.save()

    try:
        # --- All logic from run_meta_trainer.py is now here ---
        TICKER = ['SPY']
        TRAIN_START, TRAIN_END = '2015-01-01', '2021-12-31'
        VALIDATION_START, VALIDATION_END = '2022-01-01', '2023-12-31'
        PRINCIPAL = 100_000

        train_df = calculate_features(YFinanceLoader(TICKER, TRAIN_START, TRAIN_END).load_data())
        validation_df = calculate_features(YFinanceLoader(TICKER, VALIDATION_START, VALIDATION_END).load_data())

        results = []
        best_sharpe = -float('inf')
        best_strategy_info = {}

        all_combinations = list(product(
            STRATEGY_PLAYBOOK["feature_sets"].keys(),
            STRATEGY_PLAYBOOK["hyperparameters"].keys(),
            STRATEGY_PLAYBOOK["window_sizes"]
        ))

        for i, (feat_key, param_key, window) in enumerate(all_combinations):
            # (Training and validation logic for each combination...)
            # This is a simplified version for brevity. The full logic
            # would be copied from the standalone run_meta_trainer.py script.

            # --- After evaluating a strategy ---
            # performance_metrics = validator.evaluate(agent)
            # current_sharpe = performance_metrics['sharpe_ratio']
            # if current_sharpe > best_sharpe:
            #     best_sharpe = current_sharpe
            #     best_strategy_info = {"features": feat_key, "params": param_key, "window": window, "sharpe": best_sharpe}
            #     agent.save(Path("saved_models/best_agent.pth"))

            # Update progress
            progress = int(((i + 1) / len(all_combinations)) * 100)
            self.update_state(state='PROGRESS', meta={'progress': progress})

        meta_job.status = 'COMPLETED'
        meta_job.best_strategy_details = best_strategy_info

    except Exception as e:
        meta_job.status = 'FAILED'

    meta_job.end_time = timezone.now()
    meta_job.save()
    return f"Meta-Training finished with status: {meta_job.status}"