import time
import logging
from pathlib import Path
from itertools import product
from datetime import datetime
from celery import shared_task, current_app
from celery.contrib.abortable import AbortableTask
from django.utils import timezone
from trader_project import settings
import torch

from src.sessions.evaluation_session import EvaluationSession
from src.sessions.trading_session import TradingSession
from .models import TrainingJob, MetaTrainingJob, PaperTrader, EvaluationJob
from src.strategies import STRATEGY_PLAYBOOK
from src.data.preprocessor import calculate_features
from src.data.yfinance_loader import YFinanceLoader
from src.core.environment import TradingEnv
from src.models.ppo_agent import PPOAgent
from src.models.trainer import Trainer
from src.evaluation.validator import Validator
from src.sessions.training_session import TrainingSession

logger = logging.getLogger(__name__)


@shared_task(bind=True, base=AbortableTask)
def run_training_job_task(self, job_id):
    job = TrainingJob.objects.get(id=job_id)
    job.status = 'RUNNING'
    job.celery_task_id = self.request.id
    job.save()

    def progress_callback(progress_percent, latest_reward):
        job.refresh_from_db()
        job.progress = progress_percent
        job.best_reward = latest_reward if latest_reward > job.best_reward else job.best_reward
        job.save(update_fields=['progress', 'best_reward'])
        self.update_state(state='PROGRESS', meta={'progress': job.progress, 'reward': job.best_reward})

    try:
        features = STRATEGY_PLAYBOOK['feature_sets'][job.feature_set_key]
        params = STRATEGY_PLAYBOOK['hyperparameters'][job.hyperparameter_key]
        config = {
            'ticker': 'SPY', 'start_date': '2015-01-01', 'end_date': '2023-12-31',
            'features': features, 'window': job.window_size, 'params': params,
            'initial_cash': float(job.initial_cash), 'num_episodes': 500, 'target_equity': float('inf'),
        }

        session = TrainingSession(config)
        agent, result = session.run(progress_callback=progress_callback)

        if agent:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            model_name = f"Simple-{job.feature_set_key}-w{job.window_size}-{timestamp}.pth"
            save_path = Path(settings.BASE_DIR) / "saved_models" / model_name

            full_config = {"features": features, "params": params, "window": job.window_size, "training_result": result}
            agent.save(save_path, config=full_config)
            logger.info(f"Saved final simple training model to {save_path}")

        job.status = 'COMPLETED' if 'REACHED' in result else 'STOPPED'

    except Exception as e:
        logger.error(f"Training job {job.id} failed: {e}", exc_info=True)
        job.status = 'FAILED'
        job.error_message = str(e)

    job.end_time = timezone.now()
    job.save()
    return f"Training finished with status: {job.status}"


@shared_task(bind=True, base=AbortableTask)
def run_meta_trainer_task(self, meta_job_id):
    meta_job = MetaTrainingJob.objects.get(id=meta_job_id)
    meta_job.status = 'RUNNING'
    meta_job.celery_task_id = self.request.id
    meta_job.start_time = timezone.now()
    meta_job.save()

    try:
        TICKER = ['SPY']
        TRAIN_START, TRAIN_END = '2015-01-01', '2021-12-31'
        VALIDATION_START, VALIDATION_END = '2022-01-01', '2023-12-31'

        logger.info("META-TRAINER: Loading and preparing data...")
        train_df = calculate_features(YFinanceLoader(TICKER, TRAIN_START, TRAIN_END).load_data())
        validation_df = calculate_features(YFinanceLoader(TICKER, VALIDATION_START, VALIDATION_END).load_data())

        best_sharpe = -float('inf')
        best_strategy_info = {}
        champion_agent_state = None

        all_combinations = list(product(
            STRATEGY_PLAYBOOK["feature_sets"].keys(),
            STRATEGY_PLAYBOOK["hyperparameters"].keys(),
            STRATEGY_PLAYBOOK["window_sizes"]
        ))

        for i, (feat_key, param_key, window) in enumerate(all_combinations):
            if self.is_aborted():
                break

            logger.info(
                f"META-TRAINER: Running experiment {i + 1}/{len(all_combinations)}: {feat_key}, {param_key}, w={window}")

            features = STRATEGY_PLAYBOOK["feature_sets"][feat_key]
            params = STRATEGY_PLAYBOOK["hyperparameters"][param_key]

            train_env = TradingEnv(train_df, features, window, float(meta_job.initial_cash), 0.001, 0.0005)
            state_dim, action_dim = train_env.observation_space.shape[0], train_env.action_space.n
            agent = PPOAgent(state_dim, action_dim, lr=params['lr'])

            trainer_config = {"num_episodes": 50, "gamma": params['gamma'],
                              "target_equity": float(meta_job.target_equity), "patience_episodes": 20,
                              "min_reward_improvement": 100.0}
            trainer = Trainer(agent, train_env, trainer_config)
            trainer.train()

            validator = Validator(validation_df, {'features': features, 'window': window})
            performance_metrics = validator.evaluate(agent)
            current_sharpe = performance_metrics['sharpe_ratio']

            if current_sharpe > best_sharpe:
                best_sharpe = current_sharpe
                best_strategy_info = {
                    "feature_set_key": feat_key, "hyperparameter_key": param_key,
                    "features": features, "params": params, "window": window,
                    "sharpe_ratio": best_sharpe, "return_pct": performance_metrics['total_return_pct']
                }
                champion_agent_state = agent.actor.state_dict()
                logger.info(f"!!! New champion found! Sharpe: {best_sharpe:.2f}. State captured in memory. !!!")

            progress = int(((i + 1) / len(all_combinations)) * 100)
            meta_job.progress, meta_job.results = progress, best_strategy_info
            meta_job.save(update_fields=['progress', 'results'])

        if champion_agent_state and not self.is_aborted():
            logger.info("Meta-training complete. Saving the final champion model...")

            final_features = best_strategy_info['features']
            final_window = best_strategy_info['window']
            final_params = best_strategy_info['params']
            final_state_dim = len(final_features) * final_window
            final_action_dim = 3

            champion_agent = PPOAgent(final_state_dim, final_action_dim, lr=final_params['lr'])
            champion_agent.actor.load_state_dict(champion_agent_state)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            # --- MODIFIED: Shorter, more readable name ---
            model_name = f"Meta-Champion-{best_strategy_info['feature_set_key']}-sharpe{best_sharpe:.2f}-{timestamp}.pth"
            save_path = Path(settings.BASE_DIR) / "saved_models" / model_name

            champion_agent.save(save_path, config=best_strategy_info)
            logger.info(f"Final champion model saved as {model_name}")

        meta_job.status = 'COMPLETED' if not self.is_aborted() else 'STOPPED'

    except Exception as e:
        logger.error(f"Meta-Training job {meta_job.id} failed: {e}", exc_info=True)
        meta_job.status = 'FAILED'
        meta_job.error_message = str(e)

    meta_job.end_time = timezone.now()
    meta_job.save()
    return f"Meta-Training finished with status: {meta_job.status}"


@shared_task(bind=True, base=AbortableTask)
def run_paper_trader_task(self, trader_id, model_file):
    trader = PaperTrader.objects.get(id=trader_id)
    logger.info(f"Starting paper trader '{trader_id}' with model '{model_file}'")

    def should_abort():
        if self.is_aborted(): return True
        trader.refresh_from_db()
        return trader.status != 'RUNNING'

    try:
        config = {"model_file": model_file, "interval_minutes": 1, "trader_id": trader_id}
        session = TradingSession(config, abort_flag_callback=should_abort)
        session.task = self
        session.run()
    except Exception as e:
        logger.error(f"Paper trader task for trader {trader_id} failed: {e}", exc_info=True)
        trader.status = 'FAILED';
        trader.error_message = str(e)
        trader.save()
    if trader.status != 'FAILED':
        trader.refresh_from_db()
        if trader.status == 'RUNNING':
            trader.status = 'STOPPED';
            trader.save()
    logger.info(f"Paper trader task for trader {trader_id} has stopped.")
    return f"PAPER TRADER {trader_id} STOPPED"


@shared_task(bind=True, base=AbortableTask)
def run_evaluation_task(self, job_id):
    job = EvaluationJob.objects.get(id=job_id)
    job.status = 'RUNNING';
    job.celery_task_id = self.request.id
    job.save()
    try:
        config = {'model_file': job.model_file, 'start_date': str(job.start_date), 'end_date': str(job.end_date)}
        session = EvaluationSession(config)
        results = session.run()
        job.results = results;
        job.status = 'COMPLETED'
    except Exception as e:
        logger.error(f"Evaluation job {job.id} failed: {e}", exc_info=True)
        job.status = 'FAILED';
        job.error_message = str(e)
    job.save()
    return f"Evaluation finished with status: {job.status}"


@shared_task
def stop_celery_task(task_id):
    current_app.control.revoke(task_id, terminate=True, signal='SIGTERM')
