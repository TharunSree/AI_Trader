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

        # Use integer window sizes instead of strategy strings
        window_sizes = [5, 8, 10, 15, 20, 30]  # Fixed: All integers

        all_combinations = list(product(
            STRATEGY_PLAYBOOK["feature_sets"].keys(),
            STRATEGY_PLAYBOOK["hyperparameters"].keys(),
            window_sizes  # Now using integer list
        ))

        total_combinations = len(all_combinations)
        logger.info(f"META-TRAINER: Testing {total_combinations} combinations")

        for i, (feat_key, param_key, window) in enumerate(all_combinations):
            if self.is_aborted():
                logger.info("Meta-training aborted by user")
                break

            logger.info(
                f"META-TRAINER: [{i + 1}/{total_combinations}] Testing {feat_key} + {param_key} + window={window}")

            try:
                features = STRATEGY_PLAYBOOK["feature_sets"][feat_key]
                params = STRATEGY_PLAYBOOK["hyperparameters"][param_key]

                # Ensure window is integer for environment
                window_int = int(window)

                # Create training environment
                train_env = TradingEnv(
                    df=train_df,
                    observation_columns=features,
                    window_size=window_int,  # Now guaranteed integer
                    initial_cash=float(meta_job.initial_cash),
                    transaction_cost_pct=0.001,
                    slippage_pct=0.0005
                )

                # Calculate dimensions properly
                state_dim = len(features) * window_int  # Both are integers
                action_dim = train_env.action_space.n

                # Create and train agent
                agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, lr=params['lr'])

                trainer_config = {
                    "num_episodes": 50,  # Reduced for meta training speed
                    "gamma": params['gamma'],
                    "target_equity": float(meta_job.target_equity) if meta_job.target_equity else float('inf')
                }

                trainer = Trainer(agent, train_env, trainer_config)
                training_result = trainer.train()

                # Validate on out-of-sample data
                validator = Validator(validation_df, {'features': features, 'window': window_int})
                performance_metrics = validator.evaluate(agent)
                current_sharpe = performance_metrics['sharpe_ratio']

                logger.info(f"META-TRAINER: Combination {i + 1} - Sharpe: {current_sharpe:.3f}")

                # Track best performing strategy
                if current_sharpe > best_sharpe:
                    best_sharpe = current_sharpe
                    best_strategy_info = {
                        'feature_set_key': feat_key,
                        'hyperparameter_key': param_key,
                        'features': features,
                        'params': params,
                        'window': window_int,
                        'sharpe_ratio': current_sharpe,
                        'performance_metrics': performance_metrics,
                        'training_result': training_result
                    }
                    # Save the best agent's state
                    champion_agent_state = agent.actor.state_dict().copy()

                    logger.info(f"META-TRAINER: New champion! Sharpe: {current_sharpe:.3f}")

                # Update progress
                progress = int(((i + 1) / total_combinations) * 100)
                meta_job.progress = progress
                meta_job.results = best_strategy_info
                meta_job.save(update_fields=['progress', 'results'])

                # Update task state for real-time monitoring
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'activity': f"Testing combination {i + 1}/{total_combinations}",
                        'progress': progress,
                        'current_sharpe': current_sharpe,
                        'best_sharpe': best_sharpe,
                        'current_combo': f"{feat_key}+{param_key}+w{window}"
                    }
                )

            except Exception as e:
                logger.error(f"META-TRAINER: Error in combination {i + 1}: {e}")
                continue

        # Save the champion model if we found one
        if champion_agent_state and not self.is_aborted():
            logger.info("META-TRAINER: Saving champion model...")

            try:
                # Reconstruct the champion agent
                final_features = best_strategy_info['features']
                final_window = best_strategy_info['window']
                final_params = best_strategy_info['params']
                final_state_dim = len(final_features) * final_window
                final_action_dim = 3

                champion_agent = PPOAgent(
                    state_dim=final_state_dim,
                    action_dim=final_action_dim,
                    lr=final_params['lr']
                )
                champion_agent.actor.load_state_dict(champion_agent_state)

                # Create descriptive filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                model_name = f"Meta-Champion-{best_strategy_info['feature_set_key']}-sharpe{best_sharpe:.2f}-{timestamp}.pth"
                save_path = Path(settings.BASE_DIR) / "saved_models" / model_name

                # Ensure directory exists
                save_path.parent.mkdir(exist_ok=True)

                # Save with full configuration
                champion_agent.save(save_path, config=best_strategy_info)

                logger.info(f"META-TRAINER: Champion model saved as {model_name}")
                logger.info(f"META-TRAINER: Best Sharpe Ratio: {best_sharpe:.3f}")
                logger.info(
                    f"META-TRAINER: Best Strategy: {best_strategy_info['feature_set_key']} + {best_strategy_info['hyperparameter_key']} + window={best_strategy_info['window']}")

            except Exception as e:
                logger.error(f"META-TRAINER: Failed to save champion model: {e}")

        # Set final status
        if self.is_aborted():
            meta_job.status = 'STOPPED'
            logger.info("META-TRAINER: Task was aborted")
        elif best_sharpe > -float('inf'):
            meta_job.status = 'COMPLETED'
            logger.info(f"META-TRAINER: Completed successfully with best Sharpe: {best_sharpe:.3f}")
        else:
            meta_job.status = 'FAILED'
            meta_job.error_message = "No valid strategies found"
            logger.error("META-TRAINER: No valid strategies found")

    except Exception as e:
        logger.error(f"META-TRAINER: Job {meta_job.id} failed: {e}", exc_info=True)
        meta_job.status = 'FAILED'
        meta_job.error_message = str(e)

    finally:
        meta_job.end_time = timezone.now()
        meta_job.save()

    return f"Meta-Training finished with status: {meta_job.status}"



# Add this configuration in the run_paper_trader_task function
# Add this enhanced configuration in run_paper_trader_task

@shared_task(bind=True, base=AbortableTask)
def run_paper_trader_task(self, trader_id, model_file):
    trader = PaperTrader.objects.get(id=trader_id)
    logger.info(f"Starting paper trader '{trader_id}' with model '{model_file}'")

    def should_abort():
        if self.is_aborted(): return True
        trader.refresh_from_db()
        return trader.status != 'RUNNING'

    try:
        # Enhanced config with strategy detection
        config = {
            'trader_id': trader_id,
            'model_file': model_file,
            'interval_minutes': 2.5,
            'position_size': 500,
            'enable_profit_taking': True,

            # Auto-detect strategy type from model name
            'strategy_type': 'short_term' if 'scalping' in model_file.lower() or 'quick' in model_file.lower() else 'long_term',

            # Strategy-specific settings
            'profit_target_percent': 1.5 if 'short' in model_file.lower() else 3.0,
            'stop_loss_percent': 0.8 if 'short' in model_file.lower() else 1.5,
            'max_daily_trades': 15 if 'short' in model_file.lower() else 5,
            'confidence_threshold': 0.65 if 'short' in model_file.lower() else 0.75
        }

        session = TradingSession(config, abort_flag_callback=should_abort)
        session.task = self
        session.run()

    except Exception as e:
        logger.error(f"Paper trader task failed: {e}", exc_info=True)
        trader.status = 'FAILED'
        trader.error_message = str(e)
        trader.save()

    if trader.status != 'FAILED':
        trader.refresh_from_db()
        if trader.status == 'RUNNING':
            trader.status = 'STOPPED'
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
