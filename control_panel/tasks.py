from celery import shared_task
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# BYPASSED TASKS
# These old synchronous tasks have been deprecated in favor 
# of the new event-driven async_engine.py and run_training.py
# ---------------------------------------------------------

@shared_task
def run_training_job_task(*args, **kwargs):
    logger.info("Legacy training task bypassed.")
    return "Bypassed"

@shared_task
def stop_celery_task(*args, **kwargs):
    logger.info("Legacy stop task bypassed.")
    return "Bypassed"

@shared_task
def run_meta_trainer_task(*args, **kwargs):
    logger.info("Legacy meta trainer bypassed.")
    return "Bypassed"

@shared_task
def run_paper_trader_task(*args, **kwargs):
    logger.info("Legacy paper trader bypassed.")
    return "Bypassed"

@shared_task
def run_evaluation_task(*args, **kwargs):
    logger.info("Legacy evaluation bypassed.")
    return "Bypassed"