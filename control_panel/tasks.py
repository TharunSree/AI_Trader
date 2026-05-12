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

@shared_task
def run_ab_tournament(*args, **kwargs):
    """
    Phase 17 A/B Tournament Daemon.
    Evaluates currently running PaperTraders against best available idle models.
    If a swap is recommended, it checks rejection history via SystemAlert `insist_count`.
    """
    from control_panel.models import PaperTrader, TrainingJob, SystemAlert, SystemSettings
    settings = SystemSettings.load()
    if not settings.notify_ab_test:
        return "A/B Tournament Silenced by Settings."
        
    logger.info("Initiating A/B Strategy Tournament...")
    running_models = PaperTrader.objects.filter(status='RUNNING')
    best_candidate = TrainingJob.objects.filter(status='COMPLETED', is_live_trading_ready=True).order_by('-best_reward').first()
    
    if not best_candidate:
        return "No certified candidates for tournament."
        
    for r_model in running_models:
        # Use the bot's actual live profit as the benchmark (synced by async_engine telemetry)
        r_perf = float(r_model.live_net_profit or 0.0)
        
        if best_candidate.best_reward > r_perf:
            # Check insistence
            alert = SystemAlert.objects.filter(related_model_reference=f"db:{best_candidate.id}", level='AB_SWAP').first()
            
            # Define extremely good: candidate reward is significantly higher than live performance
            is_extremely_good = best_candidate.best_reward >= (r_perf + 5.0)
            max_pings = 3 if is_extremely_good else 1
            
            if alert:
                if alert.insist_count < max_pings:
                    alert.insist_count += 1
                    alert.is_read = False
                    alert.message = f"URGENT: Algorithm db:{best_candidate.id} mathematically overtakes active instance [{r_model.id}]. This is ping #{alert.insist_count}."
                    alert.save()
                    logger.info(f"Tournament Daemon Insisting Swap! Ping #{alert.insist_count}")
                else:
                    logger.info("Tournament Daemon: Alert silenced, user persistently ignored ping threshold.")
            else:
                SystemAlert.objects.create(
                    level='AB_SWAP',
                    title='A/B Algorithm Tournament Winner',
                    message=f"Neural Engine recommended a live chassis swap. Candidate [{best_candidate.name}] scored {best_candidate.best_reward:.2f} reward vs live profit ${r_perf:,.2f}.",
                    related_model_reference=f"db:{best_candidate.id}",
                    insist_count=1
                )
    return "Tournament Completed."

@shared_task
def auto_resurrect_nodes(*args, **kwargs):
    """
    Zero-Touch Watchdog: Scans the database for nodes marked 'FAILED'
    and dynamically re-launches them with a circuit breaker to prevent infinite loops.
    Max 3 resurrection attempts per 30 minutes before force-stopping the node.
    """
    from control_panel.models import PaperTrader
    from control_panel.views import _launch_trader_instance
    from src.reporting.email_dispatcher import send_node_status_email
    from django.utils import timezone as dj_tz
    from datetime import timedelta
    
    failed_nodes = PaperTrader.objects.filter(status='FAILED')
    if not failed_nodes.exists():
        return "No FAILED nodes found for resurrection."
    
    resurrected = 0
    for node in failed_nodes:
        # Circuit Breaker: Track consecutive failures via error_message prefix
        fail_marker = "[RESURRECT_COUNT:"
        current_count = 0
        err = node.error_message or ""
        if fail_marker in err:
            try:
                current_count = int(err.split(fail_marker)[1].split("]")[0])
            except (ValueError, IndexError):
                current_count = 0
        
        if current_count >= 3:
            logger.error(f"Circuit breaker tripped for node {node.id}: {current_count} consecutive resurrections. Force-stopping.")
            node.status = 'STOPPED'
            node.error_message = f"Auto-resurrect circuit breaker tripped after {current_count} consecutive failures. Manual intervention required."
            node.save(update_fields=['status', 'error_message'])
            continue
            
        logger.info(f"Watchdog detecting FAILED node {node.id}. Force rebooting (attempt {current_count + 1}/3)...")
        try:
            # Tag the resurrection count before relaunching
            node.error_message = f"{fail_marker}{current_count + 1}] {err[:200]}"
            node.save(update_fields=['error_message'])
            _launch_trader_instance(node)
            send_node_status_email(
                "Jarvis Auto-Resurrection Watchdog",
                f"T{node.id}",
                "RESURRECTED",
                f"Node {node.id} restored (attempt {current_count + 1}/3)."
            )
            resurrected += 1
            logger.info(f"Node {node.id} successfully resurrected.")
        except Exception as e:
            logger.error(f"Watchdog failed to resurrect node {node.id}: {str(e)}")
            
    return f"Watchdog sweep complete. Resurrected {resurrected} nodes."

@shared_task
def purge_decayed_logs(*args, **kwargs):
    """
    Organic Log Purge: Sweeps the local logs/ directory and unlinks files older than 14 days
    to prevent disk overload from autonomous unattended multi-week runs.
    """
    import os
    import time
    from pathlib import Path
    
    log_dir = Path(__file__).parent.parent / "logs"
    if not log_dir.exists():
        return "Log directory does not exist."
        
    current_time = time.time()
    decay_threshold_seconds = 14 * 24 * 60 * 60  # 14 Days
    purged_count = 0
    
    for log_file in log_dir.glob("*.log"):
        # Check modification time
        if (current_time - os.path.getmtime(log_file)) > decay_threshold_seconds:
            try:
                os.remove(log_file)
                purged_count += 1
                logger.info(f"Purged decayed log artifact: {log_file.name}")
            except Exception as e:
                logger.error(f"Failed to purge log {log_file.name}: {e}")
                
    return f"Purge Cycle Complete. Cleared {purged_count} legacy log files."

@shared_task
def generate_eod_reports(*args, **kwargs):
    """
    Generates End-of-Day intelligence reports for all active bots.
    Must run BEFORE the A/B tournament and cognitive mutation so they have fresh data.
    """
    from src.reporting.eod_generator import write_report_artifacts
    from src.reporting.email_dispatcher import send_bundled_report_email
    from datetime import date
    
    logger.info("Generating EOD reports for all active trading bots...")
    try:
        today = date.today()
        artifacts = write_report_artifacts(today)
        if artifacts:
            send_bundled_report_email(artifacts, today.isoformat())
            return f"EOD reports generated and emailed for {len(artifacts)} bots."
        return "No active bots found for EOD report generation."
    except Exception as e:
        logger.error(f"EOD report generation failed: {e}")
        return f"EOD report generation failed: {e}"

@shared_task
def trigger_daily_cognitive_mutation(*args, **kwargs):
    """
    Automates the daily 'Strategic Mutation' (Epoch Parameter Refinement).
    Runs immediately after the A/B evaluation tournaments conclude to ingest fresh daily data.
    """
    from src.core.code_rewriter import orchestrate_rewrite
    logger.info("Executing Scheduled Daily Cognitive Mutation...")
    try:
        orchestrate_rewrite()
        return "Daily mutation completed successfully."
    except Exception as e:
        logger.error(f"Daily mutation failed: {e}")
        return f"Daily mutation failed: {e}"
