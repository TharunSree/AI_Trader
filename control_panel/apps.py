import os
import time
import logging
import threading
from django.apps import AppConfig

logger = logging.getLogger("rl_trading_backend")

def _eod_watcher_daemon():
    """
    Background daemon that monitors the Alpaca Market Clock.
    When the market transitions from OPEN to CLOSED, it automatically
    triggers the generation of the EOD Report PDF.
    """
    import django
    # Ensure django is ready before importing models and cache
    if not django.apps.apps.ready:
        time.sleep(5)
        
    from django.core.cache import cache
    from datetime import date
    from src.execution.broker import Broker
    from src.reporting.eod_generator import write_report_artifacts

    broker = None
    was_open = False
    
    while True:
        try:
            if not broker:
                from control_panel.models import BrokerAccount
                first_account = BrokerAccount.objects.first()
                broker = Broker(account=first_account)
                
            clock = broker.get_market_clock()
            is_open = clock.get("is_open", False) if clock else False
            
            today_str = date.today().isoformat()
            cache_key = f"eod_generated_{today_str}"
            should_generate = False
            
            # Trigger 1: Detect stock market transition from OPEN -> CLOSED
            if was_open and not is_open:
                should_generate = True
                logger.info("[JARVIS SYSTEM] Market Close Detected. Triggering EOD Report...")
            
            # Trigger 2: Daily time-based fallback (for crypto-only bots)
            # Fires at ~00:00 UTC if no report has been generated today
            import datetime as _dt
            now_utc = _dt.datetime.utcnow()
            if now_utc.hour == 0 and now_utc.minute < 5 and not cache.get(cache_key):
                should_generate = True
                logger.info("[JARVIS SYSTEM] Daily midnight trigger. Generating EOD Report for crypto bots...")
                
            if should_generate and not cache.get(cache_key):
                try:
                    from src.reporting.email_dispatcher import send_report_email
                    artifacts_list = write_report_artifacts()
                    
                    logger.info(f"[JARVIS SYSTEM] Dispatching bundled Telemetry Payload for {len(artifacts_list)} instances...")
                    if artifacts_list:
                        from src.reporting.email_dispatcher import send_bundled_report_email
                        send_bundled_report_email(artifacts_list, today_str)
                    # Lock for 24 hours so it won't trigger again today
                    cache.set(cache_key, True, timeout=86400)
                    logger.info("[JARVIS SYSTEM] EOD Report Generated & Emailed Successfully.")
                except Exception as e:
                    logger.error(f"[JARVIS SYSTEM] EOD Generation Failed: {e}", exc_info=True)
            
            was_open = is_open
            
        except Exception as e:
            logger.debug(f"[EOD Daemon] Tick Error (silencable offline polling): {e}")
            broker = None # Reset broker connection if something went wrong
            
        # Poll every 60 seconds
        time.sleep(60)

def _telemetry_watcher_daemon():
    """
    Background daemon that periodically updates system CPU usage in the cache.
    This avoids blocking the HTTP request thread when loading the dashboard.
    """
    import django
    if not django.apps.apps.ready:
        time.sleep(5)
        
    from django.core.cache import cache
    try:
        import psutil
    except ImportError:
        return

    while True:
        try:
            cpu = psutil.cpu_percent(interval=1.0)
            cache.set("system_telemetry_cpu", cpu, timeout=10)
            time.sleep(2.0)
        except Exception as e:
            logger.debug(f"[Telemetry Daemon] Error: {e}")
            time.sleep(5)

def _auto_training_daemon():
    """
    Background daemon that automatically triggers model training when needed.
    Checks every 6 hours. Trains if:
    - No training has completed in the last 3 days
    - OR no models exist at all
    """
    import django
    if not django.apps.apps.ready:
        time.sleep(10)

    # Wait 2 minutes after startup before first check
    time.sleep(120)

    from django.utils import timezone as tz
    from datetime import timedelta

    while True:
        try:
            from control_panel.models import TrainingJob
            import subprocess, sys

            # Check if any training completed recently (last 3 days by checking recent IDs)
            recent_completed = TrainingJob.objects.filter(
                status='COMPLETED'
            ).order_by('-id').first()

            has_recent_training = False
            if recent_completed:
                # Check if the most recent completed job's model weights exist
                # and if there have been any completed jobs (we use a simple heuristic:
                # if the last completed job is within the most recent 5 jobs, training is fresh)
                recent_job_ids = list(TrainingJob.objects.order_by('-id').values_list('id', flat=True)[:5])
                has_recent_training = recent_completed.id in recent_job_ids

            has_any_model = TrainingJob.objects.filter(
                model_weights__isnull=False
            ).exists()

            already_running = TrainingJob.objects.filter(
                status='RUNNING'
            ).exists()

            if not already_running and (not has_recent_training or not has_any_model):
                logger.info("[AUTO-TRAIN] No recent training detected. Spawning automatic training job...")

                # Create a training job with balanced defaults
                job = TrainingJob.objects.create(
                    name=f"Auto-Train {tz.now().strftime('%Y-%m-%d')}",
                    feature_set_key='all_in',
                    hyperparameter_key='balanced',
                    ticker='SPY',
                    window_size=10,
                    training_duration_days=365,
                    initial_cash=100000.0,
                    status='PENDING'
                )

                # Spawn training subprocess
                from control_panel.views import _spawn_background_process
                log_path = os.path.join('logs', f'auto_train_job_{job.id}.log')
                os.makedirs('logs', exist_ok=True)
                cmd = [sys.executable, 'run_training.py', '--job_id', str(job.id)]
                process = _spawn_background_process(cmd, log_path)
                job.celery_task_id = str(process.pid)
                job.status = 'RUNNING'
                job.save()
                logger.info(f"[AUTO-TRAIN] Training job #{job.id} launched (PID: {process.pid})")
            else:
                logger.debug("[AUTO-TRAIN] Models are fresh. No training needed.")

        except Exception as e:
            logger.debug(f"[AUTO-TRAIN] Check error: {e}")

        # Check every 6 hours
        time.sleep(6 * 3600)


def _model_recommendation_daemon():
    """
    Background daemon that scores models based on evaluation results
    and caches the top recommendation for the Models Hub.
    Runs every 30 minutes.
    """
    import django
    if not django.apps.apps.ready:
        time.sleep(10)

    time.sleep(60)  # Wait 1 min after startup

    from django.core.cache import cache

    while True:
        try:
            from control_panel.models import EvaluationJob, TrainingJob

            # Find all completed evaluations with results
            evals = EvaluationJob.objects.filter(
                status='COMPLETED',
                results__isnull=False
            ).order_by('-id')

            model_scores = {}
            for ev in evals:
                ref = ev.model_file
                if ref in model_scores:
                    continue  # Only use most recent eval per model
                results = ev.results or {}
                sharpe = float(results.get('sharpe_ratio', 0) or 0)
                ret = float(results.get('total_return_pct', 0) or 0)
                drawdown = abs(float(results.get('max_drawdown_pct', 0) or 0))
                trades = int(results.get('total_trades', 0) or 0)

                # Composite score: high Sharpe + high return - high drawdown + activity
                score = (sharpe * 40) + (ret * 0.5) - (drawdown * 0.3) + min(trades * 0.1, 5)
                model_scores[ref] = {
                    'score': round(score, 2),
                    'sharpe': round(sharpe, 3),
                    'return_pct': round(ret, 2),
                    'max_drawdown': round(drawdown, 2),
                    'trades': trades,
                    'eval_id': ev.id
                }

            if model_scores:
                # Sort by score descending
                ranked = sorted(model_scores.items(), key=lambda x: x[1]['score'], reverse=True)
                recommendations = {}
                for ref, data in ranked:
                    grade = 'A' if data['score'] > 20 else 'B' if data['score'] > 10 else 'C' if data['score'] > 0 else 'D'
                    recommendations[ref] = {**data, 'grade': grade}

                cache.set('model_recommendations', recommendations, timeout=3600)
                best = ranked[0] if ranked else None
                if best:
                    logger.debug(f"[MODEL REC] Top model: {best[0]} (Score: {best[1]['score']}, Sharpe: {best[1]['sharpe']})")

        except Exception as e:
            logger.debug(f"[MODEL REC] Scoring error: {e}")

        time.sleep(1800)  # Every 30 minutes


def _neural_evolution_daemon():
    """
    Background daemon that automatically triggers neural evolution (code mutations)
    when running trading bots exist. Runs every 12 hours.
    Creates a ModelVariant by spawning the code_rewriter subprocess.
    """
    import django
    if not django.apps.apps.ready:
        time.sleep(10)

    # Wait 10 minutes after startup before first check (let bots warm up)
    time.sleep(600)

    while True:
        try:
            from control_panel.models import PaperTrader
            import subprocess, sys
            from pathlib import Path

            # Only trigger if there are running bots to evolve from
            running_bots = PaperTrader.objects.filter(status='RUNNING')
            if not running_bots.exists():
                logger.info("[EVOLUTION] No running bots. Skipping mutation cycle.")
                time.sleep(12 * 3600)
                continue

            # Don't trigger if a mutation is already in progress
            try:
                from control_panel.models import ModelVariant
                active_variants = ModelVariant.objects.filter(status='TESTING').count()
                if active_variants >= 3:
                    logger.info(f"[EVOLUTION] {active_variants} variants already testing. Skipping.")
                    time.sleep(12 * 3600)
                    continue
            except Exception:
                pass  # ModelVariant table might not exist yet

            logger.info("[EVOLUTION] Auto-triggering neural evolution cycle...")

            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / "auto_mutation.log"

            # Inline subprocess spawn (avoid circular import from views)
            project_root = Path(__file__).resolve().parent.parent
            cmd = [sys.executable, str(project_root / "src" / "core" / "code_rewriter.py")]
            
            try:
                log_handle = open(log_file, "a", encoding="utf-8")
                popen_kwargs = {
                    'cwd': str(project_root),
                    'stdout': log_handle,
                    'stderr': subprocess.STDOUT,
                    'stdin': subprocess.DEVNULL,
                    'start_new_session': True,
                }
                if os.name == 'nt':
                    del popen_kwargs['start_new_session']
                    popen_kwargs['creationflags'] = (
                        getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
                        | getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                    )
                
                process = subprocess.Popen(cmd, **popen_kwargs)
                logger.info(f"[EVOLUTION] Neural mutation subprocess launched (PID: {process.pid}). Logs: {log_file}")
            except Exception as spawn_err:
                logger.error(f"[EVOLUTION] Failed to spawn mutation process: {spawn_err}")

        except Exception as e:
            logger.error(f"[EVOLUTION] Auto-evolution error: {e}", exc_info=True)

        # Run every 12 hours
        time.sleep(12 * 3600)


class ControlPanelConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "control_panel"

    def ready(self):
        # We only want to start the daemon if we are running the main web server process.
        # Block engine subprocesses which inherit RUN_MAIN=true from the parent Django dev server.
        argv_str = ' '.join(os.sys.argv)
        is_webserver = 'runserver' in argv_str or 'gunicorn' in argv_str or 'uvicorn' in argv_str or 'daphne' in argv_str
        if not is_webserver:
            return
        
        # Start daemon threads if we are not in the runserver parent reloader process
        is_reloader_parent = 'runserver' in argv_str and '--noreload' not in argv_str and os.environ.get('RUN_MAIN', None) != 'true'
        if not is_reloader_parent:
            daemon_thread = threading.Thread(target=_eod_watcher_daemon, daemon=True)
            daemon_thread.start()
            logger.info("[JARVIS SYSTEM] EOD Market Monitor Daemon initialized & watching the tape.")
            
            telemetry_thread = threading.Thread(target=_telemetry_watcher_daemon, daemon=True)
            telemetry_thread.start()
            logger.info("[JARVIS SYSTEM] Telemetry Poller Daemon initialized.")

            auto_train_thread = threading.Thread(target=_auto_training_daemon, daemon=True)
            auto_train_thread.start()
            logger.info("[JARVIS SYSTEM] Auto-Training Scheduler Daemon initialized.")

            model_rec_thread = threading.Thread(target=_model_recommendation_daemon, daemon=True)
            model_rec_thread.start()
            logger.info("[JARVIS SYSTEM] Model Recommendation Engine initialized.")

            evolution_thread = threading.Thread(target=_neural_evolution_daemon, daemon=True)
            evolution_thread.start()
            logger.info("[JARVIS SYSTEM] Neural Evolution Daemon initialized (12h cycle).")

