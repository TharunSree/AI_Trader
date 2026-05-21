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
            if not clock:
                time.sleep(60)
                continue
                
            is_open = clock.get("is_open", False)
            
            # Detect transition from OPEN -> CLOSED
            if was_open and not is_open:
                today_str = date.today().isoformat()
                cache_key = f"eod_generated_{today_str}"
                
                # Use Redis/Memory cache to prevent multiple threads from firing simultaneously
                if not cache.get(cache_key):
                    logger.info("[JARVIS SYSTEM] Market Close Detected. Automatically triggering EOD Report Generation...")
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

