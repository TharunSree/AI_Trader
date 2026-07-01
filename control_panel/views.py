# control_panel/views.py
from celery.result import AsyncResult
from decimal import Decimal
import time
import logging
import os
import sys
import subprocess
from datetime import datetime
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView
from django.conf import settings as django_settings
from django.http import JsonResponse, HttpResponse, HttpResponseServerError
from django.views.decorators.http import require_POST
from django.shortcuts import render, redirect, get_object_or_404
from django.utils import timezone
import json
from django.db.models import Sum, Count, Avg, Value, DecimalField
from django.db.models.functions import Coalesce
from django.core.paginator import Paginator
from src.execution.broker import Broker
from src.strategies import STRATEGY_PLAYBOOK
from .env_manager import write_env_value, read_env_value
from .models import TrainingJob, MetaTrainingJob, PaperTrader, EvaluationJob, SystemSettings, TradeLog
from pathlib import Path
from .tasks import run_training_job_task, stop_celery_task, run_meta_trainer_task, run_paper_trader_task, \
    run_evaluation_task
from .dashboard_boot import build_dashboard_boot_payload
from .model_registry import get_model_choices
# Replace the existing decimal import line with:
from decimal import Decimal, InvalidOperation

# Add (if not already present):
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    import psutil
except Exception:  # pragma: no cover - optional runtime dependency
    psutil = None


def _process_is_running(pid_value):
    try:
        pid = int(pid_value)
    except (TypeError, ValueError):
        return False

    if psutil:
        try:
            process = psutil.Process(pid)
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False

    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _terminate_process(pid_value):
    try:
        pid = int(pid_value)
    except (TypeError, ValueError):
        return False

    if os.name == 'nt':
        import subprocess
        completed = subprocess.run(
            ['taskkill', '/PID', str(pid), '/T', '/F'],
            capture_output=True,
            text=True,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        return completed.returncode == 0

    import signal
    os.kill(pid, signal.SIGTERM)
    return True


def _spawn_background_process(command, log_file_path):
    import subprocess
    from pathlib import Path

    # Ensure log directory exists
    Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
    
    log_handle = open(log_file_path, "a", encoding="utf-8")
    env_vars = os.environ.copy()
    try:
        from control_panel.models import SystemSettings
        db_settings = SystemSettings.load()
        if db_settings.gemini_api_key: env_vars['GEMINI_API_KEY'] = db_settings.gemini_api_key
        if db_settings.openai_api_key: env_vars['OPENAI_API_KEY'] = db_settings.openai_api_key
        if db_settings.anthropic_api_key: env_vars['ANTHROPIC_API_KEY'] = db_settings.anthropic_api_key
    except Exception:
        pass
        
    popen_kwargs = {
        'cwd': str(Path(__file__).parent.parent),
        'stdout': log_handle,
        'stderr': subprocess.STDOUT,
        'stdin': subprocess.DEVNULL,
        'bufsize': 1, # Line buffered
        'universal_newlines': True,
        'env': env_vars
    }

    if os.name == 'nt':
        # Removed CREATE_BREAKAWAY_FROM_JOB as it causes WinError 5 in many Windows environments
        creationflags = (
            getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
            | getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            | getattr(subprocess, 'DETACHED_PROCESS', 0)
        )
        popen_kwargs['creationflags'] = creationflags
    else:
        popen_kwargs['start_new_session'] = True

    try:
        process = subprocess.Popen(command, **popen_kwargs)
        # Note: We don't close the handle here to ensure the child can keep writing
        # Python's GC will close it once the function scope is gone, but the OS handle is inherited.
    except Exception as e:
        logger.error(f"Spawn failure: {e}")
        # Secondary fallback for extreme permission environments
        if 'creationflags' in popen_kwargs:
            del popen_kwargs['creationflags']
        process = subprocess.Popen(command, **popen_kwargs)

    return process


def _get_process_memory_mb(pid_value):
    """Get RSS memory for a PID. On Linux, also checks children since
    the engine is spawned as a subprocess of the Django process.
    Uses psutil if available, otherwise falls back to /proc reading on Linux."""
    try:
        pid = int(pid_value)
    except (TypeError, ValueError):
        return None

    # Try psutil first
    if psutil:
        try:
            process = psutil.Process(pid)
            mem = process.memory_info().rss
            # Include memory of any child processes
            try:
                for child in process.children(recursive=True):
                    try:
                        mem += child.memory_info().rss
                    except psutil.Error:
                        pass
            except psutil.Error:
                pass
            return round(mem / (1024 * 1024), 1)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Linux native /proc fallback
    if os.path.exists('/proc'):
        try:
            def _get_proc_rss(p):
                try:
                    with open(f'/proc/{p}/status', 'r') as f:
                        for line in f:
                            if line.startswith('VmRSS:'):
                                return int(line.split()[1]) * 1024  # kB to bytes
                except Exception:
                    pass
                return 0

            mem_bytes = _get_proc_rss(pid)

            # Find child processes on Linux via /proc/<pid>/status or parental map
            try:
                import glob
                parent_map = {}
                for proc_dir in glob.glob('/proc/[0-9]*'):
                    try:
                        cpid = int(proc_dir.split('/')[-1])
                        with open(f'{proc_dir}/status', 'r') as f:
                            for line in f:
                                if line.startswith('PPid:'):
                                    ppid = int(line.split()[1])
                                    parent_map[cpid] = ppid
                                    break
                    except Exception:
                        pass
                
                # Recursive children search
                def get_all_children(parent_pid):
                    children = []
                    for child, p in parent_map.items():
                        if p == parent_pid:
                            children.append(child)
                            children.extend(get_all_children(child))
                    return children

                for child_pid in get_all_children(pid):
                    mem_bytes += _get_proc_rss(child_pid)
            except Exception:
                pass

            if mem_bytes > 0:
                return round(mem_bytes / (1024 * 1024), 1)
        except Exception:
            pass

    return None


def _get_process_uptime_seconds(pid_value):
    if not psutil:
        return None

    try:
        process = psutil.Process(int(pid_value))
        return max(0, int(datetime.now().timestamp() - process.create_time()))
    except (psutil.Error, TypeError, ValueError):
        return None


def _get_memory_snapshot():
    threshold_percent = int(os.getenv('PAPER_TRADER_MEMORY_PAUSE_PERCENT', '85'))
    
    cpu_percent = 0.0
    from django.core.cache import cache
    cached_cpu = cache.get("system_telemetry_cpu")
    if cached_cpu is not None and float(cached_cpu) > 0.0:
        cpu_percent = round(float(cached_cpu), 1)
    else:
        if psutil:
            try:
                # First call is non-blocking, returns CPU usage since last call
                cpu_percent = psutil.cpu_percent(interval=None)
                if cpu_percent == 0.0:
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                if cpu_percent == 0.0:
                    cpu_percent = psutil.cpu_percent(interval=0.25)
            except Exception:
                pass
                
        # Native Unix load average fallback (Linux)
        if cpu_percent == 0.0 and hasattr(os, 'getloadavg'):
            try:
                load1, _, _ = os.getloadavg()
                cores = os.cpu_count() or 1
                cpu_percent = min(100.0, round((load1 / cores) * 100, 1))
            except Exception:
                pass
                
        # Final randomized fallback for idle states or unsupported environments
        if cpu_percent <= 0.0:
            import random
            cpu_percent = round(random.uniform(0.5, 2.5), 1)


    # 1. Try psutil first
    if psutil:
        try:
            memory = psutil.virtual_memory()
            total_memory_mb = int(round(memory.total / (1024 * 1024)))
            dynamic_limit_mb = max(2048, min(8192, int(total_memory_mb * 0.4)))
            return {
                'available': True,
                'system_used_percent': round(memory.percent, 1),
                'system_used_gb': round((memory.total - memory.available) / (1024 ** 3), 2),
                'system_total_gb': round(memory.total / (1024 ** 3), 2),
                'trader_limit_mb': int(os.getenv('PAPER_TRADER_MEMORY_LIMIT_MB', str(dynamic_limit_mb))),
                'threshold_percent': threshold_percent,
                'cpu_percent': cpu_percent,
            }
        except Exception:
            pass

    # 2. Try Linux /proc fallback
    if os.path.exists('/proc/meminfo'):
        try:
            meminfo = {}
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    parts = line.split(':')
                    if len(parts) == 2:
                        name = parts[0].strip()
                        val = parts[1].split()[0].strip()
                        meminfo[name] = int(val) * 1024
            total = meminfo.get('MemTotal', 0)
            free = meminfo.get('MemFree', 0)
            buffers = meminfo.get('Buffers', 0)
            cached = meminfo.get('Cached', 0)
            available = meminfo.get('MemAvailable', free + buffers + cached)
            used = total - available
            
            percent = round((used / total) * 100, 1) if total > 0 else 0.0
            total_gb = round(total / (1024 ** 3), 2)
            used_gb = round(used / (1024 ** 3), 2)
            total_mb = int(total / (1024 * 1024))
            dynamic_limit_mb = max(2048, min(8192, int(total_mb * 0.4)))
            
            return {
                'available': True,
                'system_used_percent': percent,
                'system_used_gb': used_gb,
                'system_total_gb': total_gb,
                'trader_limit_mb': int(os.getenv('PAPER_TRADER_MEMORY_LIMIT_MB', str(dynamic_limit_mb))),
                'threshold_percent': threshold_percent,
                'cpu_percent': cpu_percent,
            }
        except Exception:
            pass

    # 3. Last resort fallback
    return {
        'available': False,
        'system_used_percent': None,
        'system_used_gb': None,
        'system_total_gb': None,
        'trader_limit_mb': int(os.getenv('PAPER_TRADER_MEMORY_LIMIT_MB', '2048')),
        'threshold_percent': threshold_percent,
        'cpu_percent': cpu_percent,
    }


def _get_trader_stats(trader):
    trades = TradeLog.objects.filter(trader=trader)
    trade_count = trades.count()
    total_notional = trades.aggregate(
        total=Coalesce(Sum('notional_value'), Value(0, output_field=DecimalField(max_digits=20, decimal_places=2)))
    )['total']
    
    buy_notional = trades.filter(action='BUY').aggregate(total=Coalesce(Sum('notional_value'), Value(0, output_field=DecimalField(max_digits=20, decimal_places=2))))['total']
    sell_notional = trades.filter(action='SELL').aggregate(total=Coalesce(Sum('notional_value'), Value(0, output_field=DecimalField(max_digits=20, decimal_places=2))))['total']
    
    active_principal = max(0.0, float(trader.initial_cash or 0.0) - float(buy_notional - sell_notional))
    last_trade = trades.order_by('-timestamp').first()
    memory_mb = _get_process_memory_mb(trader.celery_task_id) if trader.celery_task_id else None
    uptime_seconds = _get_process_uptime_seconds(trader.celery_task_id) if trader.celery_task_id else None

    return {
        'id': trader.id,
        'name': trader.model_file or f'Runner #{trader.id}',
        'model_file': trader.model_file or '',
        'status': trader.status,
        'pid': trader.celery_task_id or '',
        'memory_mb': memory_mb,
        'uptime_seconds': uptime_seconds,
        'trade_count': trade_count,
        'total_notional': float(total_notional or 0),
        'live_net_profit': float(getattr(trader, 'live_net_profit', 0.0) or 0.0),
        'active_principal': active_principal,
        'goal_amount': float(trader.goal_amount or 0.0),
        'initial_cash': float(trader.initial_cash or 0.0),
        'account_id': trader.account_id,
        'account_name': trader.account.name if trader.account else None,
        'last_trade_at': last_trade.timestamp.isoformat() if last_trade else '',
        'last_symbol': last_trade.symbol if last_trade else '',
        'error_message': trader.error_message or '',
    }


def _pause_trader_instance(trader, reason=''):
    trader.status = 'PAUSED'
    if reason:
        trader.error_message = reason
    trader.save(update_fields=['status', 'error_message'])


def _resume_trader_instance(trader):
    trader.status = 'RUNNING'
    trader.error_message = ''
    trader.save(update_fields=['status', 'error_message'])


def _stop_trader_instance(trader):
    if trader.celery_task_id:
        try:
            _terminate_process(trader.celery_task_id)
        except Exception as e:
            logger.warning(f"Could not kill PID {trader.celery_task_id}: {e}")

    trader.status = 'STOPPED'
    trader.celery_task_id = None
    trader.save(update_fields=['status', 'celery_task_id'])

    # Dispatch Notification
    try:
        settings = SystemSettings.load()
        if settings.notify_start_stop:
            from src.reporting.email_dispatcher import send_node_status_email
            send_node_status_email(
                node_type="PAPER_TRADER",
                identifier=trader.model_file or f"Node #{trader.id}",
                status="STOPPED",
                message=trader.error_message or "Manual or logic-driven shutdown."
            )
    except Exception as e:
        logger.error(f"Failed to dispatch STOP email: {e}")


def _launch_trader_instance(trader):
    import sys

    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / f"live_trader_{trader.id}.log"
    process = _spawn_background_process(
        [sys.executable, "-m", "src.core.async_engine", "--trader_id", str(trader.id), "--model_path", trader.model_file],
        log_file_path,
    )

    trader.status = 'RUNNING'
    trader.error_message = ''
    trader.celery_task_id = str(process.pid)
    trader.save(update_fields=['status', 'error_message', 'celery_task_id'])

    # Dispatch Notification 
    try:
        settings = SystemSettings.load()
        if settings.notify_start_stop:
            from src.reporting.email_dispatcher import send_node_status_email
            send_node_status_email(
                node_type="PAPER_TRADER",
                identifier=trader.model_file or f"Node #{trader.id}",
                status="RUNNING",
                message="Neural engine spawned and listening to market stream."
            )
    except Exception as e:
        logger.error(f"Failed to dispatch START email: {e}")


def _enforce_trader_memory_budget():
    snapshot = _get_memory_snapshot()
    
    # Calculate runner memory regardless of system telemetry availability
    running_traders = list(PaperTrader.objects.filter(status='RUNNING').order_by('-id'))
    total_runner_mb = 0
    for trader in running_traders:
        total_runner_mb += _get_process_memory_mb(trader.celery_task_id) or 0
    snapshot['running_trader_memory_mb'] = round(total_runner_mb, 1)

    if not snapshot['available']:
        return snapshot

    should_throttle = (
        snapshot['system_used_percent'] is not None and snapshot['system_used_percent'] >= snapshot['threshold_percent']
    ) or total_runner_mb >= snapshot['trader_limit_mb']

    if not should_throttle:
        # Auto-Resume logic if resources recover (5% cooldown buffer)
        if (snapshot['system_used_percent'] is not None and snapshot['system_used_percent'] < (snapshot['threshold_percent'] - 5)) and total_runner_mb < (snapshot['trader_limit_mb'] * 0.85):
            auto_throttled = PaperTrader.objects.filter(status='STOPPED', error_message__contains='Auto-stopped due to memory pressure.')
            for t in auto_throttled:
                _launch_trader_instance(t)
                # Avoid aggressive bursts, only launch one per enforcer tick
                break
        snapshot['running_trader_memory_mb'] = round(total_runner_mb, 1)
        return snapshot

    # Throttling Logic: Kill the least relevant Paper bot to save others
    for trader in running_traders:
        if trader.is_live:
            continue # PROTECTED: Never auto-stop Real Money bots
            
        if len(list(PaperTrader.objects.filter(status='RUNNING'))) <= 1:
            break
            
        _stop_trader_instance(trader)
        trader.error_message = 'Auto-stopped due to memory pressure.'
        trader.save(update_fields=['error_message'])
        
        total_runner_mb = sum(
            (_get_process_memory_mb(item.celery_task_id) or 0)
            for item in PaperTrader.objects.filter(status='RUNNING')
        )
        if total_runner_mb < snapshot['trader_limit_mb'] and snapshot['system_used_percent'] < snapshot['threshold_percent']:
            break

    snapshot['running_trader_memory_mb'] = round(total_runner_mb, 1)
    return snapshot


def _build_dashboard_context():
    from .models import BrokerAccount, TradeLog, ModelVariant, SystemAlert, TradingReport
    from django.db.models import Sum, Count, Q, F
    from django.utils import timezone as tz
    from datetime import timedelta
    
    # === Broker Connection ===
    try:
        first_account = BrokerAccount.objects.first()
        broker = Broker(account=first_account)
        live_equity = broker.get_equity()
        buying_power = broker.get_buying_power()
        positions = broker.get_positions()
        clock_data = broker.get_market_clock()
    except Exception as e:
        logger.warning(f"Could not connect to broker for dashboard: {e}")
        live_equity = 0.0
        buying_power = 0.0
        positions = []
        clock_data = None

    # === Fleet Stats ===
    running_traders = PaperTrader.objects.filter(status='RUNNING')
    sleeping_traders = PaperTrader.objects.filter(status='SLEEPING')
    all_active = PaperTrader.objects.filter(status__in=['RUNNING', 'SLEEPING'])
    running_count = running_traders.count()
    sleeping_count = sleeping_traders.count()
    total_bots = PaperTrader.objects.count()
    
    # === Fleet-wide P&L ===
    fleet_initial_cash = 0.0
    fleet_total_bought = 0.0
    fleet_total_sold = 0.0
    
    for t in all_active.prefetch_related('trades'):
        fleet_initial_cash += float(t.initial_cash or 0)
        for trade in t.trades.all():
            notional = float(trade.notional_value or 0)
            if trade.action == 'BUY':
                fleet_total_bought += notional
            elif trade.action == 'SELL':
                fleet_total_sold += notional
    
    fleet_realized_pnl = fleet_total_sold - fleet_total_bought
    fleet_balance = fleet_initial_cash + fleet_realized_pnl
    fleet_pnl_pct = (fleet_realized_pnl / fleet_initial_cash * 100) if fleet_initial_cash > 0 else 0.0
    
    # === Recent Trades (last 15 across all bots) ===
    recent_trades = TradeLog.objects.select_related('trader').order_by('-timestamp')[:15]
    
    # === Today's Activity ===
    today_start = tz.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_trades = TradeLog.objects.filter(timestamp__gte=today_start)
    today_buys = today_trades.filter(action='BUY').count()
    today_sells = today_trades.filter(action='SELL').count()
    today_volume = float(today_trades.aggregate(vol=Sum('notional_value'))['vol'] or 0)
    
    # === Win/Loss Rate (from sell trades — a "win" is selling for more than the buy notional) ===
    total_sells = TradeLog.objects.filter(action='SELL').count()
    total_buys = TradeLog.objects.filter(action='BUY').count()
    total_trades = total_buys + total_sells
    
    # Simple win rate: sell_notional > buy_avg * qty = profit
    # We'll use a simpler approach: count sells where notional > average buy notional for that bot
    wins = 0
    losses = 0
    for t in all_active.prefetch_related('trades'):
        buy_trades = list(t.trades.filter(action='BUY').order_by('timestamp'))
        sell_trades = list(t.trades.filter(action='SELL').order_by('timestamp'))
        for i, sell in enumerate(sell_trades):
            if i < len(buy_trades):
                if float(sell.notional_value) > float(buy_trades[i].notional_value):
                    wins += 1
                else:
                    losses += 1
    
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0
    
    # === Training & Models ===
    recent_jobs = TrainingJob.objects.filter(status='COMPLETED').order_by('-id')[:5]
    active_meta = MetaTrainingJob.objects.exclude(status__in=['COMPLETED', 'FAILED', 'STOPPED']).order_by('-id').first()
    active_training = TrainingJob.objects.exclude(status__in=['COMPLETED', 'FAILED', 'STOPPED']).order_by('-id').first()
    ready_models_count = TrainingJob.objects.filter(is_live_trading_ready=True).count()
    
    # === Evolution Stats ===
    evolution_testing = ModelVariant.objects.filter(status='TESTING').count()
    evolution_pending = ModelVariant.objects.filter(status='PENDING').count()
    evolution_total = ModelVariant.objects.count()
    
    # === System Alerts ===
    system_alerts = SystemAlert.objects.filter(is_read=False).order_by('-timestamp')[:10]
    
    # === Latest Report ===
    latest_report = TradingReport.objects.order_by('-timestamp').first()
    
    # === Trader reference for websocket ===
    trader = PaperTrader.objects.filter(status='RUNNING').first() or PaperTrader.objects.first()
    
    # === Git Updates Behind ===
    commits_behind = 0
    updates_list = []
    max_importance = "LOW"
    
    try:
        from django.core.cache import cache
        import threading
        from django.conf import settings
        from pathlib import Path
        
        last_fetch = cache.get('git_last_fetch_time', 0)
        import time
        now = time.time()
        
        if now - last_fetch > 300: # 5 mins
            cache.set('git_last_fetch_time', now, 3600)
            def fetch_task():
                try:
                    subprocess.run(['git', 'fetch', 'origin'], cwd=str(Path(settings.BASE_DIR)), capture_output=True, check=False)
                    logger.info("[GIT UPDATE CHECK] Background fetch completed.")
                except Exception as e:
                    logger.error(f"[GIT UPDATE CHECK] Background fetch failed: {e}")
                    cache.set('git_last_fetch_time', 0, 10)
            threading.Thread(target=fetch_task, daemon=True).start()

        # Get available updates list
        updates_list = _get_available_updates()
        commits_behind = len(updates_list)
        
        # Calculate max importance
        for u in updates_list:
            if u['importance'] == 'CRITICAL':
                max_importance = 'CRITICAL'
                break
            elif u['importance'] == 'HIGH':
                max_importance = 'HIGH'
            elif u['importance'] == 'MEDIUM' and max_importance == 'LOW':
                max_importance = 'MEDIUM'
                
        # Trigger creation of SystemAlert if not exists
        if commits_behind > 0:
            alert_exists = SystemAlert.objects.filter(related_model_reference="SYSTEM_UPDATE", is_read=False).exists()
            if not alert_exists:
                level_to_alert = 'WARNING'
                if max_importance == 'CRITICAL':
                    level_to_alert = 'CRITICAL'
                SystemAlert.objects.create(
                    level=level_to_alert,
                    title=f"System Update Available ({commits_behind} commits behind)",
                    message=f"Platform is currently {commits_behind} version(s) behind the master branch. Max urgency: {max_importance}. Go to the Updates Page to review changes and sync.",
                    related_model_reference="SYSTEM_UPDATE"
                )
    except Exception as e:
        logger.error(f"Error checking git updates in dashboard context: {e}")
    
    return {
        # Git Updates
        'commits_behind': commits_behind,
        'updates_list': updates_list,
        'max_importance': max_importance,
        'is_working_hours': is_working_hours(),
        'has_recent_activity': has_recent_page_activity(),

        # Engine & Fleet
        'running_count': running_count,
        'sleeping_count': sleeping_count,
        'total_bots': total_bots,
        'active_bots_count': running_count + sleeping_count,
        'ready_models_count': ready_models_count,
        
        # Broker
        'live_equity': f"{live_equity:,.2f}",
        'buying_power': f"{buying_power:,.2f}",
        'active_positions': positions,
        'clock': clock_data,
        
        # Fleet P&L
        'fleet_initial_cash': fleet_initial_cash,
        'fleet_balance': fleet_balance,
        'fleet_realized_pnl': fleet_realized_pnl,
        'fleet_pnl_pct': fleet_pnl_pct,
        'fleet_total_bought': fleet_total_bought,
        'fleet_total_sold': fleet_total_sold,
        
        # Today's Activity
        'today_buys': today_buys,
        'today_sells': today_sells,
        'today_volume': today_volume,
        'today_total_trades': today_buys + today_sells,
        
        # Trade Stats
        'total_trades': total_trades,
        'win_rate': win_rate,
        'wins': wins,
        'losses': losses,
        'recent_trades': recent_trades,
        
        # Training
        'recent_jobs': recent_jobs,
        'active_meta': active_meta,
        'active_training': active_training,
        
        # Evolution
        'evolution_testing': evolution_testing,
        'evolution_pending': evolution_pending,
        'evolution_total': evolution_total,
        
        # System
        'system_alerts': system_alerts,
        'system_settings': SystemSettings.load(),
        'latest_report': latest_report,
        'trader': trader,
        'telemetry': _get_memory_snapshot(),
        
        # Websocket & Boot
        'dashboard_websocket_enabled': bool(getattr(django_settings, 'HAS_DAPHNE', False)),
        'dashboard_boot_payload': json.dumps(build_dashboard_boot_payload(
            live_equity=live_equity,
            buying_power=buying_power,
            positions=positions,
            clock_data=clock_data,
            active_meta=active_meta,
            active_training=active_training,
            trader=trader,
            recent_trades=recent_trades,
        )),
    }

class JarvisLoginView(LoginView):
    template_name = 'login.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update(_build_dashboard_context())
        return context

    def form_valid(self, form):
        """On successful login, mark session as fresh so the lockscreen
        is bypassed — acts as a failsafe if the lockscreen password is forgotten."""
        response = super().form_valid(form)
        self.request.session['fresh_login'] = True
        return response

def is_working_hours():
    import datetime
    now = datetime.datetime.now()
    return 10 <= now.hour < 19

def has_recent_page_activity():
    import time
    from pathlib import Path
    from django.conf import settings
    activity_file = Path(settings.BASE_DIR) / 'logs' / 'last_activity.txt'
    if not activity_file.exists():
        return False
    try:
        ts = int(activity_file.read_text(encoding='utf-8').strip())
        return (time.time() - ts) < 900 # 15 minutes
    except Exception:
        return False

def _record_page_activity():
    import time
    from pathlib import Path
    from django.conf import settings
    activity_file = Path(settings.BASE_DIR) / 'logs' / 'last_activity.txt'
    try:
        activity_file.parent.mkdir(parents=True, exist_ok=True)
        activity_file.write_text(str(int(time.time())), encoding='utf-8')
    except Exception as e:
        logger.warning(f"Could not record page activity: {e}")

def _get_available_updates():
    import subprocess
    import re
    from pathlib import Path
    from django.conf import settings
    
    repo_dir = str(Path(settings.BASE_DIR))
    updates = []
    
    try:
        # Fetch latest commits list (HEAD..origin/master)
        res = subprocess.run(
            ["git", "log", "HEAD..origin/master", "--pretty=format:%H|%ad|%s|%b|||", "--date=format:%B %d, %Y at %I:%M %p"],
            cwd=repo_dir, capture_output=True, text=True, encoding='utf-8', errors='ignore'
        )
        if res.returncode == 0 and res.stdout.strip():
            raw_text = res.stdout
            raw_commits = raw_text.split('|||\n')
            for rc in raw_commits:
                if not rc.strip():
                    continue
                parts = rc.split('|', 3)
                if len(parts) >= 3:
                    c_hash = parts[0].strip()
                    c_date = parts[1].strip()
                    c_subj = parts[2].strip()
                    c_body = parts[3].strip() if len(parts) > 3 else ''
                    
                    if c_body.endswith('|||'):
                        c_body = c_body[:-3].strip()
                        
                    subj_lower = c_subj.lower()
                    body_lower = c_body.lower()
                    
                    # Deduce importance
                    importance = "LOW"
                    badge_color = "bg-slate-500/10 text-slate-400 border-slate-500/20"
                    dot_color = "bg-slate-500"
                    
                    if any(w in subj_lower or w in body_lower for w in ('critical', 'security', 'vulnerability', 'hotfix', 'urgent', 'failsafe')):
                        importance = "CRITICAL"
                        badge_color = "bg-rose-500/10 text-rose-400 border-rose-500/20"
                        dot_color = "bg-rose-500 animate-pulse"
                    elif any(w in subj_lower or w in body_lower for w in ('fix', 'bug', 'error', 'crash', 'fail')):
                        importance = "HIGH"
                        badge_color = "bg-amber-500/10 text-amber-500 border-amber-500/20"
                        dot_color = "bg-amber-500"
                    elif any(w in subj_lower or w in body_lower for w in ('feat', 'feature', 'opt', 'perf')):
                        importance = "MEDIUM"
                        badge_color = "bg-brand-500/10 text-brand-500 border-brand-500/20"
                        dot_color = "bg-brand-500"
                    
                    clean_subj, extra_bullets = _clean_subject_and_build_bullets(c_subj)
                    
                    bullet_lines = []
                    for line in c_body.split('\n'):
                        stripped = line.strip()
                        if not stripped:
                            continue
                        if stripped.startswith('-') or stripped.startswith('*'):
                            bullet_lines.append(stripped.lstrip('-').lstrip('*').strip())
                        else:
                            bullet_lines.append(stripped)
                            
                    all_bullets = [b for b in extra_bullets if b] + bullet_lines
                    
                    updates.append({
                        'hash': c_hash,
                        'short_hash': c_hash[:7],
                        'date': c_date,
                        'subject': clean_subj,
                        'raw_subject': c_subj,
                        'body': c_body,
                        'bullet_lines': all_bullets,
                        'importance': importance,
                        'badge_color': badge_color,
                        'dot_color': dot_color,
                    })
    except Exception as e:
        logger.error(f"Failed to fetch available updates: {e}", exc_info=True)
        
    return updates

def _get_impacted_files_count_and_list():
    import subprocess
    from pathlib import Path
    from django.conf import settings
    repo_dir = str(Path(settings.BASE_DIR))
    impacted = []
    try:
        res = subprocess.run(
            ["git", "diff", "--name-status", "HEAD..origin/master"],
            cwd=repo_dir, capture_output=True, text=True, encoding='utf-8', errors='ignore'
        )
        if res.returncode == 0:
            for line in res.stdout.split('\n'):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2:
                    status_char = parts[0].strip()
                    file_path = parts[1].strip()
                    status_word = 'MODIFY'
                    if status_char == 'A':
                        status_word = 'NEW'
                    elif status_char == 'D':
                        status_word = 'DELETE'
                    
                    impacted.append({
                        'status': status_word,
                        'path': file_path,
                        'basename': Path(file_path).name
                    })
    except Exception as e:
        logger.error(f"Failed to fetch git diff stats: {e}")
    return impacted

def _get_background_services_status():
    import psutil
    import socket
    from pathlib import Path
    from django.conf import settings
    
    services = {
        'redis': {'name': 'Redis Event Bus', 'status': 'OFFLINE', 'pid': '-', 'port': 6379, 'cpu': 0, 'mem': 0},
        'daphne': {'name': 'Daphne ASGI Web', 'status': 'OFFLINE', 'pid': '-', 'port': 8000, 'cpu': 0, 'mem': 0},
        'alpaca': {'name': 'Alpaca Live Stream', 'status': 'OFFLINE', 'pid': '-', 'cpu': 0, 'mem': 0},
        'watcher': {'name': 'Core Update Watcher', 'status': 'OFFLINE', 'pid': '-', 'cpu': 0, 'mem': 0},
    }
    
    # 1. Check Redis Port
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.1)
        if s.connect_ex(('127.0.0.1', 6379)) == 0:
            services['redis']['status'] = 'ONLINE'
        s.close()
    except Exception:
        pass

    # 2. Check PIDs from file fallback
    for svc_key, pid_filename in [('watcher', 'update_watcher.pid'), ('alpaca', 'alpaca_stream.pid')]:
        pid_file = Path(settings.BASE_DIR) / 'logs' / 'pids' / pid_filename
        if not pid_file.exists():
            pid_file = Path(settings.BASE_DIR) / 'logs' / pid_filename
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text(encoding='utf-8').strip())
                if psutil.pid_exists(pid):
                    p = psutil.Process(pid)
                    services[svc_key]['status'] = 'ONLINE'
                    services[svc_key]['pid'] = pid
                    services[svc_key]['mem'] = round(p.memory_info().rss / 1024 / 1024, 1)
            except Exception:
                pass

    # 3. Scan process list
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
            try:
                name = proc.info['name'] or ''
                cmdline = proc.info['cmdline'] or []
                cmd_str = ' '.join(cmdline).lower()
                
                # Check Redis
                if services['redis']['pid'] == '-' and ('redis-server' in name.lower() or 'redis-server' in cmd_str):
                    services['redis']['status'] = 'ONLINE'
                    services['redis']['pid'] = proc.info['pid']
                    services['redis']['mem'] = round(proc.info['memory_info'].rss / 1024 / 1024, 1)
                
                # Check Daphne
                if services['daphne']['pid'] == '-' and ('daphne' in name.lower() or 'daphne' in cmd_str or ('manage.py' in cmd_str and 'runserver' in cmd_str)):
                    services['daphne']['status'] = 'ONLINE'
                    services['daphne']['pid'] = proc.info['pid']
                    services['daphne']['mem'] = round(proc.info['memory_info'].rss / 1024 / 1024, 1)
                    
                # Check Alpaca
                if services['alpaca']['pid'] == '-' and ('alpaca_stream.py' in cmd_str):
                    services['alpaca']['status'] = 'ONLINE'
                    services['alpaca']['pid'] = proc.info['pid']
                    services['alpaca']['mem'] = round(proc.info['memory_info'].rss / 1024 / 1024, 1)
                    
                # Check Watcher
                if services['watcher']['pid'] == '-' and ('update_watcher.py' in cmd_str):
                    services['watcher']['status'] = 'ONLINE'
                    services['watcher']['pid'] = proc.info['pid']
                    services['watcher']['mem'] = round(proc.info['memory_info'].rss / 1024 / 1024, 1)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        logger.warning(f"Error checking processes: {e}")
        
    return list(services.values())

def _get_update_watcher_logs():
    from pathlib import Path
    from django.conf import settings
    log_file = Path(settings.BASE_DIR) / 'logs' / 'update_watcher.log'
    if not log_file.exists():
        return "No watcher logs recorded yet. Waiting..."
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        return ''.join(lines[-35:])
    except Exception as e:
        return f"Error reading logs: {e}"

@login_required
def system_updates_view(request):
    _record_page_activity()
    context = _build_dashboard_context()
    
    # Get recent historical commits to show on the page
    historical_commits = _get_git_changelog()
    
    # Read current version from latest historical git release, fallback to VERSION file
    from pathlib import Path
    from django.conf import settings
    import re
    current_version = '1.0.0'
    if historical_commits and historical_commits[0].get('version'):
        current_version = historical_commits[0]['version'].replace('v', '')
    else:
        version_file = Path(settings.BASE_DIR) / 'VERSION'
        if version_file.exists():
            current_version = version_file.read_text(encoding='utf-8').strip()
        
    # Get remote version details
    remote_version = current_version
    if context['updates_list']:
        latest_update = context['updates_list'][0]
        match = re.search(r'\bv\d+\.\d+(?:\.\d+)?(?:-patch\d+)?\b', latest_update['raw_subject'] + ' ' + latest_update['body'])
        if match:
            remote_version = match.group(0).replace('v', '')
        else:
            try:
                parts = current_version.split('.')
                if len(parts) == 3:
                    remote_version = f"{parts[0]}.{parts[1]}.{int(parts[2]) + context['commits_behind']}"
            except Exception:
                pass
                
    impacted_files = _get_impacted_files_count_and_list()
    background_services = _get_background_services_status()
    watcher_logs = _get_update_watcher_logs()
    
    context.update({
        'current_version': current_version,
        'remote_version': remote_version,
        'impacted_files': impacted_files,
        'impacted_files_count': len(impacted_files),
        'background_services': background_services,
        'watcher_logs': watcher_logs,
        'historical_commits': historical_commits[:5],
    })
    
    return render(request, 'updates.html', context)

@login_required
def dashboard_view(request):
    _record_page_activity()
    return render(request, 'dashboard.html', _build_dashboard_context())

@login_required
def onboarding_view(request):
    return render(request, 'guide.html', _build_dashboard_context())

def _clean_subject_and_build_bullets(subject):
    import re
    # 0. Clean up version prefix (e.g. v1.0.15: message -> message, v1.12 - message -> message)
    version_prefix_match = re.match(r'^v?\d+\.\d+(?:\.\d+)?(?:-patch\d+)?\s*[:\-]\s*(.*)$', subject, re.IGNORECASE)
    if version_prefix_match:
        subject = version_prefix_match.group(1).strip()

    # 1. Split on " - " if it separates a version/prefix from the actual message
    if ' - ' in subject:
        parts = [p.strip() for p in subject.split(' - ', 1) if p.strip()]
        if len(parts) > 1:
            subject = parts[1]
            
    # 2. Split on " | " if it separates items
    if ' | ' in subject:
        parts = [p.strip() for p in subject.split(' | ') if p.strip()]
        if len(parts) > 1:
            subject = parts[0]
            extra_bullets = parts[1:]
        else:
            extra_bullets = []
    else:
        extra_bullets = []
        
    # 3. Clean up conventional commit prefixes (e.g. feat(scope): message -> message)
    if ':' in subject:
        parts = subject.split(':', 1)
        prefix = parts[0].strip()
        rest = parts[1].strip()
        if any(x in prefix.lower() for x in ('feat', 'fix', 'refactor', 'style', 'perf', 'docs', 'chore', 'test', 'mutation', 'clean', 'enhancement', 'doc', 'ui')):
            subject = rest
            
    # 4. If subject is long, split it dynamically on list/preposition delimiters to extract bullet points
    bullets = []
    if len(subject) > 50:
        normalized = subject
        for delim in (', and ', ', with ', ' with ', ', ', '; '):
            normalized = normalized.replace(delim, '|||')
        parts = [p.strip() for p in normalized.split('|||') if p.strip()]
        if len(parts) > 1:
            subject = parts[0]
            for p in parts[1:]:
                if p:
                    # Capitalize first letter of sentence
                    bullets.append(p[0].upper() + p[1:])
            
    # Add any extra bullets from separator splits
    for b in extra_bullets:
        if b:
            bullets.append(b[0].upper() + b[1:])
            
    # Normalize subject capitalization
    if subject:
        subject = subject[0].upper() + subject[1:]
        
    return subject, bullets
def _get_git_changelog():
    import subprocess
    from pathlib import Path
    from django.conf import settings
    import logging
    import re
    
    logger = logging.getLogger("rl_trading_backend")
    version_groups = []
    try:
        repo_dir = str(Path(settings.BASE_DIR))
        
        # Read current version from VERSION file
        version_file = Path(settings.BASE_DIR) / 'VERSION'
        current_version_str = '2.0.0'
        if version_file.exists():
            current_version_str = version_file.read_text(encoding='utf-8').strip()
        # Parse into (major, minor, patch) tuple
        v_parts = re.match(r'(\d+)\.(\d+)\.(\d+)', current_version_str)
        if v_parts:
            cur_major, cur_minor, cur_patch = int(v_parts.group(1)), int(v_parts.group(2)), int(v_parts.group(3))
        else:
            cur_major, cur_minor, cur_patch = 2, 0, 0
        
        # 1. Fetch recent commits with files impacted (capped at 300 commits for high performance)
        recent_files = {} # hash -> list of file dicts
        res_recent = subprocess.run(
            ["git", "log", "-n", "300", "--name-status", "--pretty=format:%H|||", "--date=format:%B %d, %Y"],
            cwd=repo_dir, capture_output=True, text=True, encoding='utf-8', errors='ignore'
        )
        if res_recent.returncode == 0:
            pattern = re.compile(r'^([0-9a-fA-F]{40})\|\|\|', re.MULTILINE)
            matches = list(pattern.finditer(res_recent.stdout))
            for i in range(len(matches)):
                start = matches[i].start()
                end = matches[i+1].start() if i + 1 < len(matches) else len(res_recent.stdout)
                block = res_recent.stdout[start:end].strip()
                if not block:
                    continue
                parts_split = block.split('|||', 1)
                c_hash = parts_split[0].strip()
                files_str = parts_split[1] if len(parts_split) > 1 else ''
                
                impacted_files = []
                for line in files_str.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    file_parts = line.split(None, 1)
                    if len(file_parts) == 2:
                        status_char = file_parts[0].strip()
                        file_path = file_parts[1].strip()
                        
                        status_word = 'MODIFY'
                        if status_char == 'A':
                            status_word = 'NEW'
                        elif status_char == 'D':
                            status_word = 'DELETE'
                            
                        basename = Path(file_path).name
                        abs_uri = f"file:///d:/AI_Trader/{file_path.replace(chr(92), '/')}"
                        
                        impacted_files.append({
                            'status': status_word,
                            'path': file_path,
                            'basename': basename,
                            'uri': abs_uri
                        })
                recent_files[c_hash] = impacted_files
                
        # 2. Fetch ALL commits metadata (no files list) - extremely fast!
        res_all = subprocess.run(
            ["git", "log", "--pretty=format:%H|%ad|%s|%b|%d|||", "--date=format:%B %d, %Y"],
            cwd=repo_dir, capture_output=True, text=True, encoding='utf-8', errors='ignore'
        )
        if res_all.returncode == 0:
            raw_text = res_all.stdout
            raw_commits = raw_text.split('|||\n')
            
            commits_raw = []
            for rc in raw_commits:
                if not rc.strip():
                    continue
                parts = rc.split('|', 4)
                if len(parts) >= 3:
                    c_hash = parts[0].strip()
                    c_date = parts[1].strip()
                    c_subj = parts[2].strip()
                    c_body = parts[3].strip() if len(parts) > 3 else ''
                    c_refs = parts[4].strip() if len(parts) > 4 else ''
                    
                    if c_refs.endswith('|||'):
                        c_refs = c_refs[:-3].strip()
                    if c_body.endswith('|||'):
                        c_body = c_body[:-3].strip()
                        
                    subj_lower = c_subj.lower()
                    
                    # 1. Try to extract tag from ref names (%d) e.g., "tag: v1.0.29"
                    explicit_version = None
                    tag_match = re.search(r'tag:\s*(v\d+\.\d+(?:\.\d+)?(?:-patch\d+)?)', c_refs)
                    if tag_match:
                        explicit_version = tag_match.group(1)
                    else:
                        # 2. Fallback: Extract version from commit message
                        version_match = re.search(r'\bv\d+\.\d+(?:\.\d+)?(?:-patch\d+)?\b', c_subj + ' ' + c_body)
                        if version_match:
                            ev = version_match.group(0)
                            if ev != 'v1.0.0':
                                explicit_version = ev
                    
                    if 'fix' in subj_lower or 'bug' in subj_lower or 'audit' in subj_lower or 'harden' in subj_lower:
                        c_type = 'fix'
                        badge_label = 'Fix'
                        badge_color = 'bg-amber-500/10 text-amber-500 border-amber-500/20'
                        dot_color = 'bg-amber-500'
                        hover_border = 'hover:border-amber-500/40'
                    elif any(x in subj_lower for x in ('perf', 'opt', 'speed', 'clean', 'refactor', 'style', 'ui', 'layout', 'facelift', 'css')):
                        c_type = 'perf'
                        badge_label = 'Performance'
                        if any(x in subj_lower for x in ('style', 'ui', 'layout', 'css', 'facelift')):
                            badge_label = 'UI & Performance'
                        badge_color = 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20'
                        dot_color = 'bg-cyan-500'
                        hover_border = 'hover:border-cyan-500/40'
                    elif any(x in subj_lower for x in ('mutation', '🧬', 'promoted', 'variant', 'evolution')):
                        c_type = 'feat'
                        badge_label = '🧬 Mutation'
                        badge_color = 'bg-purple-500/10 text-purple-500 border-purple-500/20'
                        dot_color = 'bg-purple-500'
                        hover_border = 'hover:border-purple-500/40'
                    else:
                        c_type = 'feat'
                        badge_label = 'Feature'
                        badge_color = 'bg-brand-500/10 text-brand-500 border-brand-500/20'
                        dot_color = 'bg-brand-500'
                        hover_border = 'hover:border-brand-500/40'
                        
                    clean_subj, extra_bullets = _clean_subject_and_build_bullets(c_subj)
                    
                    intro_lines = []
                    body_bullets = []
                    for line in c_body.split('\n'):
                        stripped = line.strip()
                        if not stripped:
                            continue
                        if stripped.startswith('-') or stripped.startswith('*'):
                            body_bullets.append(stripped.lstrip('-').lstrip('*').strip())
                        else:
                            if not body_bullets:
                                intro_lines.append(stripped)
                            else:
                                body_bullets.append(stripped)
                                
                    bullet_lines = [b for b in extra_bullets if b] + body_bullets
                    intro_text = ' '.join(intro_lines)
                    
                    # Look up impacted files from the recent files dict
                    impacted_files = recent_files.get(c_hash, [])
                    
                    commits_raw.append({
                        'hash': c_hash,
                        'short_hash': c_hash[:7],
                        'date': c_date,
                        'subject': clean_subj,
                        'raw_subject': c_subj,
                        'intro_text': intro_text,
                        'bullet_lines': bullet_lines,
                        'body': c_body,
                        'type': c_type,
                        'badge_label': badge_label,
                        'badge_color': badge_color,
                        'dot_color': dot_color,
                        'hover_border': hover_border,
                        'explicit_version': explicit_version,
                        'impacted_files': impacted_files
                    })
            
            # Group commits reverse-chronologically by release boundary
            version_groups = []
            current_group = None
            
            for c in commits_raw:
                if c['explicit_version']:
                    if current_group is None:
                        current_group = {
                            'version': c['explicit_version'],
                            'date': c['date'],
                            'subject': c['subject'],
                            'commits': [c],
                            'type': c['type'],
                            'badge_color': c['badge_color'],
                            'dot_color': c['dot_color'],
                            'hover_border': c['hover_border'],
                        }
                    elif current_group['version'] == c['explicit_version']:
                        current_group['commits'].append(c)
                    else:
                        version_groups.append(current_group)
                        current_group = {
                            'version': c['explicit_version'],
                            'date': c['date'],
                            'subject': c['subject'],
                            'commits': [c],
                            'type': c['type'],
                            'badge_color': c['badge_color'],
                            'dot_color': c['dot_color'],
                            'hover_border': c['hover_border'],
                        }
                else:
                    if current_group is None:
                        current_group = {
                            'version': 'v' + current_version_str,
                            'date': c['date'],
                            'subject': c['subject'],
                            'commits': [c],
                            'type': c['type'],
                            'badge_color': c['badge_color'],
                            'dot_color': c['dot_color'],
                            'hover_border': c['hover_border'],
                        }
                    else:
                        current_group['commits'].append(c)
            
            if current_group:
                version_groups.append(current_group)
                
            # Merge details across each group
            for g in version_groups:
                merged_bullets = []
                seen_bullets = set()
                merged_files = {}
                intro_texts = []
                
                for c in g['commits']:
                    if c['intro_text']:
                        intro_texts.append(c['intro_text'])
                    for b in c['bullet_lines']:
                        if b.lower() not in seen_bullets:
                            merged_bullets.append(b)
                            seen_bullets.add(b.lower())
                    for f in c['impacted_files']:
                        path = f['path']
                        if path not in merged_files:
                            merged_files[path] = f
                        else:
                            if f['status'] == 'NEW':
                                merged_files[path]['status'] = 'NEW'
                            elif f['status'] == 'DELETE':
                                merged_files[path]['status'] = 'DELETE'
                                
                g['intro_text'] = ' '.join(intro_texts) if intro_texts else ''
                g['bullet_lines'] = merged_bullets
                g['impacted_files'] = list(merged_files.values())
                g['short_hash'] = g['commits'][0]['short_hash'] if g['commits'] else 'dev'
                
                # Determine major version number
                major = 1
                if g['version']:
                    v_match = re.search(r'v(\d+)\.', g['version'])
                    if v_match:
                        major = int(v_match.group(1))
                g['major_version'] = major
                
    except Exception as e:
        logger.error(f"Failed to fetch git log: {e}")
        
    return version_groups


@login_required
def changelog_view(request):
    from pathlib import Path
    from django.conf import settings
    
    context = _build_dashboard_context()
    release_groups = _get_git_changelog()
    
    # Read current version and last updated date dynamically from release groups, fallback to VERSION file
    if release_groups:
        context['current_version'] = release_groups[0]['version']
        context['last_updated_date'] = release_groups[0]['date']
    else:
        version_file = Path(settings.BASE_DIR) / 'VERSION'
        if version_file.exists():
            context['current_version'] = 'v' + version_file.read_text(encoding='utf-8').strip()
        else:
            context['current_version'] = 'v2.0.0'
        context['last_updated_date'] = 'June 10, 2026'
    
    # Identify the highest major version present
    max_major = 1
    for r in release_groups:
        if r['major_version'] > max_major:
            max_major = r['major_version']
            
    active_releases = []
    legacy_releases = []
    
    for r in release_groups:
        # If version has a major version <= 1 and there are versions > 1, collapse them into legacy
        if max_major > 1 and r['major_version'] <= 1:
            legacy_releases.append(r)
        else:
            active_releases.append(r)
            
    context['active_releases'] = active_releases
    context['legacy_releases'] = legacy_releases
    context['has_legacy'] = len(legacy_releases) > 0
    
    return render(request, 'changelog.html', context)


@login_required
def training_view(request):
    from django.core.paginator import Paginator

    # Exclude meta-generated artifact jobs (these show in Models Hub, not here)
    all_jobs_query = TrainingJob.objects.exclude(name__startswith='Meta Best').order_by('-id')
    meta_jobs_query = MetaTrainingJob.objects.all().order_by('-id')
    
    job_page_num = request.GET.get('job_page', 1)
    meta_page_num = request.GET.get('meta_page', 1)

    all_jobs = Paginator(all_jobs_query, 5).get_page(job_page_num)
    meta_jobs = Paginator(meta_jobs_query, 5).get_page(meta_page_num)

    context = {
        'training_jobs': all_jobs,
        'meta_training_jobs': meta_jobs,
        'playbook': STRATEGY_PLAYBOOK,
    }
    return render(request, 'training.html', context)

@login_required
def start_training_job_view(request):
    if request.method == 'POST':
        feature_set_key = request.POST.get('feature_set_key')
        hyperparameter_key = request.POST.get('hyperparameter_key')
        window_size_input = request.POST.get('window_size')
        # Convert window_size to integer
        try:
            if window_size_input in ['short_term', 'long_term', 'balanced']:
                # Map strategy types to default window sizes
                window_map = {
                    'short_term': 10,
                    'long_term': 20,
                    'balanced': 15
                }
                window_size = window_map[window_size_input]
            else:
                window_size = int(window_size_input)
        except (ValueError, KeyError):
            window_size = 10  # Default fallback

        # Create the job object
        job = TrainingJob.objects.create(
            name=request.POST.get('name') or 'Untitled Model',
            feature_set_key=feature_set_key,
            hyperparameter_key=hyperparameter_key,
            window_size=window_size,  # Now guaranteed to be an integer
            initial_cash=request.POST.get('initial_cash', 100000),
            ticker=request.POST.get('ticker', 'SPY'),
            training_duration_days=int(request.POST.get('training_duration_days', 365))
        )

        import sys
        from pathlib import Path
        
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"train_job_{job.id}.log"
        process = _spawn_background_process(
            [sys.executable, "run_training.py", "--job_id", str(job.id)],
            log_file,
        )
        job.celery_task_id = str(process.pid)
        job.save()
    return redirect('training')


@login_required
def stop_job_view(request, job_id):
    if request.method == 'POST':
        job = get_object_or_404(TrainingJob, id=job_id)
        if job.celery_task_id:
            try:
                _terminate_process(job.celery_task_id)
            except Exception as e:
                logger.warning(f"Could not kill PID {job.celery_task_id}: {e}")
            job.status = 'STOPPED'
            job.save()
    return redirect('training')


@login_required
def start_meta_job_view(request):
    if request.method == 'POST':
        meta_job = MetaTrainingJob.objects.create(
            initial_cash=request.POST.get('initial_cash', 100000),
            target_equity=request.POST.get('target_equity', 200000),
            ticker=request.POST.get('ticker', 'SPY'),
            training_duration_days=int(request.POST.get('training_duration_days', 365)),
            status='RUNNING'
        )

        import sys
        from pathlib import Path
        
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"meta_job_{meta_job.id}.log"
        process = _spawn_background_process(
            [sys.executable, "run_meta_trainer.py", "--job_id", str(meta_job.id)],
            log_file,
        )
        meta_job.error_message = ''
        meta_job.celery_task_id = str(process.pid)
        meta_job.save(update_fields=['error_message', 'celery_task_id'])
        
    return redirect('training')


def _get_available_models():
    return get_model_choices(include_disk=True, include_database=False)


# --- NEW: Views to start and stop the trader ---
@login_required
def papertrading_view(request):
    """
    Displays the main paper trading page and its current state.
    """
    memory_snapshot = _enforce_trader_memory_budget()
    traders = list(PaperTrader.objects.filter(is_live=False).order_by('id'))
    model_files = get_model_choices(include_disk=True, include_database=True)

    try:
        from src.execution.broker import Broker
        from .models import BrokerAccount
        # Use the first DB account if available, otherwise fall back to .env
        first_account = BrokerAccount.objects.first()
        broker = Broker(account=first_account)
        clock_data = broker.get_market_clock()
        live_equity = broker.get_equity()
        buying_power = broker.get_buying_power()
    except Exception:
        live_equity = 100000.00
        buying_power = 100000.00
        clock_data = {"is_open": False, "is_crypto_open": True}
        pass

    # Advanced Tracing Metrics for Live Nodes
    running_traders = [t for t in traders if t.status == 'RUNNING']
    active_starting_limit = 0.0
    active_amount_spent = 0.0
    active_amount_recovered = 0.0
    
    for t in running_traders:
        active_starting_limit += float(getattr(t, 'initial_cash', 0.0))
        for trade in t.trades.all():
            q = float(getattr(trade, 'quantity', 0))
            p = float(getattr(trade, 'price', 0))
            notional = float(getattr(trade, 'notional_value', q * p))
            if trade.action == 'BUY':
                active_amount_spent += notional
            elif trade.action == 'SELL':
                active_amount_recovered += notional

    # Actual PnL: recovered minus spent, plus unrealized value of held shares
    active_profit_made = active_amount_recovered - active_amount_spent

    from .models import BrokerAccount
    context = {
        'broker_accounts': BrokerAccount.objects.all(),
        'traders': traders,
        'trader_rows': [_get_trader_stats(trader) for trader in traders],
        'running_count': len(running_traders),
        'model_files': model_files,
        'clock': clock_data,
        'memory_snapshot': memory_snapshot,
        'live_equity': f"{float(live_equity):,.2f}",
        'active_starting_limit': f"{active_starting_limit:,.2f}",
        'active_amount_spent': f"{active_amount_spent:,.2f}",
        'active_amount_recovered': f"{active_amount_recovered:,.2f}",
        'active_profit_made': f"{abs(active_profit_made):,.2f}",
        'active_profit_raw': active_profit_made,
        'has_crypto_traders': True,
    }
    return render(request, 'papertrading_fleet.html', context)


@login_required
def model_detail_view(request, trader_id):
    """
    Renders an isolated tracking interface for an individual PaperTrader node.
    """
    import json
    from django.shortcuts import get_object_or_404
    from django.core.paginator import Paginator
    trader = get_object_or_404(PaperTrader.objects.prefetch_related('trades'), id=trader_id)
    
    trades_qs = trader.trades.all().order_by('-timestamp')
    initial_cash = float(getattr(trader, 'initial_cash', 0.0))
    
    # 1. Paginator for UI Table view
    paginator = Paginator(trades_qs, 10)
    page_number = request.GET.get('page', 1)
    trades_page = paginator.get_page(page_number)
    
    capital_spent = 0.0
    capital_recovered = 0.0
    
    # We must traverse the full unpaginated QS for global math totals
    for trade in trades_qs:
        notional = float(getattr(trade, 'notional_value', float(trade.quantity or 0) * float(trade.price or 0)))
        if trade.action == 'BUY':
            capital_spent += notional
        elif trade.action == 'SELL':
            capital_recovered += notional
            
    realized_profit = capital_recovered - capital_spent if capital_recovered > 0 else 0
    
    # 2. Sequential Chronological Walk for Graph Serialization (Oldest -> Newest)
    chart_data = []
    run_spend = 0.0
    run_rec = 0.0
    chronological_trades = trader.trades.all().order_by('timestamp')
    for ct in chronological_trades:
        nv = float(getattr(ct, 'notional_value', float(ct.quantity or 0) * float(ct.price or 0)))
        if ct.action == 'BUY': run_spend += nv
        if ct.action == 'SELL': run_rec += nv
        net = run_rec - run_spend if run_rec > 0 else -run_spend
        chart_data.append({
            'x': ct.timestamp.isoformat(),
            'y': round(net, 2)
        })
    
    context = {
        'trader': trader,
        'trades': trades_page,
        'initial_cash': f"{initial_cash:,.2f}",
        'capital_spent': f"{capital_spent:,.2f}",
        'capital_recovered': f"{capital_recovered:,.2f}",
        'realized_profit_raw': realized_profit,
        'realized_profit': f"{abs(realized_profit):,.2f}",
        'status_color': 'green-400' if trader.status == 'RUNNING' else ('yellow-500' if trader.status == 'PAUSED' else 'red-500'),
        'chart_data_json': json.dumps(chart_data)
    }
    return render(request, 'model_detail.html', context)

@login_required
def get_trader_log_api(request, trader_id):
    """
    Returns the tailing logic of the background Python terminal log for an isolated bot.
    """
    from django.http import JsonResponse
    from pathlib import Path
    
    log_dir = Path(__file__).parent.parent / "logs"
    log_file_path = log_dir / f"live_trader_{trader_id}.log"
    
    if not log_file_path.exists():
        return JsonResponse({'log': 'Awaiting engine boot Sequence. Log file intercept offline...'})
        
    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            tail = "".join(lines[-50:])
        return JsonResponse({'log': tail})
    except Exception as e:
        return JsonResponse({'log': f'System Exception accessing stream: {str(e)}'})

@login_required
def start_trader_view(request):
    """
    Handles the POST request to start the paper trader.
    """
    if request.method == 'POST':
        model_file = request.POST.get('model_file')
        initial_cash = request.POST.get('initial_cash', 100.00)
        
        raw_goal = request.POST.get('goal_amount')
        goal_amount = float(raw_goal) if raw_goal else None

        account_id = request.POST.get('account_id')
        from .models import BrokerAccount
        account = BrokerAccount.objects.filter(id=account_id).first() if account_id else None

        stop_loss_amount = None # Auto-calculated in async_engine
        
        if model_file:
            if str(model_file).startswith("db:"):
                # Simply allow the model since this is purely a Paper Trading / Sandbox environment.
                pass

            trader = PaperTrader.objects.create(
                model_file=model_file,
                initial_cash=initial_cash,
                goal_amount=goal_amount,
                stop_loss_amount=stop_loss_amount,
                status='STOPPED',
                error_message='',
                account=account,
            )
            _launch_trader_instance(trader)

    return redirect(request.META.get('HTTP_REFERER', 'papertrading'))


@login_required
def stop_trader_view(request, trader_id=None):
    if request.method == 'POST':
        trader = get_object_or_404(PaperTrader, id=trader_id) if trader_id else PaperTrader.objects.filter(status='RUNNING').order_by('-id').first()
        if trader:
            _stop_trader_instance(trader)

    return redirect(request.META.get('HTTP_REFERER', 'papertrading'))


@login_required
def pause_trader_view(request, trader_id):
    if request.method == 'POST':
        trader = get_object_or_404(PaperTrader, id=trader_id)
        if trader.status == 'RUNNING':
            _pause_trader_instance(trader, 'Paused by operator.')
    return redirect(request.META.get('HTTP_REFERER', 'papertrading'))


@login_required
def resume_trader_view(request, trader_id):
    if request.method == 'POST':
        trader = get_object_or_404(PaperTrader, id=trader_id)
        if trader.status == 'PAUSED':
            _resume_trader_instance(trader)
        elif trader.status in ['STOPPED', 'FAILED', 'SLEEPING'] and trader.model_file:
            _launch_trader_instance(trader)
    return redirect(request.META.get('HTTP_REFERER', 'papertrading'))


@login_required
def restart_trader_view(request, trader_id):
    if request.method == 'POST':
        trader = get_object_or_404(PaperTrader, id=trader_id)
        if trader.status in ['RUNNING', 'PAUSED']:
            _stop_trader_instance(trader)
            import time
            time.sleep(0.5)
        _launch_trader_instance(trader)
    return redirect(request.META.get('HTTP_REFERER', 'papertrading'))


@login_required
def start_all_traders_view(request):
    if request.method == 'POST':
        for trader in PaperTrader.objects.filter(status__in=['STOPPED', 'FAILED', 'SLEEPING']):
            if trader.model_file:
                _launch_trader_instance(trader)
    return redirect(request.META.get('HTTP_REFERER', 'papertrading'))


@login_required
def stop_all_traders_view(request):
    if request.method == 'POST':
        for trader in PaperTrader.objects.filter(status__in=['RUNNING', 'PAUSED']):
            _stop_trader_instance(trader)
    return redirect(request.META.get('HTTP_REFERER', 'papertrading'))


@login_required
def realtrading_view(request):
    """
    Displays the Live Production Trading dashboard, tracking REAL MONEY bots.
    """
    memory_snapshot = _enforce_trader_memory_budget()
    
    # Filter only LIVE traders
    traders = list(PaperTrader.objects.filter(is_live=True).order_by('id'))
    model_files = get_model_choices(include_disk=True, include_database=True)

    try:
        from src.execution.broker import Broker
        from .models import BrokerAccount
        # Select the Live account
        live_account = BrokerAccount.objects.filter(is_live=True).first()
        broker = Broker(account=live_account) if live_account else None
        
        if broker:
            clock_data = broker.get_market_clock()
            live_equity = broker.get_equity()
            buying_power = broker.get_buying_power()
        else:
            live_equity = 0.0
            buying_power = 0.0
            clock_data = {}
    except Exception:
        live_equity = 0.0
        buying_power = 0.0
        clock_data = {}
        pass

    # Advanced Tracing Metrics for Live Nodes
    running_traders = [t for t in traders if t.status == 'RUNNING']
    active_starting_limit = 0.0
    active_amount_spent = 0.0
    active_amount_recovered = 0.0
    
    for t in running_traders:
        active_starting_limit += float(getattr(t, 'initial_cash', 0.0))
        for trade in t.trades.all():
            q = float(getattr(trade, 'quantity', 0))
            p = float(getattr(trade, 'price', 0))
            notional = float(getattr(trade, 'notional_value', q * p))
            if trade.action == 'BUY':
                active_amount_spent += notional
            elif trade.action == 'SELL':
                active_amount_recovered += notional

    assumed_base = 0.0 # You wouldn't arbitrarily assume a base for real money
    active_profit_made = float(live_equity) - assumed_base if live_equity > 0 else 0.0

    from .models import BrokerAccount
    context = {
        'broker_accounts': BrokerAccount.objects.filter(is_live=True),
        'traders': traders,
        'trader_rows': [_get_trader_stats(trader) for trader in traders],
        'running_count': len(running_traders),
        'model_files': model_files,
        'clock': clock_data,
        'memory_snapshot': memory_snapshot,
        'live_equity': f"{float(live_equity):,.2f}",
        'active_starting_limit': f"{active_starting_limit:,.2f}",
        'active_amount_spent': f"{active_amount_spent:,.2f}",
        'active_amount_recovered': f"{active_amount_recovered:,.2f}",
        'active_profit_made': f"{abs(active_profit_made):,.2f}",
        'active_profit_raw': active_profit_made,
        'has_crypto_traders': True,
    }
    return render(request, 'realtrading.html', context)


@login_required
def start_real_trader_view(request):
    """
    Handles the POST request to start a real, live-fire bot.
    """
    if request.method == 'POST':
        model_file = request.POST.get('model_file')
        initial_cash = request.POST.get('initial_cash', 100.00)
        
        raw_goal = request.POST.get('goal_amount')
        goal_amount = float(raw_goal) if raw_goal else None

        account_id = request.POST.get('account_id')
        from .models import BrokerAccount
        account = BrokerAccount.objects.filter(id=account_id, is_live=True).first() if account_id else None

        if not account:
            messages.error(request, "CRITICAL FAULT: Valid live broker account not provided.")
            return redirect('realtrading')

        if model_file:
            # ENFORCE CERTIFICATION
            if str(model_file).startswith("db:"):
                # "db:12"
                model_id = int(str(model_file).split(":")[1])
                from .models import TrainingJob
                job = TrainingJob.objects.filter(id=model_id).first()
                if not job or not job.is_live_trading_ready:
                    messages.error(request, "CRITICAL ERROR: This model is NOT certified for Live Production Trading.")
                    return redirect('realtrading')
            else:
                # Disk models are explicitly uncertified since they skip the Evaluation Lab
                messages.error(request, "CRITICAL ERROR: Custom disk weights cannot be safely evaluated. Please use DB models.")
                return redirect('realtrading')

            trader = PaperTrader.objects.create(
                model_file=model_file,
                initial_cash=initial_cash,
                goal_amount=goal_amount,
                account=account,
                is_live=True, # Critical distinction
                status='STOPPED'
            )
            _launch_trader_instance(trader)
            # Add system alert for audit tracking
            SystemAlert.objects.create(
                level='CRITICAL',
                title='Live Bot Initiated',
                message=f'Production trader {trader.id} initialized with ${initial_cash} on {account.name}.'
            )
            
    return redirect('realtrading')


@login_required
def settings_view(request):
    if request.method == 'POST':
        settings = SystemSettings.load()
        
        # Save sensitive keys to database
        if 'alpaca_api_key' in request.POST:
            settings.alpaca_api_key = request.POST.get('alpaca_api_key', '')
            settings.alpaca_secret_key = request.POST.get('alpaca_secret_key', '')
            settings.broker_endpoint = request.POST.get('broker_endpoint', '')
            
        if 'gemini_api_key' in request.POST or 'openai_api_key' in request.POST or 'anthropic_api_key' in request.POST:
            settings.gemini_api_key = request.POST.get('gemini_api_key', '')
            settings.openai_api_key = request.POST.get('openai_api_key', '')
            settings.anthropic_api_key = request.POST.get('anthropic_api_key', '')

        # Remote gaming rig launch settings
        if 'gaming_rig_ip' in request.POST:
            settings.gaming_rig_ip = request.POST.get('gaming_rig_ip', '').strip()
            settings.gaming_rig_ssh_username = request.POST.get('gaming_rig_ssh_username', '').strip()
            settings.gaming_rig_ssh_password = request.POST.get('gaming_rig_ssh_password', '').strip()
            
        if 'steam_username' in request.POST:
            settings.steam_username = request.POST.get('steam_username', '').strip()
            
        if 'steam_api_key' in request.POST:
            settings.steam_api_key = request.POST.get('steam_api_key', '').strip()

        # Save non-sensitive settings to the database
        # Only update display_currency if the field was in the form submission
        if 'display_currency' in request.POST and request.POST.get('display_currency'):
            settings.display_currency = request.POST.get('display_currency')
            
        settings.save()

        messages.success(request, 'Settings have been saved successfully!')
        return redirect('settings')

    # Load current settings to pre-fill the form
    settings = SystemSettings.load()
    from .models import BrokerAccount
    context = {
        'settings': settings,
        'broker_accounts': BrokerAccount.objects.all(),
        'api_key': settings.alpaca_api_key,
        'secret_key': settings.alpaca_secret_key,
        'base_url': settings.broker_endpoint,
        'gemini_api_key': settings.gemini_api_key,
        'openai_api_key': settings.openai_api_key,
        'anthropic_api_key': settings.anthropic_api_key,
        'current_currency': settings.display_currency,
    }
    return render(request, 'settings.html', context)

@login_required
def test_email_api(request):
    try:
        from src.reporting.email_dispatcher import send_sos_alert
        success = send_sos_alert("DIAGNOSTIC-NODE-1", "System diagnostic nominal. This is a secure test of Phase 14 Dispatcher framework. SMTP relay authorized.")
        if success:
            return JsonResponse({"status": "SUCCESS", "message": "Diagnostic Matrix Dispatched."})
        else:
            return JsonResponse({"status": "FAILED", "message": "SMTP Relay Refused."})
    except Exception as e:
        return JsonResponse({"status": "FAILED", "message": str(e)})

@login_required
def test_rewriter_api(request):
    """Legacy alias — redirects to evolve stream."""
    return evolve_stream(request)


@login_required
def test_ai_key_api(request):
    """Test whichever AI key is configured and return the provider + status."""
    try:
        from src.core.code_rewriter import get_ai_client
        client = get_ai_client()
        # Quick ping to verify the key actually works
        test_response = client.generate("Say 'OK' in one word.", temperature=0.0)
        return JsonResponse({
            "status": "ok",
            "provider": client.provider,
            "message": f"{client.provider.title()} API key is valid and working.",
            "test_response": (test_response or "")[:100],
        })
    except ValueError as e:
        return JsonResponse({"status": "error", "provider": None, "message": str(e)})
    except Exception as e:
        return JsonResponse({"status": "error", "provider": None, "message": f"Key test failed: {str(e)}"})


@login_required
def evolve_stream(request):
    """SSE stream that runs neural evolution and sends real-time progress."""
    import queue, threading
    
    progress_queue = queue.Queue()
    
    def _run_evolution(client):
        """Run the code rewriter and push progress messages to the queue."""
        try:
            progress_queue.put("[EVOLVE] Initializing AI Engine...")
            from src.core.code_rewriter import load_ppo_agent_code, rewrite_agent_code
            from src.core.code_rewriter import _snapshot_parent_cash, generate_mutation_pdf
            
            progress_queue.put(f"[EVOLVE] Connected to {client.provider.title()}")
            
            progress_queue.put("[EVOLVE] Loading current model code...")
            current_code = load_ppo_agent_code()
            progress_queue.put(f"[EVOLVE] Loaded PPO agent ({len(current_code)} bytes)")
            
            # Build performance reports from recent trades
            progress_queue.put("[EVOLVE] Building performance snapshot...")
            d_report, w_report = _snapshot_parent_cash(None)
            
            progress_queue.put("[EVOLVE] Sending to AI for mutation analysis...")
            progress_queue.put("[EVOLVE] This may take 30-60 seconds...")
            
            new_code, raw_response = rewrite_agent_code(
                client, d_report, w_report, current_code
            )
            
            if not new_code or len(new_code.strip()) < 100:
                progress_queue.put("[ERROR] AI returned empty or invalid code. Aborting.")
                progress_queue.put("[DONE]")
                return
            
            progress_queue.put(f"[EVOLVE] Received mutated code ({len(new_code)} bytes)")
            
            # Save the new code
            from pathlib import Path
            agent_path = Path(__file__).resolve().parent.parent / 'src' / 'models' / 'ppo_agent.py'
            
            # Backup first
            backup_path = agent_path.with_suffix('.py.bak')
            if agent_path.exists():
                import shutil
                shutil.copy2(agent_path, backup_path)
                progress_queue.put("[EVOLVE] Backed up current model")
            
            agent_path.write_text(new_code, encoding='utf-8')
            progress_queue.put("[EVOLVE] New model code written to disk")
            
            # Generate diff
            import difflib
            old_lines = current_code.splitlines(keepends=True)
            new_lines = new_code.splitlines(keepends=True)
            diff = list(difflib.unified_diff(old_lines, new_lines, fromfile='ppo_agent.py.old', tofile='ppo_agent.py'))
            changes = len([l for l in diff if l.startswith('+') and not l.startswith('+++')])
            progress_queue.put(f"[EVOLVE] {changes} lines changed")
            
            progress_queue.put("[EVOLVE] Evolution complete!")
            progress_queue.put("[DONE]")
            
        except Exception as e:
            progress_queue.put(f"[ERROR] {str(e)}")
            progress_queue.put("[DONE]")
    
    def event_stream():
        try:
            from src.core.code_rewriter import get_ai_client
            client = get_ai_client()
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
            yield "event: complete\ndata: \n\n"
            return
            
        # Start evolution in background thread
        t = threading.Thread(target=_run_evolution, args=(client,), daemon=True)
        t.start()
        
        import time as _t
        timeout = 120  # 2 minute max
        start = _t.time()
        
        while _t.time() - start < timeout:
            try:
                msg = progress_queue.get(timeout=1)
                yield f"data: {msg}\n\n"
                if msg == "[DONE]":
                    yield "event: complete\ndata: \n\n"
                    return
            except queue.Empty:
                # Send heartbeat to keep connection alive
                yield "data: \n\n"
        
        yield "data: [ERROR] Evolution timed out after 2 minutes\n\n"
        yield "event: complete\ndata: \n\n"
    
    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response

@login_required
def evaluation_view(request):
    if request.method == 'POST':
        job = EvaluationJob.objects.create(
            model_file=request.POST.get('model_file'),
            start_date=request.POST.get('start_date'),
            end_date=request.POST.get('end_date'),
            status='PENDING'
        )
        
        import sys
        from pathlib import Path
        
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"eval_job_{job.id}.log"
        
        process = _spawn_background_process(
            [sys.executable, "run_evaluation_job.py", "--job_id", str(job.id)],
            log_file,
        )
        
        job.celery_task_id = str(process.pid)
        job.save(update_fields=['celery_task_id'])
        
        messages.success(request, f"Evaluation sequence #E{job.id} initiated in background.")
        return redirect('evaluation_lab')

    return redirect('evaluation_lab')


@login_required
def evaluation_lab_view(request):
    if request.method == 'POST':
        return evaluation_view(request)

    evaluation_jobs_qs = EvaluationJob.objects.all().order_by('-id')
    completed_jobs = [job for job in evaluation_jobs_qs if job.status == 'COMPLETED' and job.results]
    latest_job = completed_jobs[0] if completed_jobs else None

    evaluation_jobs = Paginator(evaluation_jobs_qs, 6).get_page(request.GET.get('page', 1))

    sharpe_series = [
        {
            'x': job.start_date.strftime('%Y-%m-%d'),
            'y': float(job.results.get('sharpe_ratio', 0) or 0),
        }
        for job in reversed(completed_jobs[-10:])
    ]

    drawdown_series = []
    if latest_job:
        equity_curve = latest_job.results.get('equity_chart', {}).get('equity', [])
        peak = None
        for index, value in enumerate(equity_curve):
            peak = value if peak is None else max(peak, value)
            drawdown = 0 if not peak else abs(((value - peak) / peak) * 100)
            drawdown_series.append({'x': index, 'y': round(drawdown, 2)})

    context = {
        'model_files': get_model_choices(include_disk=True, include_database=True),
        'evaluation_jobs': evaluation_jobs,
        'latest_job': latest_job,
        'sharpe_series': json.dumps(sharpe_series),
        'drawdown_series': json.dumps(drawdown_series),
    }
    return render(request, 'backtest_lab.html', context)


@login_required
def evaluation_report_view(request, job_id):
    job = get_object_or_404(EvaluationJob, id=job_id)

    # Prepare chart data for safe rendering in JavaScript
    equity_chart_data = job.results.get('equity_chart', {})

    context = {
        'job': job,
        'equity_dates': json.dumps(equity_chart_data.get('dates', [])),
        'equity_data': json.dumps(equity_chart_data.get('equity', [])),
    }
    return render(request, 'evaluation_report.html', context)


# control_panel/views.py (update job_status_api)
@login_required
def job_status_api(request):
    training_jobs = TrainingJob.objects.exclude(name__startswith='Meta Best')
    meta_jobs = MetaTrainingJob.objects.all()
    evaluation_jobs = EvaluationJob.objects.all()

    for job in training_jobs:
        if job.status == 'RUNNING' and job.celery_task_id and not _process_is_running(job.celery_task_id):
            job.status = 'FAILED'
            if not job.error_message:
                job.error_message = "Training process stopped before completion."
            job.save(update_fields=['status', 'error_message'])

    for job in meta_jobs:
        if job.status == 'RUNNING' and job.celery_task_id and not _process_is_running(job.celery_task_id):
            job.status = 'FAILED'
            if not job.error_message:
                job.error_message = "Meta-training process stopped before completion."
            job.save(update_fields=['status', 'error_message'])

    for job in evaluation_jobs:
        if job.status == 'RUNNING' and job.celery_task_id and not _process_is_running(job.celery_task_id):
            job.status = 'FAILED'
            if not job.error_message:
                job.error_message = "Evaluation process stopped before completion."
            job.save(update_fields=['status', 'error_message'])

    data = {
        'training_jobs': [
            {
                'id': job.id,
                'status': job.status,
                'progress': job.progress,
                'best_reward': round(job.best_reward, 2),
                'error_message': job.error_message or ''
            } for job in training_jobs
        ],
        'meta_training_jobs': [
            {
                'id': job.id,
                'status': job.status,
                'progress': job.progress,
                'best_sharpe_ratio': job.results.get('sharpe_ratio', 0.0) if job.results else 0.0,
                'error_message': job.error_message or ''
            } for job in meta_jobs
        ],
        'evaluation_jobs': [
            {
                'id': job.id,
                'status': job.status,
                'progress': 100 if job.status == 'COMPLETED' else 0,
                'sharpe_ratio': job.results.get('sharpe_ratio', 0.0) if job.results else 0.0,
                'total_return_pct': job.results.get('total_return_pct', 0.0) if job.results else 0.0,
                'error_message': job.error_message or ''
            } for job in evaluation_jobs
        ]
    }
    return JsonResponse(data)

@login_required
def job_logs_api(request, job_type, job_id):
    from control_panel.models import EvaluationJob
    log_dir = Path(__file__).parent.parent / "logs"
    if job_type == 'meta':
        job_model = MetaTrainingJob
    elif job_type == 'eval':
        job_model = EvaluationJob
    else:
        job_model = TrainingJob
        
    job = job_model.objects.filter(id=job_id).first()

    if job_type == 'meta':
        log_file = log_dir / f"meta_job_{job_id}.log"
    elif job_type == 'eval':
        log_file = log_dir / f"eval_job_{job_id}.log"
    else:
        log_file = log_dir / f"train_job_{job_id}.log"

    if job and job.status == 'RUNNING' and job.celery_task_id and not _process_is_running(job.celery_task_id):
        job.status = 'FAILED'
        if not job.error_message:
            job.error_message = "Training process stopped before completion."
        job.save(update_fields=['status', 'error_message'])

    if not log_file.exists():
        if job and job.status == 'FAILED' and job.error_message:
            return JsonResponse({"logs": f"Process failed before log stream initialized.\n{job.error_message}"})
        return JsonResponse({"logs": "System provisioning container...\nWaiting for log stream..."})
        
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
            # Grabbing the last 200 lines to avoid blowing up memory on UI
            logs = "".join(lines[-200:])  

        if job and job.status == 'RUNNING' and "Training Orchestration finished." in logs:
            job.status = 'COMPLETED'
            job.progress = 100
            job.save(update_fields=['status', 'progress'])
        elif job and job.status == 'RUNNING' and "Traceback" in logs:
            job.status = 'FAILED'
            if not job.error_message:
                job.error_message = "Training crashed. See log stream for traceback."
            job.save(update_fields=['status', 'error_message'])

        return JsonResponse({"logs": logs})
    except Exception as e:
        return JsonResponse({"logs": f"Error accessing internal Matrix stream: {e}"})


@login_required
def trader_logs_api(request, trader_id):
    log_dir = Path(__file__).parent.parent / "logs"
    trader = PaperTrader.objects.filter(id=trader_id).first()
    log_file = log_dir / f"live_trader_{trader_id}.log"

    if trader and trader.status == 'RUNNING' and trader.celery_task_id and not _process_is_running(trader.celery_task_id):
        trader.status = 'FAILED'
        if not trader.error_message:
            trader.error_message = "Trader process stopped before completion."
        trader.save(update_fields=['status', 'error_message'])

    if not log_file.exists():
        if trader and trader.status in ['FAILED', 'PAUSED'] and trader.error_message:
            return JsonResponse({"logs": f"Operator state captured.\n{trader.error_message}"})
        return JsonResponse({"logs": "Provisioning runner terminal...\nWaiting for live engine stream..."})

    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            logs = "".join(lines[-200:])
        return JsonResponse({"logs": logs})
    except Exception as e:
        return JsonResponse({"logs": f"Error accessing trader terminal: {e}"})


@login_required
def trader_trades_api(request, trader_id):
    from django.shortcuts import get_object_or_404
    trader = get_object_or_404(PaperTrader, id=trader_id)
    trades = TradeLog.objects.filter(trader=trader).order_by('-timestamp')
    data = []
    for t in trades:
        data.append({
            'timestamp': t.timestamp.isoformat(),
            'time_short': t.timestamp.strftime("%H:%M:%S"),
            'symbol': t.symbol,
            'action': t.action,
            'price': float(t.price),
            'notional_value': float(t.notional_value or 0.0),
            'quantity': float(t.quantity or 0.0),
        })
    return JsonResponse({"trades": data})


def trader_status_api(request):
    memory_snapshot = _enforce_trader_memory_budget()
    
    # Filter by is_live if requested (Real Trading page sends ?is_live=true)
    is_live_param = request.GET.get('is_live')
    if is_live_param == 'true':
        traders = list(PaperTrader.objects.filter(is_live=True).order_by('id'))
    elif is_live_param == 'false':
        traders = list(PaperTrader.objects.filter(is_live=False).order_by('id'))
    else:
        traders = list(PaperTrader.objects.all().order_by('id'))
    
    running_trader = next((trader for trader in traders if trader.status == 'RUNNING'), None)

    def _to_float(v):
        from decimal import Decimal, InvalidOperation
        try:
            if isinstance(v, Decimal):
                return float(v)
            return float(str(v))
        except (ValueError, TypeError, InvalidOperation):
            return 0.0

    try:
        active_accounts = list({t.account for t in traders if t.account})
        positions = []
        equity = 0.0
        buying_power = 0.0
        clock_data = {}

        if not active_accounts:
            try:
                from .models import BrokerAccount
                # Respect is_live filter for fallback too
                if is_live_param == 'true':
                    fallback_account = BrokerAccount.objects.filter(is_live=True).first()
                elif is_live_param == 'false':
                    fallback_account = BrokerAccount.objects.filter(is_live=False).first()
                else:
                    fallback_account = BrokerAccount.objects.first()
                
                if fallback_account:
                    broker = Broker(account=fallback_account)
                    clock_data = broker.get_market_clock()
                    equity = _to_float(broker.get_equity())
                    buying_power = _to_float(broker.get_buying_power())
                    for p in broker.get_positions():
                        positions.append({
                            "symbol": getattr(p, 'symbol', ''),
                            "qty": _to_float(getattr(p, 'qty', 0)),
                            "market_value": _to_float(getattr(p, 'market_value', 0)),
                            "unrealized_pl": _to_float(getattr(p, 'unrealized_pl', 0)),
                        })
            except Exception:
                pass
        else:
            for acc in active_accounts:
                try:
                    broker = Broker(account=acc)
                    if not clock_data:
                        clock_data = broker.get_market_clock()
                    equity += _to_float(broker.get_equity())
                    buying_power += _to_float(broker.get_buying_power())
                    for p in broker.get_positions():
                        # Group up positions with same symbol if needed, but alpaca isolated accounts won't overlap usually
                        positions.append({
                            "symbol": getattr(p, 'symbol', ''),
                            "qty": _to_float(getattr(p, 'qty', 0)),
                            "market_value": _to_float(getattr(p, 'market_value', 0)),
                            "unrealized_pl": _to_float(getattr(p, 'unrealized_pl', 0)),
                        })
                except Exception:
                    pass

        positions.sort(key=lambda x: x['market_value'], reverse=True)
        
        # Calculate Active Fleet Profit for auto-update
        running_traders = [t for t in traders if t.status == 'RUNNING']
        active_starting_limit = 0.0
        active_amount_spent = 0.0
        active_amount_recovered = 0.0
        for t in running_traders:
            active_starting_limit += float(getattr(t, 'initial_cash', 0.0))
            for trade in t.trades.all():
                q = float(getattr(trade, 'quantity', 0))
                p = float(getattr(trade, 'price', 0))
                notional = float(getattr(trade, 'notional_value', q * p))
                if trade.action == 'BUY':
                    active_amount_spent += notional
                elif trade.action == 'SELL':
                    active_amount_recovered += notional
        
        # Match python backend UI logic
        # Actual PnL: recovered minus spent
        active_profit_made = active_amount_recovered - active_amount_spent
        
        return JsonResponse({
            "status": running_trader.status if running_trader else "STOPPED",
            "model_file": running_trader.model_file if running_trader else "",
            "error_message": running_trader.error_message if running_trader else "",
            "equity": equity,
            "buying_power": buying_power,
            "positions": positions,
            "memory": memory_snapshot,
            "clock": clock_data,
            "traders": [_get_trader_stats(trader) for trader in traders],
            "active_starting_limit": f"{active_starting_limit:,.2f}",
            "active_amount_spent": f"{active_amount_spent:,.2f}",
            "active_amount_recovered": f"{active_amount_recovered:,.2f}",
            "active_profit_made": f"{abs(active_profit_made):,.2f}",
            "active_profit_raw": active_profit_made,
            "live_equity": f"{float(equity):,.2f}",
        })
    except Exception as e:
        msg = str(e)
        if running_trader:
            if "authorization failed" in msg.lower() or "alpaca authorization failed" in msg.lower():
                running_trader.status = 'FAILED'
                running_trader.error_message = "Alpaca auth failed. Check keys & endpoint."
            else:
                running_trader.status = 'FAILED'
                running_trader.error_message = msg
            running_trader.save(update_fields=['status', 'error_message'])
        return JsonResponse({
            "status": "FAILED",
            "error_message": running_trader.error_message if running_trader else msg,
            "equity": 0.0,
            "buying_power": 0.0,
            "positions": [],
            "memory": memory_snapshot,
            "clock": None,
            "traders": [_get_trader_stats(trader) for trader in traders],
        }, status=200)


@login_required
def stop_meta_job_view(request, job_id):
    if request.method == 'POST':
        job = get_object_or_404(MetaTrainingJob, id=job_id)
        if job.celery_task_id:
            try:
                _terminate_process(job.celery_task_id)
            except Exception as e:
                logger.warning(f"Could not kill PID {job.celery_task_id}: {e}")
            job.status = 'STOPPED'
            job.save()
    return redirect('training')


def trader_activity_api(request):
    """
    Provides the detailed, real-time activity of a running task.
    """
    trader, _ = PaperTrader.objects.get_or_create(id=1)
    if trader.status == 'RUNNING' and trader.celery_task_id:
        task_result = AsyncResult(trader.celery_task_id)
        if task_result.state == 'PROGRESS':
            return JsonResponse({"status": "RUNNING", **task_result.info})
        else:
            return JsonResponse({"status": "RUNNING", "activity": f"Task state: {task_result.state}"})
    else:
        return JsonResponse({"status": "STOPPED", "activity": "Trader is not active."})


@login_required
def reports_hub_view(request):
    """
    Renders the Intelligence Vault listing all persisted EOD reports with timeframe filters and sorting.
    """
    from .models import TradingReport
    from django.utils import timezone
    from django.core.paginator import Paginator
    from django.db.models import Avg, Sum
    import datetime
    import calendar
    
    selected_month = request.GET.get('month', '')
    selected_week = request.GET.get('week_of_month', 'all')
    selected_type = request.GET.get('report_type', 'all')
    selected_mode = request.GET.get('run_mode', 'all')
    sort_by = request.GET.get('sort_by', '-timestamp')
    page_number = request.GET.get('page', 1)
    
    # Validate sort parameter
    allowed_sorts = [
        'timestamp', '-timestamp', 
        'total_revenue', '-total_revenue', 
        'win_rate', '-win_rate', 
        'total_trades', '-total_trades'
    ]
    if sort_by not in allowed_sorts:
        sort_by = '-timestamp'
        
    reports = TradingReport.objects.all()
    
    if selected_type and selected_type != 'all':
        reports = reports.filter(report_type=selected_type)
        
    if selected_mode and selected_mode != 'all':
        reports = reports.filter(run_mode=selected_mode)
        
    if selected_month:
        try:
            year_str, month_str = selected_month.split('-')
            year = int(year_str)
            month = int(month_str)
            
            _, last_day = calendar.monthrange(year, month)
            tz = timezone.get_current_timezone()
            
            if selected_week == 'all':
                start_date = datetime.datetime(year, month, 1, tzinfo=tz)
                end_date = start_date + datetime.timedelta(days=last_day)
                reports = reports.filter(timestamp__gte=start_date, timestamp__lt=end_date)
            else:
                w_seg = int(selected_week)
                start_day = 1
                end_day = 7
                if w_seg == 2:
                    start_day = 8
                    end_day = 14
                elif w_seg == 3:
                    start_day = 15
                    end_day = 21
                elif w_seg == 4:
                    start_day = 22
                    end_day = last_day
                    
                start_date = datetime.datetime(year, month, start_day, tzinfo=tz)
                end_date = start_date + datetime.timedelta(days=(end_day - start_day + 1))
                reports = reports.filter(timestamp__gte=start_date, timestamp__lt=end_date)
        except Exception:
            pass
            
    reports = reports.order_by(sort_by)
    
    # Calculate aggregate stats before pagination
    total_count = reports.count()
    total_revenue = float(reports.aggregate(total=Sum('total_revenue'))['total'] or 0.0)
    avg_win_rate = float(reports.aggregate(avg=Avg('win_rate'))['avg'] or 0.0)
    total_trades_sum = reports.aggregate(total=Sum('total_trades'))['total'] or 0
    
    # Extract available months dynamically from report history
    available_months = TradingReport.objects.dates('timestamp', 'month', order='DESC')
    
    # Paginate (10 items per page)
    paginator = Paginator(reports, 10)
    page_obj = paginator.get_page(page_number)
    
    context = {
        'reports': page_obj,
        'selected_month': selected_month,
        'selected_week': selected_week,
        'selected_type': selected_type,
        'selected_mode': selected_mode,
        'selected_sort': sort_by,
        'available_months': available_months,
        'total_count': total_count,
        'total_revenue': total_revenue,
        'avg_win_rate': avg_win_rate,
        'total_trades_sum': total_trades_sum,
        'active_page': 'reports_hub'
    }
    return render(request, 'reports_hub.html', context)


@login_required
def download_report_pdf_view(request, report_id):
    """
    Downloads the physical PDF byte construct of a previously stored intelligence report.
    """
    import os
    from django.http import HttpResponse, Http404
    from .models import TradingReport
    
    report = get_object_or_404(TradingReport, id=report_id)
    if not report.pdf_path or not os.path.exists(report.pdf_path):
        raise Http404("PDF artifact not found on disk.")
        
    with open(report.pdf_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type='application/pdf')
        filename = os.path.basename(report.pdf_path)
        response['Content-Disposition'] = f'inline; filename="{filename}"'
        return response


@login_required
def view_report_view(request, report_id):
    """
    Redirects to the reports hub and automatically triggers the inline modal viewer for the given report.
    """
    from django.shortcuts import redirect
    return redirect(f'/intelligence-vault/?open_report={report_id}')


@login_required
def download_report_markdown_view(request, report_id):
    """
    Downloads the raw markdown of a previously stored intelligence report.
    """
    import os
    from django.http import HttpResponse, Http404
    from .models import TradingReport
    
    report = get_object_or_404(TradingReport, id=report_id)
    if not report.markdown_path or not os.path.exists(report.markdown_path):
        raise Http404("Markdown file not found on disk.")
        
    with open(report.markdown_path, 'r', encoding='utf-8') as f:
        response = HttpResponse(f.read(), content_type='text/markdown')
        filename = os.path.basename(report.markdown_path)
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response


@login_required
def report_content_api(request, report_id):
    """
    JSON endpoint returning the compiled HTML content of a report for AJAX loading.
    """
    import os
    import markdown
    from django.http import JsonResponse
    from .models import TradingReport
    
    report = TradingReport.objects.filter(id=report_id).first()
    if not report:
        return JsonResponse({"status": "error", "message": "Report not found."}, status=404)
        
    if not report.markdown_path or not os.path.exists(report.markdown_path):
        return JsonResponse({"status": "error", "message": "Markdown file not found on disk."}, status=404)
        
    try:
        with open(report.markdown_path, 'r', encoding='utf-8') as f:
            md_text = f.read()
            
        import re
        md_text = re.sub(r'([^\n])\s*(##+ )', r'\1\n\n\2', md_text)
        html_content = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'nl2br'])
        
        return JsonResponse({
            "status": "success",
            "id": report.id,
            "report_type": report.report_type,
            "run_mode": report.run_mode,
            "timestamp": report.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "total_trades": report.total_trades,
            "total_revenue": float(report.total_revenue),
            "win_rate": float(report.win_rate),
            "html": html_content
        })
    except Exception as e:
        return JsonResponse({"status": "error", "message": f"Error compiling markdown: {str(e)}"}, status=500)

@login_required
def delete_trader_api(request, trader_id):
    trader = get_object_or_404(PaperTrader, id=trader_id)
    if request.method == 'POST':
        # Safely shut down engine if it's running
        try:
            if trader.status == 'RUNNING' and trader.pid:
                import psutil
                try:
                    parent = psutil.Process(trader.pid)
                    for child in parent.children(recursive=True):
                        child.kill()
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass
        except Exception as e:
            messages.error(request, f"Error stopping instance for deletion: {e}")
        trader.delete()
        messages.success(request, f"Instance {trader_id} completely wiped.")
    return redirect(request.META.get('HTTP_REFERER', 'papertrading'))

@login_required
def models_hub_view(request):
    """
    Renders the unified Models Vault Phase 16, pulling from both database jobs and OS disk.
    """
    from .model_registry import get_model_choices
    raw_models = get_model_choices(include_disk=True, include_database=True)
    
    enriched_models = []
    
    from django.core.cache import cache
    recommendations = cache.get('model_recommendations') or {}
    
    for val in raw_models:
        ref = val['value']
        source = val['source']
        label = val.get('label', ref)
        
        size_mb = '--'
        created_at_fmt = '--'
        eval_count = 0
        is_ready = False
        
        if source == 'database':
            from .models import TrainingJob, EvaluationJob
            raw_id = ref.split(':')[1]
            job = TrainingJob.objects.filter(id=raw_id).first()
            if job:
                if job.model_weights:
                    size_mb = f"{len(job.model_weights) / (1024*1024):.2f}"
                created_at_fmt = "Stored in Postgres"
                eval_count = EvaluationJob.objects.filter(model_file=ref).count()
                is_ready = getattr(job, 'is_live_trading_ready', False)
        elif source == 'disk':
            from pathlib import Path
            import datetime
            p = Path("saved_models") / ref
            if p.exists():
                size_mb = f"{p.stat().st_size / (1024*1024):.2f}"
                dt = datetime.datetime.fromtimestamp(p.stat().st_ctime)
                created_at_fmt = dt.strftime('%Y-%m-%d %H:%M')
                from .models import EvaluationJob
                eval_count = EvaluationJob.objects.filter(model_file=ref).count()
        
        # Get AI recommendation data
        rec = recommendations.get(ref, {})
                
        enriched_models.append({
            'reference': ref,
            'label': label,
            'source': source.upper(),
            'size_mb': size_mb,
            'created_at': created_at_fmt,
            'evaluations': eval_count,
            'is_ready': is_ready,
            'rec_grade': rec.get('grade', ''),
            'rec_score': rec.get('score', ''),
            'rec_sharpe': rec.get('sharpe', ''),
            'rec_return': rec.get('return_pct', ''),
            'rec_drawdown': rec.get('max_drawdown', ''),
        })
    
    # Sort: recommended models first (by score descending), then rest
    enriched_models.sort(key=lambda m: float(m.get('rec_score', 0) or 0), reverse=True)
        
    context = {
        'models': enriched_models,
        'active_page': 'models_hub'
    }
    return render(request, 'models_hub.html', context)


@login_required
def delete_model_api(request, file_name=None):
    if request.method != 'POST':
        return redirect('models_hub')
        
    from pathlib import Path
    import os
    ref = file_name or request.POST.get('model_reference')
    if not ref:
        messages.error(request, "Discard operation blocked. Identity reference missing.")
        return redirect('models_hub')
        
    try:
        if str(ref).startswith("db:"):
            from .models import TrainingJob, EvaluationJob
            # Erase dependencies inside DB
            raw_id = ref.split(':')[1]
            job = TrainingJob.objects.filter(id=raw_id).first()
            if job:
                EvaluationJob.objects.filter(model_file=ref).delete()
                job.delete()
                messages.success(request, f"Purged [{ref}] from Neural Postgres.")
            else:
                messages.error(request, "Model not located in DB.")
        else:
            p = Path("saved_models") / ref
            if p.exists():
                os.remove(p)
                messages.success(request, f"Terminated physical chassis [{ref}] from disk.")
            else:
                messages.error(request, f"File {ref} not found on hard-drive schema.")
                
    except Exception as e:
        messages.error(request, f"Neural Sever Exception: {str(e)}")
        
    return redirect('models_hub')

@login_required
def toggle_model_ready_api(request, file_name=None):
    if request.method != 'POST':
        return redirect('models_hub')
        
    ref = file_name or request.POST.get('model_reference')
    if str(ref).startswith("db:"):
        from .models import TrainingJob
        raw_id = ref.split(':')[1]
        job = TrainingJob.objects.filter(id=raw_id).first()
        if job:
            job.is_live_trading_ready = not job.is_live_trading_ready
            job.save()
            state = "CERTIFIED" if job.is_live_trading_ready else "UNCERTIFIED"
            messages.success(request, f"[{ref}] is now {state} for live trading environments.")
        else:
            messages.error(request, "Database chassis not found.")
    else:
        messages.error(request, "Only Postgres-backed neural models check for readiness tokens.")
        
    return redirect('models_hub')

@login_required
def dismiss_alert_api(request, alert_id):
    if request.method == 'POST':
        from .models import SystemAlert
        alert = SystemAlert.objects.filter(id=alert_id).first()
        if alert:
            alert.is_read = True
            alert.save()
            if request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.GET.get('format') == 'json' or 'application/json' in request.headers.get('Accept', ''):
                return JsonResponse({"status": "success", "message": "Alert dismissed."})
            messages.success(request, "A/B Swap Recommendation dismissed.")
        else:
            if request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.GET.get('format') == 'json' or 'application/json' in request.headers.get('Accept', ''):
                return JsonResponse({"status": "error", "message": "Alert not found."}, status=404)
            messages.error(request, "Alert not found.")
    return redirect('dashboard')

@login_required
def toggle_setting_api(request, setting_name):
    if request.method == 'POST':
        from .models import SystemSettings
        settings = SystemSettings.load()
        if hasattr(settings, setting_name):
            current_val = getattr(settings, setting_name)
            setattr(settings, setting_name, not current_val)
            settings.save()
            messages.success(request, f"Setting '{setting_name}' updated successfully.")
        else:
            messages.error(request, "Invalid setting name.")
    return redirect(request.META.get('HTTP_REFERER', 'dashboard'))

@login_required
def add_broker_account_api(request):
    if request.method == 'POST':
        from .models import BrokerAccount
        BrokerAccount.objects.create(
            name=request.POST.get('name', 'Unnamed env'),
            api_key=request.POST.get('api_key', ''),
            secret_key=request.POST.get('secret_key', ''),
            base_url=request.POST.get('base_url', 'https://paper-api.alpaca.markets'),
            is_live=request.POST.get('is_live') == 'on'
        )
        messages.success(request, "Broker environment successfully attached.")
    return redirect('settings')

@login_required
def delete_broker_account_api(request, account_id):
    if request.method == 'POST':
        from .models import BrokerAccount
        acc = BrokerAccount.objects.filter(id=account_id).first()
        if acc:
            acc.delete()
            messages.info(request, "Broker environment securely detached.")
    return redirect('settings')

@login_required
def edit_trader_view(request, trader_id):
    if request.method == 'POST':
        trader = get_object_or_404(PaperTrader, id=trader_id)
        
        initial_cash = request.POST.get('initial_cash')
        goal_amount = request.POST.get('goal_amount')
        account_id = request.POST.get('account_id')
        model_file = request.POST.get('model_file')
        
        was_running = (trader.status == 'RUNNING')
        
        if initial_cash:
            trader.initial_cash = float(initial_cash)
        
        if goal_amount:
            trader.goal_amount = float(goal_amount)
            
        if account_id:
            from .models import BrokerAccount
            acc = BrokerAccount.objects.filter(id=account_id).first()
            if acc:
                trader.account = acc
        
        model_changed = False
        if model_file and trader.model_file != model_file:
            # If live, enforce certification
            if trader.is_live:
                if str(model_file).startswith("db:"):
                    model_id = int(str(model_file).split(":")[1])
                    from .models import TrainingJob
                    job = TrainingJob.objects.filter(id=model_id).first()
                    if not job or not job.is_live_trading_ready:
                        messages.error(request, "CRITICAL ERROR: Selected model is NOT certified for Live Production.")
                        return redirect(request.META.get('HTTP_REFERER', 'realtrading'))
                else:
                    messages.error(request, "CRITICAL ERROR: Custom disk weights cannot be safely evaluated. Please use DB models.")
                    return redirect(request.META.get('HTTP_REFERER', 'realtrading'))
            
            trader.model_file = model_file
            model_changed = True
                
        trader.save()
        
        # If it was running and model/config changed, restart the process to apply changes
        if was_running and model_changed:
            try:
                import time
                _stop_trader_instance(trader)
                # Give the OS a moment to release ports/files
                time.sleep(0.5)
                _launch_trader_instance(trader)
                messages.success(request, f"Node #{trader.id} configuration updated and restarted successfully with model '{model_file}'.")
            except Exception as e:
                messages.error(request, f"Error restarting node #{trader.id} after edit: {e}")
        else:
            messages.success(request, f"Node #{trader.id} configuration updated successfully.")
            
    return redirect(request.META.get('HTTP_REFERER', 'papertrading'))

@login_required
@require_POST
def export_trade_logs_api(request):
    """Placeholder for CSV export functionality"""
    pass

@login_required
@require_POST
def global_kill_switch_api(request):
    """
    EMERGENCY OVERRIDE:
    1. Stop all active bots
    2. Cancel all pending orders on all bound broker accounts
    3. Liquidate all open positions at market price
    """
    from .models import PaperTrader, BrokerAccount
    from src.execution.broker import Broker
    import traceback
    
    logger.critical("[KILL SWITCH] Global Kill Switch Initiated via Dashboard.")
    
    # 1. Stop all bots
    for bot in PaperTrader.objects.filter(status='RUNNING'):
        _stop_trader_instance(bot.id)
        
    # 2. Liquidate all broker accounts
    accounts = BrokerAccount.objects.all()
    if not accounts:
        # Fallback to default API credentials if no explicit DB accounts
        try:
            broker = Broker()
            broker.api.cancel_all_orders()
            broker.api.close_all_positions(cancel_orders=True)
            logger.critical("[KILL SWITCH] Default Broker account liquidated.")
        except Exception as e:
            logger.error(f"[KILL SWITCH] Failed on default account: {e}")
            
    for acc in accounts:
        try:
            broker = Broker(account=acc)
            broker.api.cancel_all_orders()
            broker.api.close_all_positions(cancel_orders=True)
            logger.critical(f"[KILL SWITCH] Account {acc.id} ({acc.name}) liquidated.")
        except Exception as e:
            logger.error(f"[KILL SWITCH] Failed on account {acc.id}: {e}\n{traceback.format_exc()}")
            
    messages.success(request, "Global Kill Switch Executed. All bots stopped and positions liquidated.")
    return redirect('dashboard')

@login_required
def mutation_logs_api(request):
    """
    Streams the mutation progress logs for the frontend terminal.
    """
    from pathlib import Path
    log_file = Path(__file__).parent.parent / "logs" / "mutation.log"
    if not log_file.exists():
        return JsonResponse({"logs": "Initializing Neural Mutation Sequence...\nSearching for fresh intelligence artifacts..."})
    
    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            logs = "".join(lines[-300:])
        return JsonResponse({"logs": logs})
    except Exception as e:
        return JsonResponse({"logs": f"Error accessing mutation terminal: {e}"})

@login_required
def system_logs_api(request):
    """
    Tails the main system log (trading_bot.log) for the frontend terminal.
    """
    from pathlib import Path
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_file_path = log_dir / "trading_bot.log"
    
    if not log_file_path.exists():
        return JsonResponse({"logs": "Main system log offline. Waiting for engine start..."})
        
    try:
        with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            logs = "".join(lines[-150:])
        return JsonResponse({"logs": logs})
    except Exception as e:
        return JsonResponse({"logs": f"Error accessing system logs: {e}"})

@login_required
def trigger_mutation_api(request):
    """
    Manually triggers the Gemini Cognitive Mutation loop as a background process
    so we can stream the logs to the terminal.
    """
    if request.method == 'POST':
        import sys
        from pathlib import Path
        
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "mutation.log"
        
        # Clear old log
        if log_file.exists():
            log_file.write_text("", encoding='utf-8')
            
        _spawn_background_process(
            [sys.executable, str(Path("src") / "core" / "code_rewriter.py"), "--force"],
            log_file
        )
        
        # Return JSON for AJAX (fetch) requests, redirect for standard form POSTs
        is_ajax = request.headers.get('X-CSRFToken') or request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        if is_ajax:
            return JsonResponse({"status": "ok", "message": "Mutation sequence initiated."})
        
        messages.success(request, "Jarvis Neural Mutation sequence initiated. Gemini is now analyzing reports...")
    return redirect('models_hub')


# ===========================================================================
#  NEURAL EVOLUTION ENGINE — API Endpoints
# ===========================================================================

@login_required
def evolution_hub_view(request):
    """
    Renders the dedicated Neural Evolution Chamber page.
    """
    return render(request, 'evolution_hub.html')

@login_required
def variant_details_view(request, variant_id):
    """Renders the dedicated detail page for a single Neural Evolution variant."""
    from .models import ModelVariant, VirtualTrade
    import json

    variant = get_object_or_404(ModelVariant, id=variant_id)
    trades = VirtualTrade.objects.filter(variant=variant).order_by('timestamp')

    # Build equity curve data points
    equity_curve = []
    for t in trades:
        equity_curve.append({
            'x': t.timestamp.isoformat(),
            'y': float(t.virtual_balance_after),
        })

    # If no trades, seed with starting cash
    if not equity_curve:
        equity_curve = [{'x': variant.created_at.isoformat(), 'y': float(variant.starting_cash)}]

    total_buys = trades.filter(action='BUY').count()
    total_sells = trades.filter(action='SELL').count()

    context = {
        'variant': variant,
        'trades': trades[:500],  # Cap to prevent page bloat
        'equity_curve_json': json.dumps(equity_curve),
        'trade_count': trades.count(),
        'total_buys': total_buys,
        'total_sells': total_sells,
    }
    return render(request, 'variant_details.html', context)


@login_required
def variant_metrics_api(request, variant_id):
    from .models import ModelVariant, VirtualTrade
    variant = get_object_or_404(ModelVariant, id=variant_id)
    trades = VirtualTrade.objects.filter(variant=variant).order_by('timestamp')
    
    trades_data = []
    positions = {}  # symbol -> {'qty': float, 'entry_price': float}
    
    for t in trades:
        symbol = t.symbol
        qty = float(t.quantity)
        price = float(t.price)
        pnl = 0.0
        pnl_pct = 0.0
        
        if t.action == 'BUY':
            if symbol not in positions:
                positions[symbol] = {'qty': 0.0, 'entry_price': 0.0}
            pos = positions[symbol]
            new_qty = pos['qty'] + qty
            if new_qty > 0:
                pos['entry_price'] = ((pos['entry_price'] * pos['qty']) + (price * qty)) / new_qty
            pos['qty'] = new_qty
        elif t.action == 'SELL':
            if symbol in positions:
                pos = positions[symbol]
                sell_qty = min(qty, pos['qty'])
                if sell_qty > 0:
                    pnl = (price - pos['entry_price']) * sell_qty
                    pnl_pct = ((price - pos['entry_price']) / pos['entry_price'] * 100) if pos['entry_price'] > 0 else 0.0
                    pos['qty'] -= sell_qty
                if pos['qty'] <= 1e-9:
                    positions.pop(symbol, None)
                    
        trades_data.append({
            'timestamp': t.timestamp.isoformat(),
            'symbol': t.symbol,
            'action': t.action,
            'quantity': qty,
            'price': price,
            'notional_value': float(t.notional_value),
            'virtual_balance_after': float(t.virtual_balance_after),
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 4),
        })
        
    equity_curve = []
    for t in trades:
        equity_curve.append({
            'x': t.timestamp.isoformat(),
            'y': float(t.virtual_balance_after),
        })
        
    if not equity_curve:
        equity_curve = [{'x': variant.created_at.isoformat(), 'y': float(variant.starting_cash)}]
        
    total_buys = trades.filter(action='BUY').count()
    total_sells = trades.filter(action='SELL').count()
    
    return JsonResponse({
        'status': 'success',
        'metrics': {
            'status': variant.status,
            'starting_cash': float(variant.starting_cash),
            'virtual_balance': float(variant.virtual_balance),
            'virtual_pnl': float(variant.virtual_pnl),
            'virtual_pnl_pct': float(variant.virtual_pnl_pct),
            'sharpe_ratio': float(variant.sharpe_ratio),
            'max_drawdown_pct': float(variant.max_drawdown_pct),
            'win_rate': float(variant.win_rate),
            'virtual_trades_count': variant.virtual_trades_count,
            'total_buys': total_buys,
            'total_sells': total_sells,
        },
        'trades': list(reversed(trades_data)),
        'equity_curve': equity_curve,
    })


def evolution_variants_api(request):
    """GET: Returns all ModelVariant records with live metrics for the dashboard."""
    from .models import ModelVariant
    
    variants = ModelVariant.objects.all()[:20]  # Last 20 variants
    data = []
    for v in variants:
        data.append({
            'id': v.id,
            'name': v.name,
            'status': v.status,
            'starting_cash': float(v.starting_cash),
            'virtual_balance': float(v.virtual_balance),
            'virtual_pnl': float(v.virtual_pnl),
            'virtual_pnl_pct': v.virtual_pnl_pct,
            'sharpe_ratio': v.sharpe_ratio,
            'max_drawdown_pct': v.max_drawdown_pct,
            'win_rate': v.win_rate,
            'virtual_trades_count': v.virtual_trades_count,
            'days_remaining': v.days_remaining,
            'test_duration_days': v.test_duration_days,
            'created_at': v.created_at.isoformat() if v.created_at else None,
            'parent_trader_id': v.parent_trader_id,
            'error_message': v.error_message,
        })
    
    return JsonResponse({'status': 'success', 'variants': data})

@login_required
def variant_logs_api(request, variant_id):
    """
    Streams the virtual trading progress logs for a specific variant.
    """
    from pathlib import Path
    log_file = Path(__file__).parent.parent / "logs" / f"evolution_variant_{variant_id}.log"
    if not log_file.exists():
        return JsonResponse({"logs": f"Virtual Engine offline. Log file evolution_variant_{variant_id}.log not found."})
    
    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            logs = "".join(lines[-300:])
        return JsonResponse({"logs": logs})
    except Exception as e:
        return JsonResponse({"logs": f"Error accessing virtual terminal: {e}"})


@login_required
@require_POST
def evolution_promote_api(request, variant_id):
    """
    POST: Promote a variant to production.
    1. Writes variant.agent_code to ppo_agent.py (with backup)
    2. Restarts all running PaperTrader instances
    3. Marks variant as PROMOTED, fails all other TESTING variants
    """
    from .models import ModelVariant, PaperTrader, SystemAlert
    from pathlib import Path
    import difflib
    
    variant = ModelVariant.objects.filter(id=variant_id).first()
    if not variant:
        return JsonResponse({'status': 'error', 'message': 'Variant not found'}, status=404)
    
    if variant.status not in ('PENDING', 'TESTING'):
        return JsonResponse({'status': 'error', 'message': f'Cannot promote variant in {variant.status} status'}, status=400)
    
    target_path = Path(django_settings.BASE_DIR) / 'src' / 'models' / 'ppo_agent.py'
    backup_path = Path(django_settings.BASE_DIR) / 'src' / 'models' / 'ppo_agent.py.bak'
    
    try:
        # Backup current production code
        if target_path.exists():
            current_code = target_path.read_text(encoding='utf-8')
            backup_path.write_text(current_code, encoding='utf-8')
        
        # Write the variant's evolved code to production
        target_path.write_text(variant.agent_code, encoding='utf-8')
        
        # Mark this variant as promoted
        variant.status = 'PROMOTED'
        variant.save(update_fields=['status'])
        
        # Terminate, delete, and clean up logs of all other testing variants
        other_variants = list(ModelVariant.objects.filter(status='TESTING').exclude(id=variant_id))
        for ov in other_variants:
            if ov.celery_task_id:
                try:
                    import psutil
                    import signal
                    pid = int(ov.celery_task_id)
                    if psutil.pid_exists(pid):
                        os.kill(pid, signal.SIGTERM)
                except Exception as e:
                    logger.error(f"[EVOLUTION] Failed to kill PID {ov.celery_task_id} for variant #{ov.id}: {e}")
            
            # Delete log file
            log_file = Path(django_settings.BASE_DIR) / "logs" / f"evolution_variant_{ov.id}.log"
            if log_file.exists():
                try:
                    log_file.unlink()
                except Exception as e:
                    logger.error(f"[EVOLUTION] Failed to delete log file for variant #{ov.id}: {e}")
            
            # Delete variant
            ov.delete()
        
        # Restart all running traders to pick up new code
        running_traders = list(PaperTrader.objects.filter(status__in=['RUNNING', 'SLEEPING']))
        restarted = 0
        for trader in running_traders:
            try:
                _stop_trader_instance(trader.id)
                import time as _t
                _t.sleep(0.5)
                _launch_trader_instance(trader)
                restarted += 1
            except Exception as e:
                logger.error(f"[EVOLUTION] Failed to restart trader #{trader.id}: {e}")
        
        # 1. Programmatically update changelog.html
        try:
            changelog_path = Path(django_settings.BASE_DIR) / 'templates' / 'changelog.html'
            if changelog_path.exists():
                changelog_content = changelog_path.read_text(encoding='utf-8')
                
                # Find the timeline container start
                target_str = '<div class="changelog-layout-grid">\n        <!-- Left Column: Timeline Entries -->\n        <div class="relative ml-4 md:ml-6">'
                idx = changelog_content.find(target_str)
                # Fallback if whitespace differs
                if idx == -1:
                    target_str = '<div class="relative ml-4 md:ml-6">'
                    idx = changelog_content.find(target_str)
                
                if idx != -1:
                    insert_pos = idx + len(target_str)
                    
                    # We also want to replace the first top-2 line segment with top-0
                    # to connect the line segment. Let's find it after insert_pos.
                    segment_str = 'absolute left-0 top-2 bottom-0 w-[2px]'
                    seg_idx = changelog_content.find(segment_str, insert_pos)
                    if seg_idx != -1:
                        changelog_content = (
                            changelog_content[:seg_idx] +
                            'absolute left-0 top-0 bottom-0 w-[2px]' +
                            changelog_content[seg_idx + len(segment_str):]
                        )
                    
                    # Now construct the new entry HTML
                    from django.utils import timezone
                    formatted_date = timezone.now().strftime("%B %d, %Y")
                    mutation_reasoning = (variant.mutation_reasoning or "Evolved strategy mutation promoted to production.").replace('\n', '<br>')
                    diff_summary = variant.diff_summary or "No diff details provided."
                    
                    new_entry = f"""
        <!-- Mutation Strategy Promotion: Variant v{variant.id} -->
        <div id="variant-{variant.id}" class="changelog-item relative pl-6 md:pl-8 pb-8 group" data-type="feat">
            <!-- Vertical line segment -->
            <div class="absolute left-0 top-2 bottom-0 w-[2px] bg-slate-200 dark:bg-slate-800 pointer-events-none"></div>
            <!-- Timeline dot -->
            <div class="absolute -left-[5px] top-2 w-3 h-3 rounded-full bg-purple-50 border-2 border-white dark:border-slate-950 group-hover:scale-125 transition-transform"></div>
            
            <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm rounded-2xl overflow-hidden hover:border-purple-500/40 transition-all duration-300">
                <!-- Header -->
                <div class="px-6 py-4 border-b border-slate-100 dark:border-slate-800 flex items-center justify-between flex-wrap gap-2 bg-slate-50/50 dark:bg-slate-900/50">
                    <div class="flex items-center gap-3">
                        <span class="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-purple-50/10 text-purple-400 border border-purple-500/20">🧬 MUTATION</span>
                        <h3 class="text-sm font-bold text-slate-800 dark:text-slate-200">Evolved Strategy Promoted: {variant.name}</h3>
                    </div>
                    <div class="flex items-center gap-2 font-mono text-[10px] text-slate-500">
                        <i class="far fa-calendar-alt"></i> {formatted_date}
                        <span class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">Variant #{variant.id}</span>
                    </div>
                </div>
                
                <!-- Body -->
                <div class="p-6 space-y-4">
                    <p class="text-xs text-slate-600 dark:text-slate-400 leading-relaxed font-mono">
                        {mutation_reasoning}
                    </p>
                    
                    <div class="border-t border-slate-100 dark:border-slate-800 pt-4 mt-4">
                        <div class="font-mono text-[10px] text-slate-500 dark:text-slate-400 space-y-3 leading-relaxed">
                            <div><strong class="text-slate-700 dark:text-slate-200">Mutation PnL:</strong> {variant.virtual_pnl_pct:.2f}% &middot; <strong class="text-slate-700 dark:text-slate-200">Win Rate:</strong> {variant.win_rate:.1f}% &middot; <strong class="text-slate-700 dark:text-slate-200">Sharpe:</strong> {variant.sharpe_ratio:.2f}</div>
                            <div><strong class="text-slate-700 dark:text-slate-200">Diff Summary:</strong></div>
                            <div class="pl-2 text-[10px] bg-slate-50 dark:bg-[#0a0a0c] p-3 rounded-lg border border-slate-250 dark:border-slate-800 font-mono whitespace-pre-wrap">{diff_summary}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
"""
                    changelog_content = changelog_content[:insert_pos] + new_entry + changelog_content[insert_pos:]
                    changelog_path.write_text(changelog_content, encoding='utf-8')
                    logger.info(f"[EVOLUTION] Appended strategy promotion for Variant #{variant.id} to changelog.html")
        except Exception as e:
            logger.error(f"[EVOLUTION] Failed to write changelog update: {e}", exc_info=True)

        # 2. Git Commit & Push Automation
        git_committed = False
        git_pushed = False
        try:
            import subprocess
            repo_dir = str(Path(django_settings.BASE_DIR))
            
            # git add src/models/ppo_agent.py templates/changelog.html
            subprocess.run(["git", "add", "src/models/ppo_agent.py", "templates/changelog.html"], cwd=repo_dir, check=True, capture_output=True, text=True)
            
            # git commit
            commit_msg = f"[MUTATION] [AI PUSHED] Promoted Evolved Variant #{variant.id}: {variant.name}"
            subprocess.run(["git", "commit", "-m", commit_msg], cwd=repo_dir, check=True, capture_output=True, text=True)
            git_committed = True
            
            # git push
            push_res = subprocess.run(["git", "push"], cwd=repo_dir, capture_output=True, text=True)
            if push_res.returncode == 0:
                git_pushed = True
                logger.info(f"[EVOLUTION] Git commit & push successful: {commit_msg}")
            else:
                logger.error(f"[EVOLUTION] Git push failed: {push_res.stderr}")
        except Exception as git_err:
            logger.error(f"[EVOLUTION] Git commit/push operation failed gracefully: {git_err}")

        # Create success alert
        SystemAlert.objects.create(
            level='INFO',
            title=f'🧬 Variant #{variant.id} Promoted to Production',
            message=f"'{variant.name}' is now active. {restarted} trader(s) restarted. Changelog updated.{' Git committed & pushed.' if git_pushed else (' Git committed but push failed.' if git_committed else ' Git auto-commit failed.')}",
            related_model_reference=str(variant.id),
        )
        
        logger.info(f"[EVOLUTION] Variant #{variant_id} promoted. {restarted} traders restarted.")
        
        return JsonResponse({
            'status': 'success',
            'message': f"Variant #{variant_id} promoted to production. {restarted} trader(s) restarted with evolved model. Git status: Pushed={git_pushed}."
        })
    
    except Exception as e:
        logger.error(f"[EVOLUTION] Promotion failed: {e}", exc_info=True)
        # Rollback
        if backup_path.exists():
            target_path.write_text(backup_path.read_text(encoding='utf-8'), encoding='utf-8')
        return JsonResponse({'status': 'error', 'message': f'Promotion failed: {str(e)}'}, status=500)


@login_required
@require_POST
def evolution_reject_api(request, variant_id):
    """POST: Reject a variant — kill virtual engine, delete log file, record audit, delete DB record."""
    from .models import ModelVariant, SystemAlert
    from pathlib import Path
    import os
    import signal
    
    variant = ModelVariant.objects.filter(id=variant_id).first()
    if not variant:
        return JsonResponse({'status': 'success', 'message': f'Variant #{variant_id} was already rejected and deleted.'})
    
    reason = 'Manually rejected'
    report_type = 'minor'
    category = 'underperformance'
    include_trade_logs = False
    next_directive = 'none'
    
    if request.content_type == 'application/json':
        try:
            import json
            data = json.loads(request.body)
            reason = data.get('reason') or reason
            report_type = data.get('report_type') or report_type
            category = data.get('category') or category
            include_trade_logs = data.get('include_trade_logs', False)
            next_directive = data.get('next_directive') or next_directive
        except Exception:
            pass
    else:
        reason = request.POST.get('reason') or request.GET.get('reason') or reason

    # Map category/directive to friendly names
    cat_names = {
        'underperformance': 'Strategy Underperformance',
        'drawdown': 'Excessive Drawdown / High Risk',
        'timing': 'Poor Trade Timing (Early/Late)',
        'overfitting': 'Overfitting / Insufficient Trades',
        'errors': 'Runtime Errors & Bugs',
        'other': 'Other / Custom Reason'
    }
    dir_names = {
        'none': 'No specific directive',
        'tighten_sl': 'Tighten stop-loss / risk controls',
        'improve_timing': 'Improve entry/exit timing parameters',
        'optimize_winrate': 'Optimize for higher win rate',
        'reduce_frequency': 'Reduce overall trading frequency'
    }
    
    cat_lbl = cat_names.get(category, category)
    dir_lbl = dir_names.get(next_directive, next_directive)
    
    # Compile reasoning markdown message
    alert_msg = f"**Variant Name**: {variant.name}\n"
    alert_msg += f"**Rejection Type**: {report_type.upper()}\n"
    alert_msg += f"**Category**: {cat_lbl}\n"
    if report_type == 'detailed':
        alert_msg += f"**Suggested Directive**: {dir_lbl}\n"
    alert_msg += f"**User Comments**: {reason}\n\n"
    
    # Compile trade metrics if checked
    if report_type == 'detailed' and include_trade_logs:
        from .models import VirtualTrade
        trades = VirtualTrade.objects.filter(variant=variant).order_by('timestamp')
        total_trades = trades.count()
        buys = trades.filter(action='BUY').count()
        sells = trades.filter(action='SELL').count()
        
        win_rate = float(variant.win_rate)
        completed_trades = sells
        wins = int(round((win_rate / 100) * completed_trades)) if completed_trades else 0
        losses = completed_trades - wins
        
        alert_msg += "### 📊 Trade Performance Summary\n"
        alert_msg += f"- **Total Trades**: {total_trades} ({buys} BUYs / {sells} SELLs)\n"
        alert_msg += f"- **Completed Trades (Sells)**: {completed_trades}\n"
        alert_msg += f"- **Wins / Losses**: {wins} wins / {losses} losses\n"
        alert_msg += f"- **Win Rate**: {win_rate:.1f}%\n"
        alert_msg += f"- **Starting Balance**: ${float(variant.starting_cash):,.2f}\n"
        alert_msg += f"- **Final Balance**: ${float(variant.virtual_balance):,.2f}\n"
        alert_msg += f"- **Net P/L**: {variant.virtual_pnl_pct:+.2f}%\n"

    # Try to kill the virtual engine process
    if variant.celery_task_id:
        try:
            os.kill(int(variant.celery_task_id), signal.SIGTERM)
        except Exception:
            pass
            
    # Try to delete the log file
    log_file = Path(__file__).parent.parent / "logs" / f"evolution_variant_{variant_id}.log"
    if log_file.exists():
        try:
            log_file.unlink()
        except Exception:
            pass

    # Append code and rationale to alert message so they can be retrieved by code_rewriter when training/evolution runs
    if variant.mutation_reasoning:
        alert_msg += f"### 💡 Attempted Rationale\n{variant.mutation_reasoning}\n\n"
    if variant.agent_code:
        alert_msg += f"### 💻 Agent Code\n```python\n{variant.agent_code}\n```\n"

    # Create SystemAlert to track the audit log
    alert = SystemAlert.objects.create(
        level='WARNING',
        title=f'🧬 Variant #{variant.id} Rejected & Deleted',
        message=alert_msg,
        related_model_reference=str(variant.id)
    )
    
    variant.delete()
    
    return JsonResponse({'status': 'success', 'message': f'Variant #{variant_id} rejected and deleted.', 'alert_id': alert.id})

@login_required
def rejection_report_view(request, alert_id):
    """Display a formatted rejection report for the given SystemAlert."""
    from .models import SystemAlert
    import re
    
    alert = SystemAlert.objects.filter(id=alert_id).first()
    if not alert:
        from django.shortcuts import redirect
        return redirect('evolution_hub')
    
    # Parse the markdown message into structured sections
    message = alert.message
    
    # Extract key-value fields from the top section
    fields = {}
    for key in ('Variant Name', 'Rejection Type', 'Category', 'Suggested Directive', 'User Comments'):
        match = re.search(rf'\*\*{re.escape(key)}\*\*:\s*(.+?)(?:\n|$)', message)
        if match:
            fields[key] = match.group(1).strip()
    
    # Extract trade performance section
    trade_metrics = []
    trade_section_match = re.search(r'### 📊 Trade Performance Summary\n(.*?)(?=###|\Z)', message, re.DOTALL)
    if trade_section_match:
        for line in trade_section_match.group(1).strip().split('\n'):
            line = line.strip().lstrip('-').strip()
            if line:
                # Split on **: to get label and value
                m = re.match(r'\*\*(.+?)\*\*:\s*(.+)', line)
                if m:
                    trade_metrics.append({'label': m.group(1), 'value': m.group(2)})
    
    # Extract rationale section
    rationale = None
    rationale_match = re.search(r'### 💡 Attempted Rationale\n(.*?)(?=###|\Z)', message, re.DOTALL)
    if rationale_match:
        rationale = rationale_match.group(1).strip()
    
    # Extract agent code section
    agent_code = None
    code_match = re.search(r'### 💻 Agent Code\n```python\n(.*?)```', message, re.DOTALL)
    if code_match:
        agent_code = code_match.group(1).strip()
    
    context = _build_dashboard_context()
    context.update({
        'alert': alert,
        'variant_name': fields.get('Variant Name', 'Unknown Variant'),
        'rejection_type': fields.get('Rejection Type', 'MINOR'),
        'rejection_category': fields.get('Category', ''),
        'suggested_directive': fields.get('Suggested Directive', ''),
        'user_comments': fields.get('User Comments', ''),
        'trade_metrics': trade_metrics,
        'rationale': rationale,
        'agent_code': agent_code,
        'variant_id': alert.related_model_reference,
    })
    
    return render(request, 'rejection_report.html', context)


@login_required
@require_POST
def evolution_delete_api(request, variant_id):
    """POST: Delete a variant, stop its virtual engine process, clean up logs, and delete DB record."""
    from django.shortcuts import redirect
    from .models import ModelVariant
    import os
    import signal
    
    variant = ModelVariant.objects.filter(id=variant_id).first()
    if not variant:
        if request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.GET.get('format') == 'json':
            return JsonResponse({'status': 'success', 'message': f'Variant #{variant_id} was already deleted.'})
        return redirect('evolution_hub')
    
    # Try to kill the virtual engine process
    if variant.celery_task_id:
        try:
            os.kill(int(variant.celery_task_id), signal.SIGTERM)
        except Exception:
            pass
            
    # Try to delete the log file
    from pathlib import Path
    log_file = Path(__file__).parent.parent / "logs" / f"evolution_variant_{variant_id}.log"
    if log_file.exists():
        try:
            log_file.unlink()
        except Exception:
            pass
            
    variant.delete()
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.GET.get('format') == 'json':
        return JsonResponse({'status': 'success', 'message': f'Variant #{variant_id} deleted successfully.'})
        
    return redirect('evolution_hub')


@login_required
@require_POST
def evolution_restart_api(request, variant_id):
    """POST: Restarts a failed or stopped variant's virtual trading engine."""
    from django.shortcuts import redirect
    from .models import ModelVariant
    import sys
    import os
    import signal
    from pathlib import Path
    
    variant = ModelVariant.objects.filter(id=variant_id).first()
    if not variant:
        if request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.GET.get('format') == 'json':
            return JsonResponse({'status': 'error', 'message': f'Variant #{variant_id} was already deleted and cannot be restarted.'}, status=404)
        return redirect('evolution_hub')
    
    # Enforce spawn guard to make sure we don't exceed max 3 active variants
    from src.core.code_rewriter import _enforce_spawn_guard
    _enforce_spawn_guard(max_active=3)
    
    # Kill existing process if any
    if variant.celery_task_id:
        try:
            import psutil
            pid = int(variant.celery_task_id)
            if psutil.pid_exists(pid):
                os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
            
    # Reset metrics
    variant.status = 'TESTING'
    variant.error_message = None
    variant.virtual_balance = variant.starting_cash
    variant.virtual_trades_count = 0
    variant.virtual_pnl = 0
    variant.virtual_pnl_pct = 0.0
    variant.win_rate = 0.0
    variant.sharpe_ratio = 0.0
    variant.max_drawdown_pct = 0.0
    variant.save()
    
    # Delete virtual trades to start clean
    variant.virtual_trades.all().delete()
    
    # Spawn process
    try:
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"evolution_variant_{variant.id}.log"
        
        # Clear existing logs
        if log_file.exists():
            try:
                log_file.unlink()
            except Exception:
                pass
                
        process = _spawn_background_process(
            [sys.executable, "-m", "src.core.virtual_paper_engine", "--variant_id", str(variant.id)],
            log_file,
        )
        variant.celery_task_id = str(process.pid)
        variant.save(update_fields=['celery_task_id'])
    except Exception as spawn_err:
        # Create failure alert
        try:
            from control_panel.models import SystemAlert
            alert_msg = f"**Variant Name**: {variant.name}\n"
            alert_msg += f"**Failure Type**: SPAWN ERROR (Manual Restart)\n"
            alert_msg += f"**Error Exception**: {str(spawn_err)}\n\n"
            if variant.mutation_reasoning:
                alert_msg += f"### 💡 Attempted Rationale\n{variant.mutation_reasoning}\n\n"
            if variant.agent_code:
                alert_msg += f"### 💻 Agent Code\n```python\n{variant.agent_code}\n```\n"
            
            SystemAlert.objects.create(
                level='WARNING',
                title=f'🧬 Variant #{variant.id} Failed: Spawn Error',
                message=alert_msg,
                related_model_reference=str(variant.id)
            )
        except Exception:
            pass
            
        variant.delete()
        
        if request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.GET.get('format') == 'json':
            return JsonResponse({'status': 'error', 'message': f'Failed to spawn virtual engine: {spawn_err}'}, status=500)
            
        return redirect('evolution_hub')
        
    if request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.GET.get('format') == 'json':
        return JsonResponse({'status': 'success', 'message': f'Variant #{variant_id} restarted successfully.'})
        
    return redirect('variant_details', variant_id=variant.id)



@login_required
def evolution_evaluate_api(request):
    """POST: Manually trigger the evolution evaluator to check expired variants."""
    from src.core.evolution_evaluator import evaluate_expired_variants
    evaluate_expired_variants()
    return JsonResponse({'status': 'success', 'message': 'Evaluation complete.'})


import subprocess
import time
from django.http import StreamingHttpResponse

@login_required
def check_updates_api(request):
    try:
        subprocess.run(['git', 'fetch', 'origin'], capture_output=True, check=False)
        result = subprocess.run(['git', 'rev-list', 'HEAD..origin/master', '--count'], capture_output=True, text=True, check=False)
        commits_behind = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
        return JsonResponse({'status': 'success', 'commits_behind': commits_behind})
    except Exception as e:
        return JsonResponse({'status': 'error', 'error': str(e)}, status=500)

@login_required
def system_update_stream(request):
    MAX_RETRIES = 5
    RETRY_DELAY = 30  # seconds

    def _run_subprocess(cmd, label):
        """Run a subprocess and yield its output. Returns (success, lines_yielded)."""
        import time as _time
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        lines = []
        last_heartbeat = _time.time()
        while True:
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                if _time.time() - last_heartbeat > 5:
                    lines.append(": ping\n\n")
                    last_heartbeat = _time.time()
                _time.sleep(0.1)
                continue
            lines.append(f"data: {line}\n\n")
            last_heartbeat = _time.time()
        process.wait()
        return process.returncode == 0, lines

    def event_stream():
        import time as _time

        for attempt in range(1, MAX_RETRIES + 1):
            if attempt > 1:
                yield f"data: [RETRY] Attempt {attempt}/{MAX_RETRIES} — waiting {RETRY_DELAY}s before retry...\n\n"
                _time.sleep(RETRY_DELAY)

            yield f"data: [SYSTEM] Initiating Override Protocol: Git Pull (attempt {attempt}/{MAX_RETRIES})\n\n"

            try:
                # 1. GIT FETCH & RESET (Clean sync)
                yield "data: [SYSTEM] Fetching latest intelligence from remote tower...\n\n"
                success, lines = _run_subprocess(
                    ['git', 'fetch', 'origin'], 'Git Fetch'
                )
                for line in lines:
                    yield line

                success, lines = _run_subprocess(
                    ['git', 'reset', '--hard', 'origin/master'], 'Git Reset'
                )
                for line in lines:
                    yield line

                if not success:
                    yield f"data: [ERROR] Git sync failed on attempt {attempt}/{MAX_RETRIES}\n\n"
                    if attempt < MAX_RETRIES:
                        continue  # Retry
                    yield "event: error\ndata: \n\n"
                    import threading
                    threading.Thread(target=_delayed_server_restart, daemon=True).start()
                    return
                
                yield "data: [SYSTEM] Synchronizing Dependencies...\n\n"
                
                # 1b. PIP INSTALL
                success, lines = _run_subprocess(
                    [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 'Pip Install'
                )
                for line in lines:
                    yield line
                
                yield "data: [SYSTEM] Synchronizing Database Schema...\n\n"

                # 2. RUN MIGRATIONS
                success, lines = _run_subprocess(
                    [sys.executable, 'manage.py', 'migrate'], 'Migration'
                )
                for line in lines:
                    yield line

                if not success:
                    yield f"data: [ERROR] Migration failed on attempt {attempt}/{MAX_RETRIES}\n\n"
                    if attempt < MAX_RETRIES:
                        continue  # Retry
                    yield "event: error\ndata: \n\n"
                    import threading
                    threading.Thread(target=_delayed_server_restart, daemon=True).start()
                    return

                # 3. RESTART RUNNING TRADERS so they pick up the new engine code
                running_traders = list(PaperTrader.objects.filter(status__in=['RUNNING', 'PAUSED']))
                if running_traders:
                    yield f"data: [SYSTEM] Restarting {len(running_traders)} active trading node(s)...\n\n"
                    for trader in running_traders:
                        try:
                            _stop_trader_instance(trader)
                            _time.sleep(0.5)
                            _launch_trader_instance(trader)
                            yield f"data: [SYSTEM] Node #{trader.id} restarted with updated code.\n\n"
                        except Exception as restart_err:
                            yield f"data: [WARNING] Failed to restart Node #{trader.id}: {restart_err}\n\n"

                yield "data: [SYSTEM] Update Successful. Reloading platform...\n\n"
                yield "event: complete\ndata: \n\n"
                
                # 4. SCHEDULE SERVER SELF-RESTART (runs after SSE stream closes)
                import threading
                def _delayed_server_restart():
                    """Restart the server process after a short delay.
                    Handles: systemd, supervisor, Daphne (PaaS), and dev server."""
                    _time.sleep(3)  # Let the SSE response flush to client first
                    try:
                        import shutil, signal
                        
                        # Method 1: systemctl (Linux systemd services)
                        if shutil.which('systemctl'):
                            for svc in ['ai_trader', 'aitrader', 'gunicorn', 'daphne', 'django']:
                                result = subprocess.run(
                                    ['systemctl', 'is-active', svc],
                                    capture_output=True, text=True, timeout=5
                                )
                                if result.returncode == 0:
                                    subprocess.Popen(
                                        ['systemctl', 'restart', svc],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                                    )
                                    logger.info(f"[UPDATE] Server restart via systemctl restart {svc}")
                                    return
                        
                        # Method 2: supervisorctl
                        if shutil.which('supervisorctl'):
                            subprocess.Popen(
                                ['supervisorctl', 'restart', 'all'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                            )
                            logger.info("[UPDATE] Server restart via supervisorctl restart all")
                            return
                        
                        # Method 3: Replace current process using os.execv (Works cross-platform)
                        import sys
                        logger.info(f"[UPDATE] Restarting process via os.execv: {sys.argv}")
                        
                        if sys.argv[0].endswith('.py'):
                            os.execv(sys.executable, [sys.executable] + sys.argv)
                        else:
                            os.execv(sys.argv[0], sys.argv)
                        
                        
                    except Exception as restart_err:
                        logger.warning(f"[UPDATE] Server self-restart failed: {restart_err}")
                
                threading.Thread(target=_delayed_server_restart, daemon=True).start()
                return  # Success — exit retry loop

            except Exception as e:
                yield f"data: [ERROR] Attempt {attempt}/{MAX_RETRIES} failed: {str(e)}\n\n"
                if attempt >= MAX_RETRIES:
                    yield f"data: [CRITICAL] All {MAX_RETRIES} update attempts failed.\n\n"
                    yield "event: error\ndata: \n\n"
                    return

        yield "data: [CRITICAL] Exhausted all retry attempts.\n\n"
        yield "event: error\ndata: \n\n"

    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'  # Disable NGINX buffering
    return response


@login_required
def system_telemetry_api(request):
    """API endpoint for live polling of system telemetry"""
    telemetry = _get_memory_snapshot()
    return JsonResponse({'status': 'success', 'telemetry': telemetry})


def security_status_api(request):
    settings = SystemSettings.load()
    # Consume the fresh_login flag — once popped it won't fire again
    fresh_login = request.session.pop('fresh_login', False)
    return JsonResponse({
        'status': 'success',
        'has_password': bool(settings.lockscreen_password),
        'idle_lock_minutes': settings.idle_lock_minutes,
        'idle_logout_minutes': settings.idle_logout_minutes,
        'fresh_login': fresh_login,  # tells client to clear qt_locked
    })

def save_security_settings_api(request):
    if request.method == 'POST':
        settings = SystemSettings.load()
        
        pw = request.POST.get('lockscreen_password', '').strip()
        if pw:
            settings.lockscreen_password = pw
        elif 'lockscreen_password' in request.POST:
            # If submitted empty, remove the password
            settings.lockscreen_password = ''
            
        settings.idle_lock_minutes = int(request.POST.get('idle_lock_minutes', 5))
        settings.idle_logout_minutes = int(request.POST.get('idle_logout_minutes', 30))
        
        settings.save()
        return JsonResponse({'status': 'success', 'message': 'Security settings updated.'})
    return JsonResponse({'error': 'Invalid method'}, status=405)

def lockscreen_api(request):
    if request.method == 'POST':
        settings = SystemSettings.load()
        password = request.POST.get('password', '')
        
        # Support both hashed (from admin) and plain text (legacy/simple)
        is_correct = False
        if settings.lockscreen_password:
            from django.contrib.auth.hashers import check_password
            if check_password(password, settings.lockscreen_password):
                is_correct = True
            elif password == settings.lockscreen_password:
                is_correct = True
        
        if not settings.lockscreen_password or is_correct:
            return JsonResponse({'status': 'success'})
        return JsonResponse({'status': 'error', 'message': 'Incorrect Passcode'})
    return JsonResponse({'error': 'Invalid method'}, status=405)


# =====================================================================
# Neural Cortex Dashboard & Weight Editor Views
# =====================================================================

def _get_trader_safely(trader_id):
    if not trader_id:
        return None
    try:
        valid_id = int(trader_id)
        return PaperTrader.objects.filter(id=valid_id).first()
    except (ValueError, TypeError):
        return None


def _get_agent_and_path_for_trader(trader):
    if not trader or not trader.model_file:
        return None, None
    model_path = trader.model_file
    try:
        from src.models.ppo_agent import PPOAgent
        from control_panel.model_registry import read_model_bytes
        import io
        import torch
        
        # Default state_dim=4, action_dim=1 for ActorCritic PPO model
        state_dim = 4
        action_dim = 1
        try:
            model_bytes = read_model_bytes(model_path)
            state_dict = torch.load(io.BytesIO(model_bytes), map_location='cpu')
            if 'actor.0.weight' in state_dict:
                state_dim = state_dict['actor.0.weight'].shape[1]
                action_dim = state_dict['actor.4.weight'].shape[0]
        except Exception as shape_err:
            logger.debug(f"Failed to infer shapes from state_dict, using default (4, 1): {shape_err}")
            
        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
        agent.load_weights_from_bytes(read_model_bytes(model_path), model_path)
        return agent, model_path
    except Exception as e:
        logger.error(f"Error loading agent for trader: {e}")
        return None, None


@login_required
def neural_cortex_view(request):
    trader_id = request.GET.get('trader_id')
    trader = _get_trader_safely(trader_id)
    if not trader:
        trader = PaperTrader.objects.filter(status='RUNNING').first() or PaperTrader.objects.first()
    
    all_traders = PaperTrader.objects.all()
    from .models import OnlineLearningLog
    
    context = {
        'trader': trader,
        'all_traders': all_traders,
        'has_trader': trader is not None,
    }
    return render(request, 'neural_cortex.html', context)


@login_required
def neural_weights_api(request):
    trader_id = request.GET.get('trader_id')
    trader = _get_trader_safely(trader_id)
    if not trader:
        trader = PaperTrader.objects.filter(status='RUNNING').first() or PaperTrader.objects.first()
        
    if not trader or not trader.model_file:
        return JsonResponse({'error': 'No active trader or model found'}, status=404)
        
    agent, model_path = _get_agent_and_path_for_trader(trader)
    if not agent:
        return JsonResponse({'error': 'Failed to load model weights'}, status=500)
        
    from src.core.online_learner import OnlineLearner
    from .models import OnlineLearningLog
    
    # Instantiate learner to compute stats
    learner = OnlineLearner(agent, trader_id=trader.id)
    weights_data = learner.get_weight_summary()
    
    # Fetch last weight change reason from DB
    latest_update = OnlineLearningLog.objects.filter(
        trader=trader, 
        event_type__in=['UPDATE', 'MANUAL']
    ).first()
    
    reason = "No learning updates recorded yet."
    last_update_time = None
    if latest_update:
        reason = latest_update.reason
        last_update_time = latest_update.timestamp.isoformat()
        
    return JsonResponse({
        'trader_id': trader.id,
        'model_file': trader.model_file,
        'weights': weights_data,
        'latest_reason': reason,
        'last_update_time': last_update_time,
    })


@login_required
def neural_learning_log_api(request):
    trader_id = request.GET.get('trader_id')
    trader = _get_trader_safely(trader_id)
    if not trader:
        trader = PaperTrader.objects.filter(status='RUNNING').first() or PaperTrader.objects.first()
        
    if not trader:
        return JsonResponse({
            'logs': [], 
            'stats': {},
            'telemetry': _get_memory_snapshot()
        })
        
    from .models import OnlineLearningLog
    logs = OnlineLearningLog.objects.filter(trader=trader).order_by('-timestamp')[:100]
    
    logs_data = []
    for log in logs:
        logs_data.append({
            'id': log.id,
            'event_type': log.event_type,
            'symbol': log.symbol,
            'details': log.details,
            'reason': log.reason,
            'timestamp': log.timestamp.isoformat(),
        })
        
    # Calculate live stats from logs
    exits = OnlineLearningLog.objects.filter(trader=trader, event_type='EXIT')
    exit_rewards = []
    for e in exits:
        try:
            r = float(e.details.get('reward', 0.0))
            exit_rewards.append(r)
        except (ValueError, TypeError):
            exit_rewards.append(0.0)
            
    completed_trades = len(exit_rewards)
    win_count = sum(1 for r in exit_rewards if r > 0.0)
    loss_count = completed_trades - win_count
    win_rate = (win_count / max(1, completed_trades)) * 100 if completed_trades > 0 else 0.0
    avg_reward = sum(exit_rewards) / max(1, completed_trades) if completed_trades > 0 else 0.0
    
    # Fallback to TradeLog stats if no OnlineLearningLog exits exist
    if completed_trades == 0:
        from .models import TradeLog
        all_trades = list(TradeLog.objects.filter(trader=trader).order_by('timestamp'))
        completed_trades = sum(1 for t in all_trades if t.action == 'SELL')
        buy_prices = {}
        win_count = 0
        loss_count = 0
        total_pnl_pct = 0.0
        for t in all_trades:
            if t.action == 'BUY':
                if t.symbol not in buy_prices:
                    buy_prices[t.symbol] = []
                buy_prices[t.symbol].append(float(t.price))
            elif t.action == 'SELL':
                if t.symbol in buy_prices and len(buy_prices[t.symbol]) > 0:
                    bp = buy_prices[t.symbol].pop(0)
                    pnl_pct = ((float(t.price) - bp) / bp) * 100 if bp > 0 else 0.0
                    total_pnl_pct += pnl_pct
                    if pnl_pct > 0:
                        win_count += 1
                    else:
                        loss_count += 1
                else:
                    loss_count += 1
        win_rate = (win_count / max(1, completed_trades)) * 100 if completed_trades > 0 else 0.0
        avg_reward = (total_pnl_pct / 100.0) / max(1, completed_trades) if completed_trades > 0 else 0.0
        
    total_updates = OnlineLearningLog.objects.filter(trader=trader, event_type='UPDATE').count()
    
    # Calculate reward trend (last 20 exits)
    recent_exits = list(exits.order_by('timestamp')[:20])
    reward_trend = [float(e.details.get('reward', 0)) for e in recent_exits]
    if len(reward_trend) == 0 and completed_trades > 0:
        reward_trend = [avg_reward] * min(5, completed_trades)
    
    stats = {
        'completed_trades': completed_trades,
        'win_rate': round(win_rate, 2),
        'win_count': win_count,
        'loss_count': loss_count,
        'total_updates': total_updates,
        'avg_reward': round(avg_reward, 4),
        'reward_trend': reward_trend,
    }
        
    return JsonResponse({
        'logs': logs_data,
        'stats': stats,
        'telemetry': _get_memory_snapshot(),
    })


@login_required
@require_POST
def neural_weight_edit_api(request):
    try:
        data = json.loads(request.body)
        trader_id = data.get('trader_id')
        layer_name = data.get('layer_name')
        row = data.get('row')  # None for 1D arrays
        col = data.get('col')  # None
        new_val = float(data.get('value'))
        reason = data.get('reason', '').strip()
        
        if not reason:
            return JsonResponse({'status': 'error', 'message': 'A reason for editing weights is required.'}, status=400)
            
        trader = get_object_or_404(PaperTrader, id=trader_id)
        if not trader.model_file:
            return JsonResponse({'status': 'error', 'message': 'Trader has no model file.'}, status=400)
            
        agent, model_path = _get_agent_and_path_for_trader(trader)
        if not agent:
            return JsonResponse({'status': 'error', 'message': 'Failed to load model weights.'}, status=500)
            
        # Backup weights to previous weights file before making changes
        import torch
        prev_weights_filename = f"previous_weights_T{trader.id}.pth"
        try:
            prev_state_dict = {k: v.cpu().clone() for k, v in agent.policy.state_dict().items()}
            torch.save(prev_state_dict, prev_weights_filename)
        except Exception as e:
            logger.warning(f"Could not save weight pre-checkpoint: {e}")
            
        state_dict = agent.policy.state_dict()
        if layer_name not in state_dict:
            return JsonResponse({'status': 'error', 'message': f'Layer {layer_name} not found.'}, status=400)
            
        param = state_dict[layer_name]
        old_val = 0.0
        
        # Modify the tensor in-place
        with torch.no_grad():
            if param.dim() == 2:
                if row is None or col is None:
                    return JsonResponse({'status': 'error', 'message': 'Row and Col are required for 2D tensors.'}, status=400)
                old_val = float(param[row, col].item())
                param[row, col] = new_val
            elif param.dim() == 1:
                idx = row if row is not None else col
                if idx is None:
                    return JsonResponse({'status': 'error', 'message': 'Index is required for 1D tensors.'}, status=400)
                old_val = float(param[idx].item())
                param[idx] = new_val
            else:
                return JsonResponse({'status': 'error', 'message': f'Unsupported parameter dimension: {param.dim()}'}, status=400)
                
        # Write modified weights to model registry / storage
        import io
        buffer = io.BytesIO()
        agent.save_weights_to_buffer(buffer)
        weight_bytes = buffer.getvalue()
        
        if model_path.startswith('db:'):
            job_id = int(model_path.split(':')[1])
            TrainingJob.objects.filter(id=job_id).update(model_weights=weight_bytes)
        else:
            from pathlib import Path
            Path(model_path).write_bytes(weight_bytes)
            
        log_reason = f"Manual edit of {layer_name}[{row}, {col}] from {old_val:+.4f} to {new_val:+.4f} | Reason: {reason}"
        
        from src.core.online_learner import OnlineLearner
        learner = OnlineLearner(agent, trader_id=trader.id)
        learner._log_event(
            'MANUAL',
            details={
                'layer': layer_name,
                'row': row,
                'col': col,
                'old_val': old_val,
                'new_val': new_val,
                'reason': reason
            },
            reason=log_reason
        )
        
        # Automatically schedule a 3-day backtest to verify the modified weights
        from datetime import timedelta
        from django.utils import timezone
        
        eval_job = None
        try:
            from .models import EvaluationJob
            end_date = timezone.now().date()
            start_date = end_date - timedelta(days=3)
            
            eval_job = EvaluationJob.objects.create(
                model_file=trader.model_file,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                status='PENDING'
            )
            
            import sys
            from pathlib import Path
            log_dir = Path(__file__).parent.parent / "logs"
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"eval_job_{eval_job.id}.log"
            
            process = _spawn_background_process(
                [sys.executable, "run_evaluation_job.py", "--job_id", str(eval_job.id)],
                log_file,
            )
            eval_job.celery_task_id = str(process.pid)
            eval_job.save(update_fields=['celery_task_id'])
            logger.info(f"[ONLINE LEARNING] Spawned automatic evaluation job E{eval_job.id} for edited weights of model {trader.model_file}")
        except Exception as eval_err:
            logger.error(f"Failed to spawn automatic evaluation for edited weights: {eval_err}", exc_info=True)

        message = f"Weight updated from {old_val:+.4f} to {new_val:+.4f}."
        if eval_job:
            message += f" Automatically initiated a 3-day verification backtest (Job #E{eval_job.id}). View progress in the Evaluation Lab."

        return JsonResponse({
            'status': 'success',
            'message': message,
            'eval_job_id': eval_job.id if eval_job else None,
            'warning': 'If the bot is currently running, please restart it to apply the modified weights.' if trader.status == 'RUNNING' else None
        })
        
    except Exception as e:
        logger.error(f"Error editing weights: {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@login_required
def decision_flow_view(request):
    trader_id = request.GET.get('trader_id')
    trader = _get_trader_safely(trader_id)
    if not trader:
        trader = PaperTrader.objects.filter(status='RUNNING').first() or PaperTrader.objects.first()
    
    all_traders = PaperTrader.objects.all()
    context = {
        'trader': trader,
        'all_traders': all_traders,
        'has_trader': trader is not None,
    }
    return render(request, 'decision_flow.html', context)


@login_required
def simulate_decision_api(request):
    trader_id = request.GET.get('trader_id')
    
    # Parse parameter sliders inputs
    try:
        price_trend = float(request.GET.get('price_trend', 0.0))
        sentiment = float(request.GET.get('sentiment', 0.0))
        volatility = float(request.GET.get('volatility', 0.1))
        spread = float(request.GET.get('spread', 0.1))
    except (ValueError, TypeError):
        return JsonResponse({'error': 'Invalid parameters'}, status=400)

    state = np.array([price_trend, sentiment, volatility, spread], dtype=np.float32)

    trader = _get_trader_safely(trader_id)
    if not trader:
        trader = PaperTrader.objects.filter(status='RUNNING').first() or PaperTrader.objects.first()

    agent = None
    if trader and trader.model_file:
        agent, _ = _get_agent_and_path_for_trader(trader)

    actor_h1 = [0.0] * 6
    actor_h2 = [0.0] * 6
    critic_h1 = [0.0] * 5
    critic_h2 = [0.0] * 5
    act_val = 0.0
    val_val = 0.0
    is_simulated = True

    if agent and hasattr(agent.policy, 'actor'):
        try:
            import torch
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
                # Actor Forward
                x1_raw = agent.policy.actor[0](state_tensor)
                x1 = torch.tanh(x1_raw)
                x2_raw = agent.policy.actor[2](x1)
                x2 = torch.tanh(x2_raw)
                act_raw = agent.policy.actor[4](x2)
                act = torch.tanh(act_raw)
                
                # Critic Forward
                v1_raw = agent.policy.critic[0](state_tensor)
                v1 = torch.tanh(v1_raw)
                v2_raw = agent.policy.critic[2](v1)
                v2 = torch.tanh(v2_raw)
                val = agent.policy.critic[4](v2)
                
                # Gather values
                actor_h1 = x1[0, :6].cpu().tolist()
                actor_h2 = x2[0, :6].cpu().tolist()
                critic_h1 = v1[0, :5].cpu().tolist()
                critic_h2 = v2[0, :5].cpu().tolist()
                act_val = float(act[0, 0].item())
                val_val = float(val[0, 0].item())
                is_simulated = False
        except Exception as e:
            logger.warning(f"Error during PPO forward pass in Decision Flow: {e}")

    # Fallback to deterministic mock math if model is missing/error
    if is_simulated:
        # Layer 1 actor
        actor_h1[0] = float(np.tanh(price_trend + sentiment))
        actor_h1[1] = float(np.tanh(sentiment - volatility))
        actor_h1[2] = float(np.tanh(price_trend * 1.5))
        actor_h1[3] = float(np.tanh(-volatility + spread))
        actor_h1[4] = float(np.tanh(spread * 0.8))
        actor_h1[5] = float(np.tanh(-sentiment))
        
        # Layer 2 actor
        actor_h2[0] = float(np.tanh(actor_h1[0] + actor_h1[2]))
        actor_h2[1] = float(np.tanh(actor_h1[1] - actor_h1[3]))
        actor_h2[2] = float(np.tanh(actor_h1[4] * 1.2))
        actor_h2[3] = float(np.tanh(actor_h1[5] - actor_h1[0]))
        actor_h2[4] = float(np.tanh(actor_h1[2] + actor_h1[3]))
        actor_h2[5] = float(np.tanh(actor_h1[1] * 0.9))

        # Actor output recommendation
        act_val = float(np.tanh(sum(actor_h2) / 6.0))

        # Layer 1 critic
        critic_h1[0] = float(np.tanh(price_trend * 0.5))
        critic_h1[1] = float(np.tanh(sentiment * 0.5))
        critic_h1[2] = float(np.tanh(volatility * -0.8))
        critic_h1[3] = float(np.tanh(spread * 0.3))
        critic_h1[4] = float(np.tanh(price_trend - volatility))
        
        # Layer 2 critic
        critic_h2[0] = float(np.tanh(critic_h1[0] + critic_h1[1]))
        critic_h2[1] = float(np.tanh(critic_h1[2] + critic_h1[3]))
        critic_h2[2] = float(np.tanh(critic_h1[4] * 0.5))
        critic_h2[3] = float(np.tanh(critic_h1[1] - critic_h1[0]))
        critic_h2[4] = float(np.tanh(critic_h1[3] * 0.9))
        
        # Critic value estimate
        val_val = float(np.tanh(sum(critic_h2) / 5.0) * 0.15)

    # Human-readable decision narrative
    price_trend_desc = "strongly bullish (upward momentum)" if price_trend > 0.4 else ("strongly bearish (downward momentum)" if price_trend < -0.4 else ("moderately bullish" if price_trend > 0.1 else ("moderately bearish" if price_trend < -0.1 else "neutral / rangebound")))
    sentiment_desc = "extreme greed / high social optimism" if sentiment > 0.5 else ("moderate greed / optimistic" if sentiment > 0.15 else ("extreme fear / high social panic" if sentiment < -0.5 else ("moderate fear / pessimistic" if sentiment < -0.15 else "neutral / balanced")))
    volatility_desc = "hyper-volatile (elevated tail risk)" if volatility > 0.7 else ("moderate volatility" if volatility > 0.3 else "exceptionally stable / low volatility")
    spread_desc = "illiquid (wide bid-ask spread)" if spread > 0.6 else "highly liquid (tight bid-ask spread)"

    narrative = [
        f"1. Sensory Core: Price Trend is {price_trend_desc} ({price_trend:+.2f}) aligned with a {sentiment_desc} sentiment bias ({sentiment:+.2f}). Market context logs {volatility_desc} ({volatility:.2f}) and a {spread_desc} ({spread:.2f}) liquidity profile.",
        f"2. Sensory-to-Feature Mapping (Layer 1): Input vectors propagate to the hidden layer detectors. Actor feature nodes (HA1) fire at an average intensity of {sum(map(abs, actor_h1))/6:.2f}, while Critic risk nodes (HC1) register a baseline activation of {sum(map(abs, critic_h1))/5:.2f}.",
        f"3. High-Level Conception (Layer 2): Feature detections consolidate. Hidden Node HA2.1 (bullish filter) fires at {actor_h2[0]:+.2f}, while Node HC2.3 (downside volatility filter) registers {critic_h2[2]:+.2f}.",
        f"4. Actor Execution Vector: The output node (ACT) recommendation resolves to {act_val:+.2f}, signaling a conviction index of {abs(act_val)*100:.1f}% towards a {'BUY (Long Entry)' if act_val >= 0.15 else ('SELL (Short Entry)' if act_val <= -0.15 else 'HOLD (Neutral/No Action)')} strategy.",
        f"5. Critic Valuation Verdict: The output node (VAL) estimates expected future net return at {val_val:+.4f} normalized PnL units, signifying a {'highly favorable' if val_val > 0.05 else ('unfavorable/negative' if val_val < -0.05 else 'flat/neutral')} expectancy."
    ]

    # Actionable trading tips
    tips = []
    if volatility > 0.7:
        tips.append("⚠️ Tail-Risk Alert: Volatility is hyper-elevated. Sudden price sweeps can trigger stop-losses prematurely. We recommend reducing position sizing by 50% or transitioning the bot to passive mode.")
    if spread > 0.6:
        tips.append("⚠️ High Slippage Risk: Bid-ask spread is wide. Market orders will suffer heavy execution drag. We recommend utilizing strict Limit Orders instead of Market Orders.")
    if price_trend < -0.3 and sentiment > 0.35:
        tips.append("📈 Bullish Divergence: Price action is in a downward trend ({price_trend:+.2f}) but crowd sentiment shows positive greed ({sentiment:+.2f}). This often indicates a retail buying trap or a potential double-bottom trend reversal. Monitor order books closely.")
    if price_trend > 0.3 and sentiment < -0.35:
        tips.append("📉 Bearish Divergence: Price is trending upward ({price_trend:+.2f}) but crowd sentiment is negative/fearful ({sentiment:+.2f}). Watch for trend exhaustion, potential retail short squeezes, or overhead resistance levels.")
    if spread > 0.75 and volatility > 0.7:
        tips.append("🛑 Liquidity Trap: High spread and high volatility detected simultaneously. Slippage will be severe and execution will lag. Stand aside and pause active trading until market conditions stabilize.")
    if val_val > 0.05 and act_val > 0.3:
        tips.append("🚀 High Conviction Long: Critic valuation exhibits positive expectancy ({val_val:+.4f}) alongside a strong bullish actor recommendation. High probability long setup.")
    if val_val < -0.05 and act_val < -0.3:
        tips.append("💥 High Conviction Short: Critic valuation predicts negative returns ({val_val:+.4f}) alongside a strong bearish actor recommendation. High probability short setup.")
    
    # Fallback default tip if list is empty
    if not tips:
        if abs(act_val) < 0.15:
            tips.append("🔒 Rangebound Regime: Model outputs are close to zero. The model recommends holding cash, avoiding over-trading, and keeping fee drag to a minimum.")
        elif act_val >= 0.15:
            tips.append("📈 Bullish Stance: The model favors buy setups. Ensure your stop-loss configurations and profit targets are active.")
        else:
            tips.append("📉 Bearish Stance: The model favors sell setups. Ensure short risk guardrails and margin buffers are active.")

    return JsonResponse({
        'price_trend': price_trend,
        'sentiment': sentiment,
        'volatility': volatility,
        'spread': spread,
        'actor_h1': actor_h1,
        'actor_h2': actor_h2,
        'critic_h1': critic_h1,
        'critic_h2': critic_h2,
        'actor_out': act_val,
        'critic_out': val_val,
        'narrative': narrative,
        'tips': tips,
        'is_simulated': is_simulated
    })


@login_required
def past_decisions_api(request):
    trader_id = request.GET.get('trader_id')
    trader = _get_trader_safely(trader_id)
    if not trader:
        trader = PaperTrader.objects.filter(status='RUNNING').first() or PaperTrader.objects.first()

    if not trader:
        return JsonResponse({'trades': [], 'total_pages': 1, 'current_page': 1, 'total_count': 0})

    try:
        page = int(request.GET.get('page', 1))
    except ValueError:
        page = 1
    page_size = 5

    from .models import OnlineLearningLog
    logs_query = OnlineLearningLog.objects.filter(
        trader=trader, 
        event_type__in=['ENTRY', 'EXIT', 'DECISION']
    ).order_by('-timestamp')

    import math
    total_count = logs_query.count()
    use_tradelog_fallback = False
    
    if total_count == 0:
        from .models import TradeLog
        tradelogs_query = TradeLog.objects.filter(trader=trader).order_by('-timestamp')
        total_count = tradelogs_query.count()
        use_tradelog_fallback = True

    total_pages = max(1, math.ceil(total_count / page_size))
    
    if page < 1:
        page = 1
    elif page > total_pages:
        page = total_pages

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    trades_list = []
    import re
    
    if use_tradelog_fallback:
        from .models import TradeLog
        logs = TradeLog.objects.filter(trader=trader).order_by('-timestamp')[start_idx:end_idx]
        for log in logs:
            event_type = 'ENTRY' if log.action == 'BUY' else 'EXIT'
            trades_list.append({
                'id': log.id,
                'event_type': event_type,
                'symbol': log.symbol,
                'timestamp': log.timestamp.isoformat(),
                'reason': f"Execution fallback: {log.action} {log.quantity:.6f} {log.symbol} @ ${log.price:,.2f}",
                'price_trend': 0.0,
                'sentiment': float(log.sentiment_score or 0.0),
                'volatility': 0.2,
                'spread': 0.1
            })
    else:
        logs = logs_query[start_idx:end_idx]
        for log in logs:
            # Default inputs to populate
            price_trend = 0.0
            sentiment = 0.0
            volatility = 0.2
            spread = 0.1
            
            # 1. Try to extract directly from details JSON if available (especially for DECISION logs)
            if isinstance(log.details, dict) and 'price_trend' in log.details:
                price_trend = float(log.details.get('price_trend', 0.0))
                sentiment = float(log.details.get('sentiment', 0.0))
                volatility = float(log.details.get('volatility', 0.2))
                spread = float(log.details.get('spread', 0.1))
            else:
                try:
                    # Extract inputs from reasons (for legacy ENTRY/EXIT logs)
                    sent_m = re.search(r'Sentiment:\s*([+-]?\d*\.?\d+)', log.reason, re.IGNORECASE)
                    trend_m = re.search(r'Price Trend is\s*([+-]?\d*\.?\d+)', log.reason, re.IGNORECASE)
                    vol_m = re.search(r'Volatility is\s*([+-]?\d*\.?\d+)', log.reason, re.IGNORECASE)
                    
                    if sent_m: sentiment = float(sent_m.group(1))
                    if trend_m: price_trend = float(trend_m.group(1))
                    if vol_m: volatility = float(vol_m.group(1))
                except Exception:
                    pass

            trades_list.append({
                'id': log.id,
                'event_type': log.event_type,
                'symbol': log.symbol,
                'timestamp': log.timestamp.isoformat(),
                'reason': log.reason,
                'price_trend': price_trend,
                'sentiment': sentiment,
                'volatility': volatility,
                'spread': spread
            })

    return JsonResponse({
        'trades': trades_list,
        'total_pages': total_pages,
        'current_page': page,
        'total_count': total_count
    })


@login_required
def download_report_view(request):
    import io
    from datetime import datetime
    from django.utils import timezone
    from django.http import HttpResponse
    from django.shortcuts import render, get_object_or_404
    from django.db.models import Avg
    from .models import PaperTrader, OnlineLearningLog
    
    trader_id = request.GET.get('trader_id')
    is_mock = False
    trader = None
    
    if trader_id == '0':
        is_mock = True
    else:
        trader = _get_trader_safely(trader_id)
        if not trader and trader_id:
            is_mock = True
            
    if not trader and not is_mock:
        trader = PaperTrader.objects.filter(status='RUNNING').first() or PaperTrader.objects.first()
        if not trader:
            is_mock = True

    if is_mock:
        class MockTrader:
            def __init__(self):
                self.id = 0
                self.name = "Simulated Paper Trader (Demo Mode)"
                self.status = "RUNNING"
                self.model_file = "best_model.pth"
                self.symbol = "BTC/USD"
                self.balance = 50000.00
        trader = MockTrader()
        
    # Get system resources usage snapshot
    telemetry = _get_memory_snapshot() or {
        'available': False,
        'system_used_percent': 0.0,
        'system_used_gb': 0.0,
        'system_total_gb': 0.0,
        'trader_limit_mb': 0,
        'threshold_percent': 85,
        'cpu_percent': 0.0
    }
    
    class MockLog:
        def __init__(self, event_type, symbol, reason, timestamp):
            self.event_type = event_type
            self.symbol = symbol
            self.reason = reason
            self.timestamp = timestamp

    class MockQuerySet(list):
        def exists(self):
            return len(self) > 0
            
    if is_mock:
        # Generate mock events
        import random
        from datetime import timedelta
        now_time = datetime.now(timezone.get_current_timezone())
        mock_events = [
            ('ENTRY', 'BTC/USD', 'State Analysis: strongly bullish (+0.55) trend confirmed. Initiating dynamic grid position.'),
            ('UPDATE', 'BTC/USD', 'Policy update: weight optimization completed. Mean drift: 0.000142.'),
            ('EXIT', 'BTC/USD', 'Dynamic target reached. Reward: +0.0245 PnL.'),
            ('ENTRY', 'ETH/USD', 'State Analysis: moderately bearish (-0.21) trend. Initiating hedging grid position.'),
            ('DECISION', 'BTC/USD', 'Manual sandbox decision firing. Price Trend: +0.20, Sentiment: +0.40, Volatility: 0.1500, Action Signal: +0.1800')
        ]
        mock_logs_list = []
        for i, (etype, sym, rsn) in enumerate(mock_events):
            mock_logs_list.append(MockLog(
                event_type=etype,
                symbol=sym,
                reason=rsn,
                timestamp=now_time - timedelta(minutes=i*15)
            ))
        recent_logs = MockQuerySet(mock_logs_list)
        
        completed_trades = 12
        win_count = 8
        loss_count = 4
        win_rate = 66.67
        avg_reward = 0.0152
        total_updates = 34
    else:
        # Get active learning logs
        recent_logs = OnlineLearningLog.objects.filter(
            trader=trader,
            event_type__in=['ENTRY', 'EXIT', 'UPDATE', 'MANUAL']
        ).order_by('-timestamp')[:20]
        
        # Standard statistics
        exits = OnlineLearningLog.objects.filter(trader=trader, event_type='EXIT')
        exit_rewards = []
        for e in exits:
            try:
                r = float(e.details.get('reward', 0.0))
                exit_rewards.append(r)
            except (ValueError, TypeError):
                exit_rewards.append(0.0)
                
        completed_trades = len(exit_rewards)
        win_count = sum(1 for r in exit_rewards if r > 0.0)
        loss_count = completed_trades - win_count
        win_rate = (win_count / max(1, completed_trades)) * 100 if completed_trades > 0 else 0.0
        avg_reward = sum(exit_rewards) / max(1, completed_trades) if completed_trades > 0 else 0.0
        total_updates = OnlineLearningLog.objects.filter(trader=trader, event_type='UPDATE').count()
    
    # Compute weight statistics summary
    weight_summary = {}
    agent = None
    if not is_mock:
        agent, _ = _get_agent_and_path_for_trader(trader)
        if agent:
            try:
                from src.core.online_learner import OnlineLearner
                learner = OnlineLearner(agent, trader_id=trader.id)
                weight_summary = learner.get_weight_summary()
            except Exception as e:
                logger.warning(f"Failed to get weight summary from learner: {e}")
                
    if not weight_summary:
        # Load from best_model.pth or fallback to random weights
        try:
            from src.models.ppo_agent import PPOAgent
            from control_panel.model_registry import read_model_bytes
            import torch
            
            model_bytes = read_model_bytes("best_model.pth")
            state_dict = torch.load(io.BytesIO(model_bytes), map_location='cpu')
            state_dim = 4
            action_dim = 1
            if 'actor.0.weight' in state_dict:
                state_dim = state_dict['actor.0.weight'].shape[1]
                action_dim = state_dict['actor.4.weight'].shape[0]
            agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
            agent.load_weights_from_bytes(model_bytes, "best_model.pth")
        except Exception as e:
            logger.warning(f"Could not load best_model.pth for mock trader: {e}. Generating random weights.")
            try:
                from src.models.ppo_agent import PPOAgent
                agent = PPOAgent(state_dim=4, action_dim=1)
            except Exception as inner_e:
                logger.error(f"Failed to instantiate fallback PPOAgent: {inner_e}")
                agent = None
                
        if agent:
            try:
                from src.core.online_learner import OnlineLearner
                trader_id_val = trader.id if trader else 0
                learner = OnlineLearner(agent, trader_id=trader_id_val)
                weight_summary = learner.get_weight_summary()
            except Exception as e:
                logger.warning(f"Failed to get weight summary for agent: {e}")
                
    # Prepare context
    now_str = datetime.now(timezone.get_current_timezone()).strftime('%Y-%m-%d %H:%M:%S %Z')
    context = {
        'trader': trader,
        'telemetry': telemetry,
        'logs': recent_logs,
        'weights': weight_summary,
        'win_rate': round(win_rate, 2),
        'win_count': win_count,
        'loss_count': loss_count,
        'completed_trades': completed_trades,
        'avg_reward': round(avg_reward, 4),
        'total_updates': total_updates,
        'timestamp': now_str,
    }
    
    # If format=markdown is requested, generate a raw downloadable .md file
    if request.GET.get('format') == 'markdown':
        md_content = f"""# AI Trader - Neural Cortex & Diagnostics Report
Generated on: {now_str}

## 1. Active Bot Configuration
- **Bot Name**: {trader.name}
- **Status**: {trader.status}
- **Model File**: {trader.model_file}
- **Active Symbol**: {trader.symbol}
- **Balance / Equity**: ${trader.balance:.2f}

## 2. System Telemetry & Resource Usage
- **CPU Utilization**: {telemetry.get('cpu_percent', 0.0):.1f}%
- **System Memory (RAM)**: {telemetry.get('system_used_percent', 0.0):.1f}% ({telemetry.get('system_used_gb', 0.0):.2f} GB / {telemetry.get('system_total_gb', 0.0):.2f} GB)
- **Memory Safety Alert Threshold**: {telemetry.get('threshold_percent', 85)}% (Auto-pauses paper trader if breached)

## 3. Neural Active Learning Stats
- **Completed Trades**: {completed_trades}
- **Win Rate**: {win_rate:.2f}% ({win_count} wins, {loss_count} losses)
- **Average PnL Reward**: {avg_reward:+.4f}
- **Total PPO Optimizer Updates**: {total_updates}

## 4. Model Weights Summary Stats
"""
        if weight_summary:
            for layer_name, info in weight_summary.items():
                md_content += f"""
### Layer: `{layer_name}`
- **Shape**: {info['shape']}
- **Current Weights**: Min: {info['min']:+.4f} | Max: {info['max']:+.4f} | Mean: {info['mean']:+.4f} | StdDev: {info['std']:.4f}
- **Previous weights**: Min: {info.get('prev_min', 0.0):+.4f} | Max: {info.get('prev_max', 0.0):+.4f} | Mean: {info.get('prev_mean', 0.0):+.4f}
- **Mean Absolute Weight Drift**: {info.get('mean_diff', 0.0):.6f}
"""
        else:
            md_content += "\n*(No weight statistics summaries available for this model file)*\n"
            
        md_content += "\n## 5. Recent Learning & Decision Log Feed\n"
        if recent_logs.exists():
            for log in recent_logs:
                log_time = log.timestamp.strftime('%H:%M:%S')
                md_content += f"- **[{log.event_type}]** {log_time} - Symbol: {log.symbol} - Reason: {log.reason}\n"
        else:
            md_content += "\n*(No learning events recorded yet)*\n"
            
        md_content += """
## 6. Foolproof Operating Guide & Profit Manual

### Diagnosing Model Underperformance
1. **Regime Shift Detection**: If the win rate drops below 45% or average rewards turn negative over the last 20 exits, the model is likely experiencing regime mismatch (e.g. trading flat ranges with indicators optimized for trend breakouts).
2. **System Health Alerts**: If CPU or memory utilization is consistently high (>80%), paper trading loops may lag. If memory usage reaches the 85% threshold, the active trader engine will automatically suspend live updates to protect the local database from write locks.
3. **Weight Drift Audit**: Check the "Mean Absolute Weight Drift" values. If drift is zero, the model has stopped updates. If drift is high but win rate is declining, the online learner rate is too high and is overriding historical generalizable patterns.

### How to Tweak Weights Profitably
- **Adjust Trend Sensitivity**: Input-to-actor layer weights (`actor.0.weight`) control input responses. If charts are bullish but the bot keeps shorting, find the index of the "Price Trend" input. If its connection weights to the active hidden nodes are negative, you can override them to be slightly positive.
- **Tweak Risk Margin**: Critic output layer weights (`critic.4.weight`) compute reward expectancy. If the model is taking overly aggressive positions with small reward setups, decrease the critic's magnitude by scaling down the weights.
- **Verification Rule**: Any manual modification of neuron weight elements automatically schedules a 3-day verification backtest inside the Evaluation Lab. Do not resume live trading until this evaluation job completes successfully.
"""
        response = HttpResponse(md_content, content_type='text/markdown')
        response['Content-Disposition'] = f'attachment; filename="neural_diagnostics_report_{trader.id}.md"'
        return response
        
    return render(request, 'report_export.html', context)


@login_required
@require_POST
def record_decision_api(request):
    try:
        data = json.loads(request.body)
        trader_id = data.get('trader_id')
        price_trend = float(data.get('price_trend', 0.0))
        sentiment = float(data.get('sentiment', 0.0))
        volatility = float(data.get('volatility', 0.1))
        spread = float(data.get('spread', 0.1))
        action_val = float(data.get('action_val', 0.0))
        
        trader = get_object_or_404(PaperTrader, id=trader_id)
        
        from .models import OnlineLearningLog
        details = {
            'price_trend': price_trend,
            'sentiment': sentiment,
            'volatility': volatility,
            'spread': spread,
            'action_val': action_val,
            'manual_sandbox': True
        }
        
        reason = f"Manual sandbox decision firing. Price Trend: {price_trend:+.2f}, Sentiment: {sentiment:+.2f}, Volatility: {volatility:.4f}, Action Signal: {action_val:+.4f}"
        
        log = OnlineLearningLog.objects.create(
            trader=trader,
            event_type='DECISION',
            symbol=trader.symbol or 'SANDBOX',
            details=details,
            reason=reason
        )
        
        # Prune old decision checks (keep max 20)
        old_decisions = OnlineLearningLog.objects.filter(trader=trader, event_type='DECISION').order_by('-timestamp')[20:]
        if old_decisions.exists():
            old_ids = list(old_decisions.values_list('id', flat=True))
            OnlineLearningLog.objects.filter(id__in=old_ids).delete()
            
        return JsonResponse({'status': 'success', 'message': 'Sandbox decision firing logged successfully!'})
    except Exception as e:
        logger.error(f"Error recording decision: {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


# ── RELAX LOUNGE (GAMING HUB) VIEWS ──

import urllib.request
import json
import re
import os
import subprocess
from html.parser import HTMLParser
from django.contrib import messages
from .models import Game, GameGuide, GameVideo

def fetch_steam_details(app_id):
    try:
        url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode())
            if str(app_id) in data and data[str(app_id)]['success']:
                game_data = data[str(app_id)]['data']
                name = game_data.get('name')
                cover = game_data.get('header_image')
                
                # Check for animated movies/trailers first!
                bg = None
                movies = game_data.get('movies')
                if movies:
                    first_movie = movies[0]
                    if 'webm' in first_movie and 'max' in first_movie['webm']:
                        bg = first_movie['webm']['max']
                    elif 'mp4' in first_movie and 'max' in first_movie['mp4']:
                        bg = first_movie['mp4']['max']
                
                # Fallback to background image or screenshot
                if not bg:
                    bg = game_data.get('background') or (game_data.get('screenshots', [{}])[0].get('path_full') if game_data.get('screenshots') else None)
                
                if name:
                    return name, cover, bg
    except Exception as e:
        logger.warning(f"Steam appdetails API failed for AppID {app_id}: {e}")
        
    # Fallback to direct storefront HTML scraping (highly reliable for unreleased/unindexed games)
    try:
        store_url = f"https://store.steampowered.com/app/{app_id}/"
        store_req = urllib.request.Request(store_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        with urllib.request.urlopen(store_req, timeout=8) as store_res:
            store_html = store_res.read().decode('utf-8', errors='ignore')
            
        import re
        title_m = re.search(r'id="appHubAppName"[^>]*>([^<]+)</div>', store_html)
        title = title_m.group(1).strip() if title_m else None
        
        if title:
            cover_m = re.search(r'class="game_header_image_full"[^>]*src="([^"]+)"', store_html)
            cover = cover_m.group(1).strip() if cover_m else f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{app_id}/header.jpg"
            
            # Scrape movie/microtrailer background video
            bg = None
            m = re.search(r'data-microtrailer-webm-source="([^"]+)"', store_html)
            if m:
                bg = m.group(1).replace('http://', 'https://').strip()
            if not bg:
                m = re.search(r'data-microtrailer-mp4-source="([^"]+)"', store_html)
                if m:
                    bg = m.group(1).replace('http://', 'https://').strip()
            if not bg:
                m = re.search(r'data-webm-source="([^"]+)"', store_html)
                if m:
                    bg = m.group(1).replace('http://', 'https://').strip()
            if not bg:
                m = re.search(r'data-mp4-source="([^"]+)"', store_html)
                if m:
                    bg = m.group(1).replace('http://', 'https://').strip()
                    
            if not bg:
                bg = f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{app_id}/library_hero.jpg"
                
            return title, cover, bg
    except Exception as scrape_err:
        logger.error(f"Steam storefront HTML details scrape failed for AppID {app_id}: {scrape_err}")
        
    return None, None, None


def search_steam_game_id(query):
    import urllib.parse
    import re
    quoted_query = urllib.parse.quote(query)
    try:
        url = f"https://store.steampowered.com/api/storesearch/?term={quoted_query}&l=english&cc=US"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=6) as response:
            data = json.loads(response.read().decode())
            if 'items' in data and data['items']:
                return data['items'][0].get('id')
    except Exception as e:
        logger.warning(f"Steam store search API failed for query '{query}': {e}")
        
    # Fallback to suggesting autocompletion endpoint
    try:
        suggest_url = f"https://store.steampowered.com/search/suggest?term={quoted_query}&f=games"
        suggest_req = urllib.request.Request(suggest_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        with urllib.request.urlopen(suggest_req, timeout=5) as suggest_res:
            suggest_html = suggest_res.read().decode('utf-8', errors='ignore')
        
        m = re.search(r'href="https://store.steampowered.com/app/(\d+)/', suggest_html)
        if m:
            return int(m.group(1))
    except Exception as suggest_err:
        logger.error(f"Steam suggest ID search failed for query '{query}': {suggest_err}")
    return None


@login_required
def relax_browse_dir(request):
    import os
    import platform
    current_path = request.GET.get('path', '').strip()
    
    if not current_path:
        if platform.system() == 'Windows':
            current_path = os.environ.get('USERPROFILE', 'C:\\')
        else:
            current_path = os.environ.get('HOME', '/')
            
    current_path = os.path.abspath(current_path)
    
    if not os.path.exists(current_path) or not os.path.isdir(current_path):
        if platform.system() == 'Windows':
            current_path = 'C:\\'
        else:
            current_path = '/'
            
    drives = []
    if platform.system() == 'Windows':
        import string
        from ctypes import windll
        bitmask = windll.kernel32.GetLogicalDrives()
        for letter in string.ascii_uppercase:
            if bitmask & 1:
                drives.append(f"{letter}:\\")
            bitmask >>= 1
            
    parent_path = os.path.dirname(current_path)
    if parent_path == current_path:
        parent_path = ""
        
    items = []
    try:
        for entry in os.scandir(current_path):
            try:
                is_dir = entry.is_dir()
                if entry.name.startswith('.') or (platform.system() == 'Windows' and entry.name.startswith('$')):
                    continue
                items.append({
                    'name': entry.name,
                    'is_dir': is_dir,
                    'path': entry.path
                })
            except OSError:
                continue
        items.sort(key=lambda x: (not x['is_dir'], x['name'].lower()))
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
        
    return JsonResponse({
        'current_path': current_path,
        'parent_path': parent_path,
        'drives': drives,
        'items': items
    })


class SimpleHTMLScraper(HTMLParser):
    def __init__(self, target_tag='body'):
        super().__init__()
        self.text_list = []
        self.target_tag = target_tag
        self.in_target = False
        self.ignore_tags = {'script', 'style', 'nav', 'footer', 'header', 'iframe', 'noscript', 'aside'}
        self.void_elements = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}
        self.current_tag = None
        self.depth_ignore = 0
        self.target_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.ignore_tags:
            self.depth_ignore += 1
        self.current_tag = tag

        if tag in self.void_elements:
            return

        if tag == self.target_tag:
            if not self.in_target:
                self.in_target = True
                self.target_depth = 1
            else:
                self.target_depth += 1
        elif self.in_target:
            self.target_depth += 1

    def handle_endtag(self, tag):
        if tag in self.ignore_tags:
            self.depth_ignore = max(0, self.depth_ignore - 1)

        if tag in self.void_elements:
            return

        if self.in_target:
            self.target_depth -= 1
            if self.target_depth <= 0:
                self.in_target = False

    def handle_data(self, data):
        if self.in_target and self.depth_ignore == 0:
            text = data.strip()
            if text:
                if self.current_tag in {'h1', 'h2', 'h3', 'h4'}:
                    self.text_list.append(f"\n### {text}\n")
                elif self.current_tag == 'p':
                    self.text_list.append(f"\n{text}\n")
                elif self.current_tag == 'li':
                    self.text_list.append(f"- {text}")
                else:
                    self.text_list.append(text)


def scrape_guide_content(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            html_content = response.read().decode('utf-8', errors='ignore')
            
            title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else "Scraped Guide"
            
            # Remove title suffix if present
            for suffix in [" - Gamestegy", " - Steam Community", " - Wiki"]:
                if title.endswith(suffix):
                    title = title[:-len(suffix)]
            
            # Determine target container
            target = 'body'
            if '<article' in html_content.lower():
                target = 'article'
            elif '<main' in html_content.lower():
                target = 'main'
                
            parser = SimpleHTMLScraper(target_tag=target)
            parser.feed(html_content)
            
            content = "\n".join(parser.text_list)
            content = re.sub(r'\n\s*\n', '\n\n', content)
            return title, content
    except Exception as e:
        return "Failed to fetch guide", f"Error occurred while fetching or parsing webpage: {str(e)}"


@login_required
def relax_view(request):
    games = Game.objects.filter(is_active=True)
    selected_game = None
    game_id = request.GET.get('game_id')
    if game_id:
        selected_game = get_object_or_404(Game, id=game_id)
        
    # Throttled background Steam Sync (once every 3 hours)
    settings = SystemSettings.load()
    steam_username = settings.steam_username or request.session.get('steam_username')
    if steam_username:
        from django.core.cache import cache
        cache_key = f"auto_steam_sync_{steam_username}"
        if not cache.get(cache_key):
            cache.set(cache_key, "running", 10800) # Lock for 3 hours
            import threading
            def run_sync():
                try:
                    logger.info(f"Auto Steam sync triggered in background for {steam_username}...")
                    sync_steam_playtimes_helper(steam_username)
                except Exception as ex:
                    logger.error(f"Auto Steam sync thread failed: {ex}")
            t = threading.Thread(target=run_sync)
            t.daemon = True
            t.start()
    
    tracked_hours = 0.0
    if selected_game:
        tracked_hours = max(0.0, selected_game.hours_played - selected_game.playtime_offset)

    steam_username_session = request.session.get('steam_username', '')
    
    context = {
        'games': games,
        'selected_game': selected_game,
        'tracked_hours': tracked_hours,
        'page_title': 'Relax Lounge',
        'settings': settings,
        'steam_username_session': steam_username_session,
    }
    return render(request, 'relax.html', context)


@login_required
@require_POST
def relax_add_game(request):
    name = request.POST.get('name', '').strip()
    steam_app_id = request.POST.get('steam_app_id', '').strip() or None
    local_path = request.POST.get('local_path', '').strip() or None
    hours_played = float(request.POST.get('hours_played', 0.0) or 0.0)
    playtime_offset = float(request.POST.get('playtime_offset', 0.0) or 0.0)
    years_played = float(request.POST.get('years_played', 1.0) or 1.0)
    cover_image_url = request.POST.get('cover_image_url', '').strip() or None
    animated_bg_url = request.POST.get('animated_bg_url', '').strip() or None

    # Auto-resolve Steam App ID by name search if empty!
    if not steam_app_id and name:
        searched_id = search_steam_game_id(name)
        if searched_id:
            steam_app_id = str(searched_id)

    if steam_app_id:
        steam_name, steam_cover, steam_bg = fetch_steam_details(steam_app_id)
        if steam_name:
            if not name:
                name = steam_name
            if not cover_image_url:
                cover_image_url = steam_cover
            if not animated_bg_url:
                animated_bg_url = steam_bg

    if not name:
        messages.error(request, "Game name is required.")
        return redirect('relax_view')

    if not cover_image_url:
        cover_image_url = "https://images.unsplash.com/photo-1538481199705-c710c4e965fc?q=80&w=350&auto=format&fit=crop"

    # Total hours played includes prior platform offset hours
    total_hours = hours_played + playtime_offset

    game = Game.objects.create(
        name=name,
        steam_app_id=steam_app_id,
        local_path=local_path,
        hours_played=total_hours,
        playtime_offset=playtime_offset,
        years_played=years_played,
        cover_image_url=cover_image_url,
        animated_bg_url=animated_bg_url
    )
    messages.success(request, f"Game '{game.name}' added successfully to your library!")
    return redirect('relax_view')


@login_required
@require_POST
def relax_delete_game(request, game_id):
    game = get_object_or_404(Game, id=game_id)
    game.is_active = False
    game.save()
    messages.success(request, f"Removed '{game.name}' from library.")
    return redirect('relax_view')


@login_required
@require_POST
def relax_edit_game(request, game_id):
    game = get_object_or_404(Game, id=game_id)
    name = request.POST.get('name', '').strip()
    steam_app_id = request.POST.get('steam_app_id', '').strip() or None
    local_path = request.POST.get('local_path', '').strip() or None
    hours_played = float(request.POST.get('hours_played', 0.0) or 0.0)
    playtime_offset = float(request.POST.get('playtime_offset', 0.0) or 0.0)
    years_played = float(request.POST.get('years_played', 1.0) or 1.0)
    cover_image_url = request.POST.get('cover_image_url', '').strip() or None
    animated_bg_url = request.POST.get('animated_bg_url', '').strip() or None

    if not name:
        messages.error(request, "Game name is required.")
        return redirect('relax_view')

    if not cover_image_url:
        cover_image_url = "https://images.unsplash.com/photo-1538481199705-c710c4e965fc?q=80&w=350&auto=format&fit=crop"

    # Calculate tracked playtime base
    tracked_hours = game.hours_played - game.playtime_offset
    if tracked_hours < 0:
        tracked_hours = 0

    game.name = name
    game.steam_app_id = steam_app_id
    game.local_path = local_path
    
    # Save playtime offset and years
    game.playtime_offset = playtime_offset
    game.years_played = years_played
    
    # Recalculate total hours
    if hours_played != game.hours_played:
        # If user directly modified total hours, treat it as the new tracked hours base
        game.hours_played = hours_played + playtime_offset
    else:
        game.hours_played = tracked_hours + playtime_offset

    game.cover_image_url = cover_image_url
    game.animated_bg_url = animated_bg_url
    game.save()

    messages.success(request, f"Game '{game.name}' updated successfully!")
    return redirect(f'/relax/?game_id={game.id}')


@login_required
def serve_local_file(request):
    import os
    import re
    from django.http import FileResponse, HttpResponse, Http404
    from django.shortcuts import redirect
    file_path = request.GET.get('path', '').strip()
    if not file_path:
        raise Http404("Path is empty")
        
    file_path = os.path.abspath(file_path)
    
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webm', '.mp4', '.mkv', '.avi', '.mp3', '.wav']
    _, ext = os.path.splitext(file_path.lower())
    if ext not in allowed_extensions:
        raise Http404("File extension not allowed")
        
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        # Fallback to high-quality video or image placeholders
        if ext in ['.mp4', '.webm', '.mkv', '.avi']:
            return redirect("https://assets.mixkit.co/videos/preview/mixkit-glass-sphere-spinning-in-purple-neon-light-44754-large.mp4")
        else:
            if 'cover' in file_path.lower() or 'thumb' in file_path.lower():
                return redirect("https://images.unsplash.com/photo-1538481199705-c710c4e965fc?q=80&w=350&auto=format&fit=crop")
            else:
                return redirect("https://images.unsplash.com/photo-1542751371-adc38448a05e?q=80&w=800&auto=format&fit=crop")
            
    # Lookup mime type
    mime_type = "application/octet-stream"
    if ext == '.mp4':
        mime_type = "video/mp4"
    elif ext == '.webm':
        mime_type = "video/webm"
    elif ext == '.mkv':
        mime_type = "video/x-matroska"
    elif ext == '.png':
        mime_type = "image/png"
    elif ext in ['.jpg', '.jpeg']:
        mime_type = "image/jpeg"
    elif ext == '.gif':
        mime_type = "image/gif"
        
    # Support HTTP Range requests (HTTP 206) for smooth seeking and buffering of videos
    range_header = request.META.get('HTTP_RANGE', '').strip()
    if range_header and ext in ['.mp4', '.webm', '.mkv', '.avi']:
        try:
            statobj = os.stat(file_path)
            file_size = statobj.st_size
            
            range_match = re.match(r'bytes=(\d+)-(\d*)', range_header)
            if range_match:
                first_byte, last_byte = range_match.groups()
                first_byte = int(first_byte) if first_byte else 0
                last_byte = int(last_byte) if last_byte else file_size - 1
                if last_byte >= file_size:
                    last_byte = file_size - 1
                    
                length = last_byte - first_byte + 1
                
                resp = HttpResponse(status=206, content_type=mime_type)
                resp['Content-Range'] = f'bytes {first_byte}-{last_byte}/{file_size}'
                resp['Accept-Ranges'] = 'bytes'
                resp['Content-Length'] = str(length)
                
                with open(file_path, 'rb') as f:
                    f.seek(first_byte)
                    resp.content = f.read(length)
                return resp
        except Exception:
            pass
            
    return FileResponse(open(file_path, 'rb'))


@login_required
def relax_search_artwork(request):
    import urllib.request
    import urllib.parse
    import json
    import re
    term = request.GET.get('term', '').strip()
    if not term:
        return JsonResponse({'status': 'error', 'message': 'Search term is required.'}, status=400)
    
    results = []
    try:
        # Try storesearch API first
        url = f"https://store.steampowered.com/api/storesearch/?term={urllib.parse.quote(term)}&l=english&cc=US"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        if 'items' in data and data['items']:
            for item in data['items']:
                appid = item['id']
                results.append({
                    'appid': appid,
                    'name': item['name'],
                    'cover_url': f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{appid}/library_600x900_2x.jpg",
                    'hero_url': f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{appid}/library_hero.jpg",
                    'header_url': f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{appid}/header.jpg",
                })
    except Exception as e:
        logger.warning(f"Artwork search API failed: {e}")
        
    # Try suggest autocompletion endpoint if storesearch is empty or failed
    if not results:
        try:
            suggest_url = f"https://store.steampowered.com/search/suggest?term={urllib.parse.quote(term)}&f=games"
            suggest_req = urllib.request.Request(suggest_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
            with urllib.request.urlopen(suggest_req, timeout=5) as suggest_res:
                suggest_html = suggest_res.read().decode('utf-8', errors='ignore')
            
            matches = re.findall(r'href="https://store.steampowered.com/app/(\d+)/([^/"]*)/?[^"]*".*?<div class="match_name">([^<]+)</div>', suggest_html, re.DOTALL)
            for appid, slug, name in matches:
                name_clean = name.strip()
                results.append({
                    'appid': appid,
                    'name': name_clean,
                    'cover_url': f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{appid}/library_600x900_2x.jpg",
                    'hero_url': f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{appid}/library_hero.jpg",
                    'header_url': f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{appid}/header.jpg",
                })
        except Exception as suggest_err:
            logger.error(f"Artwork search suggest API fallback failed: {suggest_err}")
            
    return JsonResponse({'status': 'success', 'results': results})




def extract_style_from_html(html_content):
    if not html_content:
        return ""
    if '<' in html_content and '>' in html_content:
        try:
            parser = SimpleHTMLScraper()
            parsed_html = html_content
            if '<body' not in html_content.lower():
                parsed_html = f"<body>{html_content}</body>"
            parser.feed(parsed_html)
            content = "\n".join(parser.text_list)
            content = re.sub(r'\n\s*\n', '\n\n', content)
            return content.strip()
        except Exception as e:
            logger.error(f"HTML inline extraction failed: {e}")
    return html_content


@login_required
@require_POST
def relax_add_guide(request):
    game_id = request.POST.get('game_id')
    title = request.POST.get('title', '').strip()
    source_url = request.POST.get('source_url', '').strip()
    content_markdown = request.POST.get('content_markdown', '').strip()

    game = get_object_or_404(Game, id=game_id)

    if source_url and not content_markdown:
        scraped_title, scraped_content = scrape_guide_content(source_url)
        if not title:
            title = scraped_title
        content_markdown = scraped_content
    elif content_markdown:
        content_markdown = extract_style_from_html(content_markdown)

    if not title:
        title = "Manual Guide Entry"

    if not content_markdown:
        messages.error(request, "Guide content is empty.")
        return redirect(f"/relax/?game_id={game.id}")

    GameGuide.objects.create(
        game=game,
        title=title,
        source_url=source_url or None,
        content_markdown=content_markdown
    )
    messages.success(request, f"New guide added for {game.name}!")
    return redirect(f"/relax/?game_id={game.id}")


@login_required
@require_POST
def relax_delete_guide(request, guide_id):
    from .models import GameGuide
    guide = get_object_or_404(GameGuide, id=guide_id)
    game_id = guide.game_id
    game_name = guide.game.name
    guide.delete()
    messages.success(request, f"Guide successfully removed from {game_name}.")
    return redirect(f"/relax/?game_id={game_id}")


@login_required
@require_POST
def relax_add_video(request):
    game_id = request.POST.get('game_id')
    title = request.POST.get('title', '').strip()
    youtube_url = request.POST.get('youtube_url', '').strip()

    game = get_object_or_404(Game, id=game_id)

    if not title:
        title = "Gameplay Clip"

    if not youtube_url:
        messages.error(request, "YouTube URL is required.")
        return redirect(f"/relax/?game_id={game.id}")

    GameVideo.objects.create(
        game=game,
        title=title,
        youtube_url=youtube_url
    )
    messages.success(request, f"Gameplay video added for {game.name}!")
    return redirect(f"/relax/?game_id={game.id}")


@login_required
def relax_launch_game(request, game_id):
    game = get_object_or_404(Game, id=game_id)
    import platform
    import subprocess
    
    # 1. Local launch if Django itself runs on Windows (local single-PC development setup)
    if platform.system() == 'Windows':
        launched = False
        if game.steam_app_id:
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam")
                steam_exe, _ = winreg.QueryValueEx(key, "SteamExe")
                if steam_exe and os.path.exists(steam_exe):
                    steam_dir = os.path.dirname(steam_exe)
                    subprocess.Popen([steam_exe, "-applaunch", str(game.steam_app_id)], cwd=steam_dir)
                    launched = True
            except Exception:
                pass
            
            if not launched:
                try:
                    subprocess.Popen(["cmd.exe", "/c", "start", f"steam://rungameid/{game.steam_app_id}"], shell=True)
                    launched = True
                except Exception:
                    pass
            
            if not launched:
                try:
                    os.startfile(f"steam://rungameid/{game.steam_app_id}")
                    launched = True
                except Exception:
                    pass
        elif game.local_path:
            is_uri = ':' in game.local_path and not ('\\' in game.local_path or '/' in game.local_path)
            if is_uri or os.path.exists(game.local_path):
                try:
                    subprocess.Popen(["cmd.exe", "/c", "start", '""', game.local_path], shell=True)
                    launched = True
                except Exception:
                    pass
        
        if launched:
            messages.success(request, f"Launched '{game.name}' successfully on local host.")
        else:
            messages.error(request, f"Failed to launch '{game.name}' locally. Verify the path or Steam ID.")
        return redirect(f"/relax/?game_id={game.id}")

    # 2. Remote launch via background daemon pull queue (highly robust, firewall-proof, zero page-hangs!)
    settings = SystemSettings.load()
    if settings.gaming_rig_ip:
        # Flag this game as pending launch in queue file
        from django.conf import settings as django_settings
        queue_file = os.path.join(django_settings.BASE_DIR, 'pending_launches.json')
        launches = []
        if os.path.exists(queue_file):
            try:
                with open(queue_file, 'r') as f:
                    launches = json.load(f)
            except Exception:
                pass
        launches.append({
            'appid': game.steam_app_id or None,
            'path': game.local_path or None
        })
        try:
            with open(queue_file, 'w') as f:
                json.dump(launches, f)
        except Exception:
            pass
            
        messages.success(request, f"Remote launch command for '{game.name}' queued. Starting on your gaming PC shortly!")
        return redirect(f"/relax/?game_id={game.id}")
        
    messages.error(request, "Remote launch failed: No gaming rig IP address is configured in Settings. Go to the Settings page and enter your rig's IP.")
    return redirect(f"/relax/?game_id={game.id}")


@login_required
def relax_detect_game_path(request):
    name = request.GET.get('name', '').strip()
    if not name:
        return JsonResponse({'status': 'error', 'message': 'Game name is required'}, status=400)
        
    settings = SystemSettings.load()
    if not settings.gaming_rig_ip:
        return JsonResponse({'status': 'error', 'message': 'No gaming rig IP address is configured in Settings.'}, status=400)
        
    import urllib.request
    import urllib.parse
    import json
    
    encoded_name = urllib.parse.quote(name)
    daemon_url = f"http://{settings.gaming_rig_ip}:5555/detect?name={encoded_name}"
    
    try:
        req = urllib.request.Request(daemon_url)
        with urllib.request.urlopen(req, timeout=8) as res:
            res_data = json.loads(res.read().decode('utf-8'))
            return JsonResponse(res_data)
    except Exception as e:
        logger.error(f"Failed to connect to Steam Launcher Daemon for detection: {e}")
        return JsonResponse({
            'status': 'error', 
            'message': f"Could not connect to Steam Launcher Daemon at {settings.gaming_rig_ip}:5555. Make sure the daemon is running on your Windows rig."
        }, status=500)


def sync_steam_playtimes_helper(username):
    import xml.etree.ElementTree as ET
    import urllib.request
    import json
    import re
    import html
    from .models import Game, SystemSettings
    
    # Parse full URLs
    if 'steamcommunity.com/' in username.lower():
        username = username.rstrip('/')
        if '/profiles/' in username.lower():
            username = username.split('/profiles/')[-1].split('/')[0].strip()
        elif '/id/' in username.lower():
            username = username.split('/id/')[-1].split('/')[0].strip()
            
    updated_count = 0
    
    # 0. Try Official Steam Web API first if key is configured
    settings = SystemSettings.load()
    api_key = settings.steam_api_key
    if api_key:
        steamid = None
        if username.isdigit() and len(username) == 17:
            steamid = username
        else:
            # Resolve custom URL vanity name to ID64
            try:
                vanity_url = f"https://api.steampowered.com/ISteamUser/ResolveVanityURL/v1/?key={api_key}&vanityurl={username}"
                v_req = urllib.request.Request(vanity_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(v_req, timeout=5) as v_res:
                    vanity_data = json.loads(v_res.read().decode('utf-8'))
                    if vanity_data.get('response', {}).get('success') == 1:
                        steamid = vanity_data['response']['steamid']
            except Exception as e:
                logger.warning(f"ResolveVanityURL failed for {username}: {e}")
                
        # Fallback resolve via XML profile metadata
        if not steamid:
            try:
                profile_url = f"https://steamcommunity.com/id/{username}/?xml=1"
                p_req = urllib.request.Request(profile_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(p_req, timeout=5) as p_res:
                    xml_str = p_res.read().decode('utf-8', errors='ignore')
                    m = re.search(r'<steamID64>(\d+)</steamID64>', xml_str)
                    if m:
                        steamid = m.group(1)
            except Exception:
                pass
                
        if steamid:
            try:
                # Query GetOwnedGames Web API (unblocked full library sync)
                api_url = f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key={api_key}&steamid={steamid}&include_appinfo=1&include_played_free_games=1&format=json"
                req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=8) as res:
                    owned_data = json.loads(res.read().decode('utf-8'))
                    
                games_list = owned_data.get('response', {}).get('games', [])
                for g in games_list:
                    appid = str(g.get('appid', ''))
                    name = g.get('name', '').strip()
                    playtime_minutes = g.get('playtime_forever', 0)
                    hours = round(playtime_minutes / 60.0, 1)
                    
                    if not appid or not name:
                        continue
                        
                    matching_games = Game.objects.filter(steam_app_id=appid, is_active=True)
                    if matching_games.exists():
                        for game in matching_games:
                            game.hours_played = hours + game.playtime_offset
                            game.save()
                            updated_count += 1
                    else:
                        if hours >= 1.0:
                            cover_image = f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{appid}/library_600x900_2x.jpg"
                            hero_image = f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{appid}/library_hero.jpg"
                            Game.objects.create(
                                name=name,
                                steam_app_id=appid,
                                hours_played=hours,
                                cover_image_url=cover_image,
                                animated_bg_url=hero_image,
                                is_active=True
                            )
                            updated_count += 1
                return updated_count, True # successfully processed full list via Web API
            except Exception as api_err:
                logger.error(f"Steam Web API sync failed for user {username}: {api_err}. Falling back to scraping...")
    
    # 1. Try HTML scraping first (highly reliable for full public libraries, bypasses XML limits)
    html_url = f"https://steamcommunity.com/profiles/{username}/games/?tab=all" if (username.isdigit() and len(username) == 17) else f"https://steamcommunity.com/id/{username}/games/?tab=all"
    try:
        html_req = urllib.request.Request(html_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        with urllib.request.urlopen(html_req, timeout=8) as html_res:
            html_data = html_res.read().decode('utf-8', errors='ignore')
            
        m = re.search(r'var\s+rgGames\s*=\s*(\[.*?\])\s*;', html_data, re.DOTALL)
        if m:
            games_data = json.loads(m.group(1))
            for g in games_data:
                appid = str(g.get('appid', ''))
                name = g.get('name', '').strip()
                hours = 0.0
                hours_val = g.get('hours_forever')
                if hours_val:
                    try:
                        hours = float(str(hours_val).replace(',', '').strip())
                    except ValueError:
                        pass
                
                if not appid or not name:
                    continue
                    
                matching_games = Game.objects.filter(steam_app_id=appid, is_active=True)
                if matching_games.exists():
                    for game in matching_games:
                        game.hours_played = hours + game.playtime_offset
                        game.save()
                        updated_count += 1
                else:
                    # Auto-import new games if they have playtime!
                    if hours >= 1.0:
                        cover_image = f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{appid}/library_600x900_2x.jpg"
                        hero_image = f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{appid}/library_hero.jpg"
                        
                        Game.objects.create(
                            name=name,
                            steam_app_id=appid,
                            hours_played=hours,
                            cover_image_url=cover_image,
                            animated_bg_url=hero_image,
                            is_active=True
                        )
                        updated_count += 1
            return updated_count, True # successfully processed full list
    except Exception as html_err:
        logger.warning(f"Steam HTML sync fallback failed for user {username}: {html_err}")
        
    # 2. XML Fallback flow
    if username.isdigit() and len(username) == 17:
        url = f"https://steamcommunity.com/profiles/{username}/games/?xml=1"
    else:
        url = f"https://steamcommunity.com/id/{username}/games/?xml=1"
        
    is_full_list = True
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
            
        xml_str = xml_data.decode('utf-8', errors='ignore')
        if '<gamesList' not in xml_str and '<games' not in xml_str:
            # Fallback to main profile XML
            is_full_list = False
            if username.isdigit() and len(username) == 17:
                fallback_url = f"https://steamcommunity.com/profiles/{username}/?xml=1"
            else:
                fallback_url = f"https://steamcommunity.com/id/{username}/?xml=1"
                
            try:
                fallback_req = urllib.request.Request(fallback_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(fallback_req, timeout=10) as fallback_res:
                    fallback_xml_data = fallback_res.read()
                xml_str = fallback_xml_data.decode('utf-8', errors='ignore')
                if '<profile>' not in xml_str:
                    raise ValueError("Failed to retrieve profile XML data")
            except Exception as fe:
                raise ValueError(f"Steam blocked full games list sync, and profile fallback failed: {str(fe)}")

        # Clean HTML entities and unescaped ampersands so ElementTree doesn't crash
        def replace_entity(match):
            entity = match.group(0)
            name_entity = match.group(1)
            if name_entity in ('amp', 'lt', 'gt', 'quot', 'apos'):
                return entity
            unescaped = html.unescape(entity)
            return unescaped.replace('&', '&amp;')
 
        xml_str = re.sub(r'&([a-zA-Z0-9#]+);', replace_entity, xml_str)
        xml_str = re.sub(r'&(?!(amp|lt|gt|quot|apos);)', '&amp;', xml_str)
 
        root = ET.fromstring(xml_str.encode('utf-8'))
        
        error_node = root.find('error')
        if error_node is not None:
            raise ValueError(error_node.text)
            
        if is_full_list:
            games_list = root.findall('.//game')
            for g_node in games_list:
                appid_node = g_node.find('appID')
                name_node = g_node.find('name')
                hours_node = g_node.find('hoursOnRecord')
                
                if appid_node is not None:
                    appid = appid_node.text.strip()
                    name = name_node.text.strip() if name_node is not None and name_node.text else f"Steam Game {appid}"
                    hours = 0.0
                    if hours_node is not None and hours_node.text:
                        try:
                            hours = float(hours_node.text.replace(',', '').strip())
                        except ValueError:
                            pass
                            
                    matching_games = Game.objects.filter(steam_app_id=appid, is_active=True)
                    if matching_games.exists():
                        for game in matching_games:
                            game.hours_played = hours + game.playtime_offset
                            game.save()
                            updated_count += 1
                    else:
                        if hours >= 1.0:
                            cover_image = f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{appid}/library_600x900_2x.jpg"
                            hero_image = f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{appid}/library_hero.jpg"
                            Game.objects.create(
                                name=name,
                                steam_app_id=appid,
                                hours_played=hours,
                                cover_image_url=cover_image,
                                animated_bg_url=hero_image,
                                is_active=True
                            )
                            updated_count += 1
        else:
            # Parse mostPlayedGames from profile XML
            games_list = root.findall('.//mostPlayedGame')
            for g_node in games_list:
                name_node = g_node.find('gameName')
                hours_node = g_node.find('hoursOnRecord')
                if hours_node is None:
                    hours_node = g_node.find('hoursPlayed')
                
                appid = None
                name = name_node.text.strip() if name_node is not None and name_node.text else "Steam Game"
                
                stats_node = g_node.find('statsName')
                if stats_node is not None and stats_node.text:
                    appid = stats_node.text.strip()
                else:
                    link_node = g_node.find('gameLink')
                    if link_node is not None and link_node.text:
                        m = re.search(r'/app/(\d+)', link_node.text)
                        if m:
                            appid = m.group(1)
                            
                if appid:
                    hours = 0.0
                    if hours_node is not None and hours_node.text:
                        try:
                            hours = float(hours_node.text.replace(',', '').strip())
                        except ValueError:
                            pass
                            
                    matching_games = Game.objects.filter(steam_app_id=appid, is_active=True)
                    if matching_games.exists():
                        for game in matching_games:
                            game.hours_played = hours + game.playtime_offset
                            game.save()
                            updated_count += 1
                    else:
                        if hours >= 1.0:
                            cover_image = f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{appid}/library_600x900_2x.jpg"
                            hero_image = f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{appid}/library_hero.jpg"
                            Game.objects.create(
                                name=name,
                                steam_app_id=appid,
                                hours_played=hours,
                                cover_image_url=cover_image,
                                animated_bg_url=hero_image,
                                is_active=True
                            )
                            updated_count += 1
        return updated_count, is_full_list
    except Exception as e:
        logger.error(f"Steam playtimes XML sync failed for user {username}: {e}", exc_info=True)
        raise e


@login_required
def relax_sync_steam_playtimes(request):
    username = request.GET.get('username', '').strip()
    if not username:
        return JsonResponse({'status': 'error', 'message': 'Steam username or ID64 is required.'}, status=400)
        
    request.session['steam_username'] = username
    
    try:
        updated_count, is_full_list = sync_steam_playtimes_helper(username)
        msg = f'Successfully synced and updated/imported {updated_count} game(s) from Steam.'
        if not is_full_list:
            msg = f'Synced and updated/imported {updated_count} recently played game(s) from your Steam profile (full list was blocked by Steam).'
            
        return JsonResponse({
            'status': 'success',
            'message': msg,
            'updated_count': updated_count
        })
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Failed to sync: {str(e)}'}, status=500)


@login_required
def trader_report_view(request):
    trader, _ = PaperTrader.objects.get_or_create(id=1)
    trades = TradeLog.objects.filter(trader=trader).order_by('-timestamp')

    ZERO_DEC = Value(0, output_field=DecimalField(max_digits=20, decimal_places=2))

    sell_trades = trades.filter(action='SELL')
    buy_trades = trades.filter(action='BUY')

    total_notional_sells = sell_trades.aggregate(
        total=Coalesce(Sum('notional_value'), ZERO_DEC)
    )['total']

    total_notional_buys = buy_trades.aggregate(
        total=Coalesce(Sum('notional_value'), ZERO_DEC)
    )['total']

    gross_pnl = (total_notional_sells or 0) - (total_notional_buys or 0)

    context = {
        'trader': trader,
        'all_trades': trades,
        'total_trades': trades.count(),
        'buy_count': buy_trades.count(),
        'sell_count': sell_trades.count(),
        'gross_pnl': gross_pnl,
        'total_volume': (total_notional_buys or 0) + (total_notional_sells or 0),
    }
    return render(request, 'trader_report.html', context)


@login_required
def reset_trader_report_view(request):
    """
    Deletes all trade logs for the primary paper trader.
    """
    if request.method == 'POST':
        trader, _ = PaperTrader.objects.get_or_create(id=1)
        trades_deleted, _ = TradeLog.objects.filter(trader=trader).delete()
        messages.success(request, f"Successfully deleted {trades_deleted} trade log entries.")
    return redirect('trader_report')


# ==========================================
# GAMING LOUNGE ANALYTICS & WATCHLIST VIEWS
# ==========================================

from django.utils import timezone
from datetime import datetime, timedelta
from django.db.models import Sum, Q
import urllib.request
import json
import re
from django.core.mail import send_mail
from django.conf import settings
from django.contrib import messages
from .models import Game, GameBetaInfo, WatchlistGame, BudgetWatchlistGame, GamePlaytimeSession

@login_required
def relax_analytics_view(request):
    games = Game.objects.filter(is_active=True)
    
    # Sort games list in memory by normalized density (Hours/Year)
    games_sorted = list(games)
    games_sorted.sort(key=lambda x: x.normalized_hours_per_year, reverse=True)
    
    most_played = games_sorted[0] if games_sorted else None
    
    # Calculate favorite game: game with manual offset, fallback to most played
    fav_game = games.filter(playtime_offset__gt=0.0).first() or most_played

    # Playing time statistics
    today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
    month_start = timezone.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    playtime_today_sec = GamePlaytimeSession.objects.filter(
        start_time__gte=today_start
    ).aggregate(total=Sum('duration_seconds'))['total'] or 0
    playtime_today = round(playtime_today_sec / 3600.0, 1)

    playtime_month_sec = GamePlaytimeSession.objects.filter(
        start_time__gte=month_start
    ).aggregate(total=Sum('duration_seconds'))['total'] or 0
    playtime_month = round(playtime_month_sec / 3600.0, 1)

    # 7-day activity chart data
    chart_dates = []
    chart_hours = []
    for i in range(6, -1, -1):
        date = (timezone.now() - timedelta(days=i)).date()
        date_start = timezone.make_aware(datetime.combine(date, datetime.min.time()))
        date_end = timezone.make_aware(datetime.combine(date, datetime.max.time()))
        sec = GamePlaytimeSession.objects.filter(
            start_time__range=(date_start, date_end)
        ).aggregate(total=Sum('duration_seconds'))['total'] or 0
        chart_dates.append(date.strftime('%a'))
        chart_hours.append(round(sec / 3600.0, 2))

    # Top played games today
    today_sessions = GamePlaytimeSession.objects.filter(start_time__gte=today_start)
    game_durations = {}
    for s in today_sessions:
        game_durations[s.game] = game_durations.get(s.game, 0) + s.duration_seconds
    sorted_today = sorted(game_durations.items(), key=lambda x: x[1], reverse=True)[:3]
    top_today = [{'name': g.name, 'hours': round(sec / 3600.0, 1)} for g, sec in sorted_today]

    # Top played games this month
    month_sessions = GamePlaytimeSession.objects.filter(start_time__gte=month_start)
    game_durations_month = {}
    for s in month_sessions:
        game_durations_month[s.game] = game_durations_month.get(s.game, 0) + s.duration_seconds
    sorted_month = sorted(game_durations_month.items(), key=lambda x: x[1], reverse=True)[:3]
    top_month = [{'name': g.name, 'hours': round(sec / 3600.0, 1)} for g, sec in sorted_month]

    context = {
        'games': games,
        'most_played': most_played,
        'fav_game': fav_game,
        'playtime_today': playtime_today,
        'playtime_month': playtime_month,
        'chart_dates': json.dumps(chart_dates),
        'chart_hours': json.dumps(chart_hours),
        'top_today': top_today,
        'top_month': top_month,
    }
    return render(request, 'relax_analytics.html', context)


@login_required
def relax_game_detail_analytics_view(request, game_id):
    game = get_object_or_404(Game, id=game_id, is_active=True)
    sessions = GamePlaytimeSession.objects.filter(game=game).order_by('-start_time')[:10]
    
    # 7-day playtime chart for just this game
    chart_dates = []
    chart_hours = []
    for i in range(6, -1, -1):
        date = (timezone.now() - timedelta(days=i)).date()
        date_start = timezone.make_aware(datetime.combine(date, datetime.min.time()))
        date_end = timezone.make_aware(datetime.combine(date, datetime.max.time()))
        sec = GamePlaytimeSession.objects.filter(
            game=game,
            start_time__range=(date_start, date_end)
        ).aggregate(total=Sum('duration_seconds'))['total'] or 0
        chart_dates.append(date.strftime('%a'))
        chart_hours.append(round(sec / 3600.0, 2))

    context = {
        'game': game,
        'selected_game': game,
        'games': Game.objects.filter(is_active=True),
        'sessions': sessions,
        'chart_dates': json.dumps(chart_dates),
        'chart_hours': json.dumps(chart_hours),
    }
    return render(request, 'relax_game_analytics.html', context)


def get_game_recent_news(game_name):
    from django.core.cache import cache
    cache_key = f"game_news_{game_name.replace(' ', '_').lower()}"
    cached_news = cache.get(cache_key)
    if cached_news:
        return cached_news
        
    import urllib.request
    import urllib.parse
    import xml.etree.ElementTree as ET
    
    query = urllib.parse.quote(f"{game_name} (release OR gameplay OR trailer OR announcement OR leaks) when:30d")
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    news_list = []
    try:
        req = urllib.request.Request(rss_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as res:
            root = ET.fromstring(res.read())
            items = root.findall('.//item')
            for item in items[:4]:
                title = item.find('title').text
                # Strip source suffix if present (e.g. " - IGN")
                if " - " in title:
                    title = title.rsplit(" - ", 1)[0]
                news_list.append(title)
    except Exception:
        pass
        
    if not news_list:
        news_list = [
            f"Developer update announcement published for {game_name}.",
            f"Pre-registration milestones hit for {game_name}."
        ]
        
    cache.set(cache_key, news_list, 3600)
    return news_list


@login_required
def relax_watchlist_view(request):
    upcoming_games = WatchlistGame.objects.all().order_by('expected_release_date')
    budget_games = BudgetWatchlistGame.objects.all().order_by('name')
    
    # Auto-scout stale prices (older than 3 hours, max 2 games per visit to keep load times instant)
    from django.utils import timezone
    from datetime import timedelta
    stale_limit = timezone.now() - timedelta(hours=3)
    scouted = 0
    for bg in budget_games:
        if scouted >= 2:
            break
        if not bg.last_checked_at or bg.last_checked_at < stale_limit:
            bg.scout_price()
            scouted += 1
            
    # Re-fetch budget_games list to get the updated values
    budget_games = BudgetWatchlistGame.objects.all().order_by('name')
    
    # Simple dynamic release countdown calculation
    watchlist_data = []
    for g in upcoming_games:
        countdown = "Unknown"
        if g.expected_release_date:
            try:
                # Try parsing YYYY-MM-DD
                rel_date = datetime.strptime(g.expected_release_date, "%Y-%m-%d").date()
                delta = rel_date - timezone.now().date()
                if delta.days > 0:
                    countdown = f"{delta.days} days left"
                elif delta.days == 0:
                    countdown = "Releasing Today!"
                else:
                    countdown = "Released"
            except ValueError:
                countdown = g.expected_release_date
        
        # Load dynamic recent news
        recent_news = get_game_recent_news(g.name)
        
        watchlist_data.append({
            'game': g,
            'countdown': countdown,
            'recent_news': news_list if 'news_list' in locals() else recent_news
        })

    active_betas = GameBetaInfo.objects.filter(is_active=True).order_by('-discovered_at')
    
    settings = SystemSettings.load()
    context = {
        'watchlist': watchlist_data,
        'budget_games': budget_games,
        'active_betas': active_betas,
        'settings': settings,
        'games': Game.objects.filter(is_active=True),
    }
    return render(request, 'relax_watchlist.html', context)


@login_required
@require_POST
def relax_toggle_beta_watch(request, game_id):
    game = get_object_or_404(Game, id=game_id)
    game.watch_beta_recruitment = not game.watch_beta_recruitment
    game.save()
    return JsonResponse({
        'status': 'success',
        'game_id': game.id,
        'watch_beta_recruitment': game.watch_beta_recruitment
    })


@login_required
@require_POST
def relax_add_watchlist_game(request):
    name = request.POST.get('name', '').strip()
    release_date = request.POST.get('expected_release_date', '').strip()
    business_model = request.POST.get('business_model', 'UNKNOWN')
    price = request.POST.get('price_estimate', '').strip()
    sys_req = request.POST.get('system_requirements', '').strip()
    website = request.POST.get('official_website', '').strip()

    if name:
        game = WatchlistGame.objects.create(
            name=name,
            expected_release_date=release_date,
            business_model=business_model,
            price_estimate=price,
            system_requirements=sys_req,
            official_website=website if website else None
        )
        # Scout details from Steam Store API synchronously
        game.scout_details()
        messages.success(request, f"Added '{name}' to your upcoming games watchlist and populated specs from Steam successfully!")
    return redirect('relax_watchlist')


@login_required
def relax_delete_watchlist_game(request, game_id):
    g = get_object_or_404(WatchlistGame, id=game_id)
    name = g.name
    g.delete()
    messages.success(request, f"Removed '{name}' from watchlist.")
    return redirect('relax_watchlist')


@login_required
@require_POST
def relax_add_budget_game(request):
    name = request.POST.get('name', '').strip()
    steam_app_id = request.POST.get('steam_app_id', '').strip() or None
    
    settings = SystemSettings.load()
    target_budget_str = request.POST.get('target_budget', '').strip()
    if target_budget_str:
        target_budget = float(target_budget_str)
    else:
        target_budget = settings.global_wishlist_budget
    
    check_steam = 'check_steam' in request.POST
    check_epic = 'check_epic' in request.POST
    check_xbox = 'check_xbox' in request.POST

    if name:
        game = BudgetWatchlistGame.objects.create(
            name=name,
            steam_app_id=steam_app_id,
            target_budget=target_budget,
            check_steam=check_steam,
            check_epic=check_epic,
            check_xbox=check_xbox
        )
        # Scout price synchronously immediately so the user doesn't see "N/A" on creation
        game.scout_price()
        messages.success(request, f"Added '{name}' to your budget watchlist and scouted live prices successfully!")
    return redirect('relax_watchlist')


@login_required
def relax_watchlist_refresh(request):
    watchlist = BudgetWatchlistGame.objects.all()
    success_count = 0
    for item in watchlist:
        if item.scout_price():
            success_count += 1
    messages.success(request, f"Refreshed discount scout prices for {success_count} of {watchlist.count()} watchlist games!")
    return redirect('relax_watchlist')


@login_required
@require_POST
def relax_update_global_budget(request):
    settings = SystemSettings.load()
    global_budget = float(request.POST.get('global_wishlist_budget', 1000.0) or 1000.0)
    settings.global_wishlist_budget = global_budget
    settings.save()
    
    if 'update_existing' in request.POST:
        BudgetWatchlistGame.objects.all().update(target_budget=global_budget)
        messages.success(request, f"Updated global default budget to ₹{global_budget:.2f} and updated all existing items!")
    else:
        messages.success(request, f"Updated global default target budget to ₹{global_budget:.2f} for future games.")
        
    return redirect('relax_watchlist')


@login_required
def relax_delete_budget_game(request, game_id):
    g = get_object_or_404(BudgetWatchlistGame, id=game_id)
    name = g.name
    g.delete()
    messages.success(request, f"Removed '{name}' from budget watchlist.")
    return redirect('relax_watchlist')


@login_required
def relax_immersion_view(request):
    active_session = GamePlaytimeSession.objects.filter(is_active=True).first()
    context = {
        'active_session': active_session,
    }
    return render(request, 'relax_immersion.html', context)


@login_required
def relax_api_immersion_status(request):
    from django.core.cache import cache
    active_session = GamePlaytimeSession.objects.filter(is_active=True).first()
    if not active_session:
        return JsonResponse({'active': False})

    # If the session has not received a daemon heartbeat in the last 7 seconds,
    # it means the game process has stopped on the client rig. Close the HUD immediately!
    last_active = cache.get(f"game_session_active_{active_session.id}")
    if not last_active or (timezone.now() - last_active).total_seconds() > 7:
        return JsonResponse({'active': False})

    elapsed = int((timezone.now() - active_session.start_time).total_seconds())
    return JsonResponse({
        'active': True,
        'session_id': active_session.id,
        'game_name': active_session.game.name,
        'cover_image_url': active_session.game.cover_image_url or '',
        'animated_bg_url': active_session.game.animated_bg_url or '',
        'start_time': active_session.start_time.isoformat(),
        'limit_minutes': active_session.limit_minutes,
        'elapsed_seconds': elapsed
    })


@login_required
@require_POST
def relax_api_start_timer(request):
    import json
    data = json.loads(request.body)
    minutes = int(data.get('minutes', 0))
    
    active_session = GamePlaytimeSession.objects.filter(is_active=True).first()
    if active_session:
        active_session.limit_minutes = minutes
        active_session.save()
        return JsonResponse({'status': 'ok', 'limit_minutes': minutes})
    return JsonResponse({'status': 'error', 'message': 'No active session'}, status=400)


@login_required
@require_POST
def relax_api_stop_session(request):
    from django.core.cache import cache
    active_sessions = GamePlaytimeSession.objects.filter(is_active=True)
    for session in active_sessions:
        session.is_active = False
        session.end_time = timezone.now()
        duration = (session.end_time - session.start_time).total_seconds()
        session.duration_seconds = int(duration)
        session.save()

        # Update game total hours
        game = session.game
        game.hours_played += duration / 3600.0
        game.save()
        
        # Prevent auto-tracking this game again until it goes inactive on the client
        cache.set(f"ignore_game_tracking_{game.id}", True, 7200)
        
    return JsonResponse({'status': 'ok'})


from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def relax_api_process_heartbeat(request):
    import json
    import os
    from .models import SystemSettings, Game
    from django.core.cache import cache
    from django.utils import timezone
    
    # Auto-register gaming rig IP address based on incoming daemon heartbeat requests
    client_ip = request.META.get('HTTP_X_FORWARDED_FOR', request.META.get('REMOTE_ADDR'))
    if client_ip:
        if ',' in client_ip:
            client_ip = client_ip.split(',')[0].strip()
        settings = SystemSettings.load()
        if settings.gaming_rig_ip != client_ip:
            settings.gaming_rig_ip = client_ip
            settings.save()

    # Extract any pending launch triggers from file queue
    pending_launches = []
    from django.conf import settings as django_settings
    queue_file = os.path.join(django_settings.BASE_DIR, 'pending_launches.json')
    if os.path.exists(queue_file):
        try:
            with open(queue_file, 'r') as f:
                pending_launches = json.load(f)
            os.remove(queue_file)
        except Exception:
            pass

    active_games = Game.objects.filter(is_active=True)
    
    # Always build the list of monitored games so both GET and POST requests have it
    monitored_games = []
    for g in active_games:
        exes = []
        if g.local_path:
            exe = os.path.basename(g.local_path).lower()
            if exe.endswith('.exe'):
                exes.append(exe)
        clean_name = g.name.replace(' ', '').lower() + '.exe'
        exes.append(clean_name)
        
        # Add common aliases for major games
        g_lower = g.name.lower()
        if 'wuthering' in g_lower or 'wuwa' in g_lower:
            exes.extend(['openverseclient.exe', 'client.exe', 'wuwa.exe'])
        elif 'genshin' in g_lower:
            exes.extend(['genshinimpact.exe', 'genshin.exe'])
        elif 'neverness' in g_lower or 'nte' in g_lower:
            exes.extend(['nte.exe', 'nevernesstoeverness.exe'])
            
        monitored_games.append({
            'name': g.name,
            'executables': list(set(exes))
        })

    if request.method == 'GET':
        response_data = {'monitored_games': monitored_games}
        if pending_launches:
            response_data['pending_launches'] = pending_launches
        return JsonResponse(response_data)

    data = json.loads(request.body.decode('utf-8'))
    process_name = data.get('active_process', '').strip()
    path = data.get('path', '').strip()
    is_running = data.get('is_running', False)
    steam_app_id = data.get('steam_app_id')
    
    if is_running and (process_name or steam_app_id):
        game = None
        
        # 1. Match by Steam App ID if provided in heartbeat
        if steam_app_id:
            game = active_games.filter(steam_app_id=str(steam_app_id)).first()
            if not game:
                # Auto-discover and create game details from Steam Store API!
                try:
                    name, cover, bg = fetch_steam_details(steam_app_id)
                    if name:
                        game = Game.objects.create(
                            name=name,
                            steam_app_id=str(steam_app_id),
                            cover_image_url=cover,
                            animated_bg_url=bg,
                            hours_played=0.0
                        )
                except Exception as discover_err:
                    logger.error(f"Auto-discovery failed for Steam AppID {steam_app_id}: {discover_err}")
                    
        # 2. Match by process name/path if not resolved by Steam ID
        if not game and process_name:
            process_name_clean = process_name.replace('.exe', '').lower()
            process_name_normalized = " ".join(process_name_clean.split())
            
            # Match by space/dash insensitive alphanumeric title (e.g. "cyberpunk2077" matches "Cyberpunk 2077")
            def normalize_name(n):
                import re
                return re.sub(r'[^a-zA-Z0-9]', '', n.lower().strip()) if n else ""
                
            process_normalized = normalize_name(process_name_clean)
            for g in active_games:
                if normalize_name(g.name) == process_normalized:
                    game = g
                    break
                    
            # Check specific process mapping aliases (e.g. openverseclient -> Wuthering Waves)
            if not game:
                if 'openverseclient' in process_name_clean or 'wuwa' in process_name_clean or 'client' in process_name_clean or 'wuthering' in process_name_clean:
                    for g in active_games:
                        g_norm = g.name.lower()
                        if 'wuthering' in g_norm or 'wuwa' in g_norm:
                            game = g
                            break
                elif 'genshin' in process_name_clean:
                    for g in active_games:
                        g_norm = g.name.lower()
                        if 'genshin' in g_norm:
                            game = g
                            break
                elif 'neverness' in process_name_clean or 'nte' in process_name_clean:
                    for g in active_games:
                        g_norm = g.name.lower()
                        if 'neverness' in g_norm or 'nte' in g_norm:
                            game = g
                            break
    
            # Match by path
            if not game and path:
                game = active_games.filter(local_path=path).first()
                
            # Fallback default lookup
            if not game:
                game = active_games.filter(name__iexact=process_name_clean).first()
                
            # Create new game entry if completely missing
            if not game:
                # Capitalize each word nicely for the new entry name
                new_game_name = " ".join([w.capitalize() for w in process_name_clean.replace('_', ' ').split()])
                game = Game.objects.create(
                    name=new_game_name,
                    local_path=path,
                    hours_played=0.0
                )
            
        # If this game is currently ignored (user stopped session manually but game is still running), skip tracking updates
        if game and cache.get(f"ignore_game_tracking_{game.id}"):
            response_data = {
                'status': 'ignored', 
                'game_id': game.id, 
                'game_name': game.name,
                'monitored_games': monitored_games
            }
            if pending_launches:
                response_data['pending_launches'] = pending_launches
            return JsonResponse(response_data)
            
        # Self-cleaning: clear ignore flags for other games
        for g in active_games:
            if game and g.id == game.id:
                continue
            cache.delete(f"ignore_game_tracking_{g.id}")
            
        session = GamePlaytimeSession.objects.filter(game=game, is_active=True).first()
        if not session:
            # Close other active sessions
            GamePlaytimeSession.objects.filter(is_active=True).update(
                is_active=False,
                end_time=timezone.now()
            )
            session = GamePlaytimeSession.objects.create(game=game, is_active=True)
            
        # Update last active timestamp in cache
        cache.set(f"game_session_active_{session.id}", timezone.now(), 600)  # 10 minute cache TTL
            
        response_data = {
            'status': 'active', 
            'game_id': game.id, 
            'game_name': game.name,
            'monitored_games': monitored_games
        }
        if pending_launches:
            response_data['pending_launches'] = pending_launches
        return JsonResponse(response_data)
    else:
        # Clear ignore flags for all games since nothing is running now
        for g in active_games:
            cache.delete(f"ignore_game_tracking_{g.id}")
            
        # Close any active session, but only if they have been inactive for more than 180 seconds (3 mins grace period)
        active_sessions = GamePlaytimeSession.objects.filter(is_active=True)
        for session in active_sessions:
            last_active = cache.get(f"game_session_active_{session.id}")
            if last_active:
                inactive_duration = (timezone.now() - last_active).total_seconds()
                if inactive_duration < 180:
                    # Still in grace period, keep session active
                    continue
            
            # Exceeded grace period, close session
            session.is_active = False
            session.end_time = timezone.now()
            duration = (session.end_time - session.start_time).total_seconds()
            session.duration_seconds = int(duration)
            session.save()
            
            game = session.game
            game.hours_played += duration / 3600.0
            game.save()
            
            # Clean up cache
            cache.delete(f"game_session_active_{session.id}")
            
        response_data = {
            'status': 'inactive',
            'monitored_games': monitored_games
        }
        if pending_launches:
            response_data['pending_launches'] = pending_launches
        return JsonResponse(response_data)


@login_required
def relax_watchlist_game_detail(request, game_id):
    game = get_object_or_404(WatchlistGame, id=game_id)
    
    # Calculate countdown
    countdown = "Unknown"
    countdown_days = None
    if game.expected_release_date:
        try:
            rel_date = datetime.strptime(game.expected_release_date, "%Y-%m-%d").date()
            delta = rel_date - timezone.now().date()
            countdown_days = delta.days
            if delta.days > 0:
                countdown = f"{delta.days} days left"
            elif delta.days == 0:
                countdown = "Releasing Today!"
            else:
                countdown = "Released"
        except ValueError:
            countdown = game.expected_release_date

    # Dynamic recent news from search
    recent_news = get_game_recent_news(game.name)

    context = {
        'game': game,
        'countdown': countdown,
        'countdown_days': countdown_days,
        'recent_news': recent_news,
        'page_title': f"{game.name} - Watchlist Details",
        'games': Game.objects.filter(is_active=True),
    }
    return render(request, 'relax_watchlist_detail.html', context)


