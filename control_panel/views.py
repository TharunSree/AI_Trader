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
import yfinance as yf

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
    
    return {
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

@login_required
def dashboard_view(request):
    return render(request, 'dashboard.html', _build_dashboard_context())

@login_required
def onboarding_view(request):
    return render(request, 'guide.html', _build_dashboard_context())

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
    Renders the Intelligence Vault listing all persisted EOD reports with timeframe filters.
    """
    from .models import TradingReport
    from django.utils import timezone
    import datetime
    import calendar
    
    selected_month = request.GET.get('month', '')
    selected_week = request.GET.get('week_of_month', 'all')
    
    reports = TradingReport.objects.all()
    
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
        except Exception as e:
            pass
            
    reports = reports.order_by('-timestamp')
    
    context = {
        'reports': reports,
        'selected_month': selected_month,
        'selected_week': selected_week,
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
            messages.success(request, "A/B Swap Recommendation dismissed.")
        else:
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
            [sys.executable, str(Path("src") / "core" / "code_rewriter.py")],
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
        
        # Fail all other testing variants
        ModelVariant.objects.filter(status='TESTING').exclude(id=variant_id).update(
            status='FAILED',
            error_message='Terminated: another variant was promoted'
        )
        
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
        
        # Create success alert
        SystemAlert.objects.create(
            level='INFO',
            title=f'🧬 Variant #{variant.id} Promoted to Production',
            message=f"'{variant.name}' is now the active trading model. {restarted} trader(s) restarted.",
            related_model_reference=str(variant.id),
        )
        
        logger.info(f"[EVOLUTION] Variant #{variant_id} promoted. {restarted} traders restarted.")
        
        return JsonResponse({
            'status': 'success',
            'message': f"Variant #{variant_id} promoted to production. {restarted} trader(s) restarted with evolved model."
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
    """POST: Reject a variant — mark as FAILED and stop its virtual engine."""
    from .models import ModelVariant
    
    variant = ModelVariant.objects.filter(id=variant_id).first()
    if not variant:
        return JsonResponse({'status': 'error', 'message': 'Variant not found'}, status=404)
    
    variant.status = 'FAILED'
    variant.error_message = 'Manually rejected from dashboard'
    variant.save(update_fields=['status', 'error_message'])
    
    # Try to kill the virtual engine process
    if variant.celery_task_id:
        try:
            import signal
            os.kill(int(variant.celery_task_id), signal.SIGTERM)
        except Exception:
            pass
    
    return JsonResponse({'status': 'success', 'message': f'Variant #{variant_id} rejected.'})


@login_required
@require_POST
def evolution_delete_api(request, variant_id):
    """POST: Delete a variant, stop its virtual engine process, clean up logs, and delete DB record."""
    from django.shortcuts import get_object_or_404, redirect
    from .models import ModelVariant
    import os
    import signal
    
    variant = get_object_or_404(ModelVariant, id=variant_id)
    
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
        result = subprocess.run(['git', 'rev-list', 'HEAD...origin/master', '--count'], capture_output=True, text=True, check=False)
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
