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
    popen_kwargs = {
        'cwd': str(Path(__file__).parent.parent),
        'stdout': log_handle,
        'stderr': subprocess.STDOUT,
        'stdin': subprocess.DEVNULL,
        'bufsize': 1, # Line buffered
        'universal_newlines': True
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
    if not psutil:
        return None

    try:
        process = psutil.Process(int(pid_value))
        return round(process.memory_info().rss / (1024 * 1024), 1)
    except (psutil.Error, TypeError, ValueError):
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

    try:
        import psutil
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
        }
    except Exception:
        return {
            'available': False,
            'system_used_percent': None,
            'system_used_gb': None,
            'system_total_gb': None,
            'trader_limit_mb': int(os.getenv('PAPER_TRADER_MEMORY_LIMIT_MB', '2048')),
            'threshold_percent': threshold_percent,
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
        'live_net_profit': float(sell_notional - buy_notional),
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
    if not snapshot['available']:
        return snapshot

    running_traders = list(PaperTrader.objects.filter(status='RUNNING').order_by('-id'))
    total_runner_mb = 0
    for trader in running_traders:
        total_runner_mb += _get_process_memory_mb(trader.celery_task_id) or 0

    snapshot['running_trader_memory_mb'] = round(total_runner_mb, 1)
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
    try:
        from .models import BrokerAccount
        first_account = BrokerAccount.objects.first()
        broker = Broker(account=first_account)
        live_equity = broker.get_equity()
        buying_power = broker.get_buying_power()
        positions = broker.get_positions()
        clock_data = broker.get_market_clock()
    except Exception as e:
        logger.warning(f"Could not connect to broker for dashboard: {e}")
        live_equity = 100000.00
        buying_power = 100000.00
        positions = []
        clock_data = None

    recent_jobs = TrainingJob.objects.filter(status='COMPLETED').order_by('-id')[:5]
    active_meta = MetaTrainingJob.objects.exclude(status__in=['COMPLETED', 'FAILED', 'STOPPED']).order_by('-id').first()
    active_training = TrainingJob.objects.exclude(status__in=['COMPLETED', 'FAILED', 'STOPPED']).order_by('-id').first()
    trader = PaperTrader.objects.filter(status='RUNNING').first() or PaperTrader.objects.first()

    # Advanced Tracing Metrics for Live Nodes
    running_traders = PaperTrader.objects.prefetch_related('trades').filter(status='RUNNING')
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

    # Compute Net PNL of the active bots including unrealized inventory
    # Assuming standard account baseline of 200k or 100k
    assumed_base = 200000.0 if float(live_equity) > 150000 else 100000.0
    active_profit_made = float(live_equity) - assumed_base
    
    active_bots_count = PaperTrader.objects.filter(status='RUNNING').count()
    ready_models_count = TrainingJob.objects.filter(is_live_trading_ready=True).count()
    
    return {
        'recent_jobs': recent_jobs,
        'live_equity': f"{live_equity:,.2f}",
        'buying_power': f"{buying_power:,.2f}",
        'active_positions': positions,
        'active_starting_limit': active_starting_limit,
        'active_amount_spent': active_amount_spent,
        'active_amount_recovered': active_amount_recovered,
        'active_profit_made': active_profit_made,
        'active_bots_count': active_bots_count,
        'ready_models_count': ready_models_count,
        'system_settings': SystemSettings.load(),
        'system_alerts': __import__('control_panel.models').models.SystemAlert.objects.filter(is_read=False).order_by('-timestamp')[:5],
        'active_meta': active_meta,
        'active_training': active_training,
        'trader': trader,
        'clock': clock_data,
        'dashboard_websocket_enabled': bool(getattr(django_settings, 'HAS_DAPHNE', False)),
        'dashboard_boot_payload': build_dashboard_boot_payload(
            live_equity=live_equity,
            buying_power=buying_power,
            positions=positions,
            clock_data=clock_data,
            active_meta=active_meta,
            active_training=active_training,
            trader=trader,
        ),
    }

class JarvisLoginView(LoginView):
    template_name = 'login.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update(_build_dashboard_context())
        return context

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
        # Save sensitive keys to .env file
        write_env_value("API_KEY", request.POST.get('alpaca_api_key', ''))
        write_env_value("SECRET_KEY", request.POST.get('alpaca_secret_key', ''))
        write_env_value("BASE_URL", request.POST.get('broker_endpoint', ''))

        # Save non-sensitive settings to the database
        settings = SystemSettings.load()
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
        'api_key': read_env_value("API_KEY"),
        'secret_key': read_env_value("SECRET_KEY"),
        'base_url': read_env_value("BASE_URL"),
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
    try:
        from src.core.code_rewriter import orchestrate_rewrite
        import threading
        # Run it in a background thread so the UI doesn't hang!
        t = threading.Thread(target=orchestrate_rewrite, daemon=True)
        t.start()
        return JsonResponse({"status": "SUCCESS", "message": "Neural Sandbox Mutator Launched. Check logs!"})
    except Exception as e:
        return JsonResponse({"status": "FAILED", "message": str(e)})

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
                
        enriched_models.append({
            'reference': ref,
            'label': label,
            'source': source.upper(),
            'size_mb': size_mb,
            'created_at': created_at_fmt,
            'evaluations': eval_count,
            'is_ready': is_ready
        })
        
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
        
        if initial_cash:
            trader.initial_cash = float(initial_cash)
        
        if goal_amount:
            trader.goal_amount = float(goal_amount)
            
        if account_id:
            from .models import BrokerAccount
            acc = BrokerAccount.objects.filter(id=account_id).first()
            if acc:
                trader.account = acc
                
        trader.save()
    return redirect(request.META.get('HTTP_REFERER', 'papertrading'))

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
    def event_stream():
        yield "data: [SYSTEM] Initiating Override Protocol: Git Pull\n\n"
        try:
            # 1. GIT PULL
            process = subprocess.Popen(
                ['git', 'pull', 'origin', 'master'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Read line by line with a heartbeat fallback
            import time as _time
            last_heartbeat = _time.time()

            while True:
                line = process.stdout.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    # Send a heartbeat every 5 seconds of silence to keep proxies alive
                    if _time.time() - last_heartbeat > 5:
                        yield ": ping\n\n"
                        last_heartbeat = _time.time()
                    _time.sleep(0.1) # Small sleep to prevent CPU spiking
                    continue
                yield f"data: {line}\n\n"
                last_heartbeat = _time.time()

            process.wait()
            if process.returncode != 0:
                yield f"data: [ERROR] Git pull failed with code {process.returncode}\n\n"
                yield "event: error\ndata: \n\n"
                return

            yield "data: [SYSTEM] Synchronizing Database Schema...\n\n"
            
            # 2. RUN MIGRATIONS
            mig_process = subprocess.Popen(
                [sys.executable, 'manage.py', 'migrate'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            last_heartbeat = _time.time()
            while True:
                line = mig_process.stdout.readline()
                if not line:
                    if mig_process.poll() is not None:
                        break
                    if _time.time() - last_heartbeat > 5:
                        yield ": ping\n\n"
                        last_heartbeat = _time.time()
                    _time.sleep(0.1)
                    continue
                yield f"data: {line}\n\n"
                last_heartbeat = _time.time()

            mig_process.wait()
            if mig_process.returncode != 0:
                yield f"data: [ERROR] Migration failed with code {mig_process.returncode}\n\n"
                yield "event: error\ndata: \n\n"
                return

            yield "data: [SYSTEM] Update Successful. Reloading platform...\n\n"
            yield "event: complete\ndata: \n\n"
        except Exception as e:
            yield f"data: [CRITICAL ERROR] {str(e)}\n\n"
            yield "event: error\ndata: \n\n"

    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'  # Disable NGINX buffering
    return response

def security_status_api(request):
    settings = SystemSettings.load()
    return JsonResponse({
        'status': 'success',
        'has_password': bool(settings.lockscreen_password),
        'idle_lock_minutes': settings.idle_lock_minutes,
        'idle_logout_minutes': settings.idle_logout_minutes
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
