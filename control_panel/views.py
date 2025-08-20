# control_panel/views.py
from celery.result import AsyncResult
from decimal import Decimal
import time
import logging
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
import json
from django.db.models import Sum, Count, Avg, Value, DecimalField
from django.db.models.functions import Coalesce
from src.execution.broker import Broker
from src.strategies import STRATEGY_PLAYBOOK
from .env_manager import write_env_value, read_env_value
from .models import TrainingJob, MetaTrainingJob, PaperTrader, EvaluationJob, SystemSettings, TradeLog
from pathlib import Path
from .tasks import run_training_job_task, stop_celery_task, run_meta_trainer_task, run_paper_trader_task, \
    run_evaluation_task
# Replace the existing decimal import line with:
from decimal import Decimal, InvalidOperation

# Add (if not already present):
from collections import defaultdict
import yfinance as yf

logger = logging.getLogger(__name__)


@login_required
def dashboard_view(request):
    """
    The view for the main dashboard page.
    The context is now minimal, as data is loaded via API.
    """
    # We can still pass in recent jobs for the initial page load
    recent_jobs = TrainingJob.objects.filter(status='COMPLETED').order_by('-id')[:5]

    context = {
        'recent_jobs': recent_jobs
    }
    return render(request, 'dashboard.html', context)


@login_required
def training_view(request):
    if request.method == 'POST':
        TrainingJob.objects.create(
            feature_set_key=request.POST.get('feature_set_key'),
            hyperparameter_key=request.POST.get('hyperparameter_key'),
            window_size=request.POST.get('window_size'),
            initial_cash=request.POST.get('initial_cash', 100000)
        )
        return redirect('training')

    all_jobs = TrainingJob.objects.all().order_by('-id')
    meta_jobs = MetaTrainingJob.objects.all().order_by('-id')
    context = {
        'training_jobs': all_jobs,
        'meta_training_jobs': meta_jobs,
        'playbook': STRATEGY_PLAYBOOK,
    }
    return render(request, 'training.html', context)


@login_required
def start_training_job_view(request):
    if request.method == 'POST':
        # 1. Create the job object in the database
        job = TrainingJob.objects.create(
            feature_set_key=request.POST.get('feature_set_key'),
            hyperparameter_key=request.POST.get('hyperparameter_key'),
            window_size=request.POST.get('window_size'),
            initial_cash=request.POST.get('initial_cash', 100000)
        )
        # 2. CRUCIAL FIX: Send the job to the Celery worker
        run_training_job_task.delay(job.id)

    return redirect('training')


@login_required
def stop_job_view(request, job_id):
    if request.method == 'POST':
        job = get_object_or_404(TrainingJob, id=job_id)
        if job.celery_task_id:
            stop_celery_task.delay(job.celery_task_id)
            job.status = 'STOPPED'
            job.save()
    return redirect('training')


@login_required
def start_meta_job_view(request):
    if request.method == 'POST':
        meta_job = MetaTrainingJob.objects.create(
            initial_cash=request.POST.get('initial_cash'),
            target_equity=request.POST.get('target_equity'),
            status='PENDING'
        )
        run_meta_trainer_task.delay(meta_job.id)
    return redirect('training')


def _get_available_models():
    model_dir = Path("saved_models")
    if not model_dir.exists():
        return []
    return [f.name for f in model_dir.iterdir() if f.suffix == '.pth']


# --- NEW: Views to start and stop the trader ---
@login_required
def papertrading_view(request):
    """
    Displays the main paper trading page and its current state.
    """
    trader, _ = PaperTrader.objects.get_or_create(id=1)
    saved_models_path = Path("saved_models")
    model_files = [p.name for p in saved_models_path.glob("*.pth")] if saved_models_path.exists() else []

    context = {
        'trader': trader,
        'trader_status': trader.status,
        'model_files': model_files
    }
    return render(request, 'papertrading.html', context)


@login_required
def start_trader_view(request):
    """
    Handles the POST request to start the paper trader.
    """
    if request.method == 'POST':
        model_file = request.POST.get('model_file')
        # --- REMOVED: No longer need to get initial_cash from the form ---

        trader, _ = PaperTrader.objects.get_or_create(id=1)
        trader.error_message = ''

        if trader.status in ('STOPPED', 'FAILED') and model_file:
            # --- MODIFIED: The task call is now simpler ---
            task = run_paper_trader_task.delay(trader.id, model_file)

            trader.status = 'RUNNING'
            trader.model_file = model_file
            # No need to save initial_cash anymore
            trader.celery_task_id = task.id
            trader.save()

    return redirect('papertrading')


@login_required
def stop_trader_view(request):
    if request.method == 'POST':
        trader = get_object_or_404(PaperTrader, id=1)

        try:
            broker = Broker()
            positions = broker.get_positions()
            if positions:
                logger.info(f"Stopping trader. Liquidating {len(positions)} open positions.")
                for position in positions:
                    try:
                        broker.api.close_position(position.symbol)
                        time.sleep(1)  # API rate limiting
                    except Exception as e:
                        logger.error(f"Could not close position {position.symbol}: {e}")
                messages.info(request, f"Trader stopped and attempted to close {len(positions)} positions.")
            else:
                messages.info(request, "Trader stopped. No open positions to close.")
        except Exception as e:
            logger.error(f"Failed to liquidate positions on stop: {e}")
            messages.error(request, "Trader stopped, but failed to connect to broker to liquidate positions.")

        if trader.celery_task_id:
            stop_celery_task.delay(trader.celery_task_id)

        trader.status = 'STOPPED'
        trader.celery_task_id = None
        trader.save()

    return redirect('papertrading')


@login_required
def realtrading_view(request):
    # This view is structurally the same as the paper trading one

    # In a real app, you might have a separate model for the real trader,
    # but for now, we can manage its state with a different ID or flag.
    context = {
        'model_files': _get_available_models(),
    }
    return render(request, 'realtrading.html', context)


@login_required
def start_real_trader_view(request):
    # WARNING: This would start a REAL MONEY trading bot.
    # We will leave this placeholder for now.
    # The logic would be similar to start_trader_view but would
    # configure the TradingSession to use live API keys and endpoints.
    messages.warning(request, "Real trading start function is not yet implemented.")
    return redirect('realtrading')


@login_required
def stop_real_trader_view(request):
    messages.info(request, "Real trading stop function is not yet implemented.")
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
    context = {
        'api_key': read_env_value("API_KEY"),
        'secret_key': read_env_value("SECRET_KEY"),
        'base_url': read_env_value("BASE_URL"),
        'current_currency': settings.display_currency,
    }
    return render(request, 'settings.html', context)


@login_required
def evaluation_view(request):
    if request.method == 'POST':
        job = EvaluationJob.objects.create(
            model_file=request.POST.get('model_file'),
            start_date=request.POST.get('start_date'),
            end_date=request.POST.get('end_date'),
            status='PENDING'
        )
        run_evaluation_task.delay(job.id)
        return redirect('evaluation')

    # For the GET request, show the form and the list of past jobs
    context = {
        'model_files': _get_available_models(),
        'evaluation_jobs': EvaluationJob.objects.all().order_by('-id')
    }
    return render(request, 'evaluation.html', context)


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
    training_jobs = TrainingJob.objects.all()
    meta_jobs = MetaTrainingJob.objects.all()

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
        ]
    }
    return JsonResponse(data)


def trader_status_api(request):
    trader_model, _ = PaperTrader.objects.get_or_create(id=1)

    def _to_float(v):
        from decimal import Decimal, InvalidOperation
        try:
            if isinstance(v, Decimal):
                return float(v)
            return float(str(v))
        except (ValueError, TypeError, InvalidOperation):
            return 0.0

    try:
        broker = Broker()
        equity = _to_float(broker.get_equity())
        buying_power = _to_float(broker.get_buying_power())
        positions_raw = broker.get_positions()
        positions = []
        for p in positions_raw:
            positions.append({
                "symbol": getattr(p, 'symbol', ''),
                "qty": _to_float(getattr(p, 'qty', 0)),
                "market_value": _to_float(getattr(p, 'market_value', 0)),
                "unrealized_pl": _to_float(getattr(p, 'unrealized_pl', 0)),
            })
        positions.sort(key=lambda x: x['market_value'], reverse=True)
        return JsonResponse({
            "status": trader_model.status,
            "model_file": trader_model.model_file,
            "error_message": trader_model.error_message or "",
            "equity": equity,
            "buying_power": buying_power,
            "positions": positions
        })
    except Exception as e:
        msg = str(e)
        if "authorization failed" in msg.lower() or "alpaca authorization failed" in msg.lower():
            trader_model.status = 'FAILED'
            trader_model.error_message = "Alpaca auth failed. Check keys & endpoint."
            trader_model.save(update_fields=['status', 'error_message'])
        else:
            trader_model.status = 'FAILED'
            trader_model.error_message = msg
            trader_model.save(update_fields=['status', 'error_message'])
        return JsonResponse({
            "status": "FAILED",
            "error_message": trader_model.error_message,
            "equity": 0.0,
            "buying_power": 0.0,
            "positions": []
        }, status=200)


@login_required
def stop_meta_job_view(request, job_id):
    if request.method == 'POST':
        job = get_object_or_404(MetaTrainingJob, id=job_id)
        if job.celery_task_id:
            stop_celery_task.delay(job.celery_task_id)
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
def trader_report_view(request):
    trader, _ = PaperTrader.objects.get_or_create(id=1)
    trades = TradeLog.objects.filter(trader=trader).order_by('-timestamp')

    # Define a decimal zero matching your notional_value field precision
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
