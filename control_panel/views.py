# control_panel/views.py
from celery.result import AsyncResult
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
import json

from src.execution.broker import Broker
from .env_manager import write_env_value, read_env_value
from .models import TrainingJob, MetaTrainingJob, PaperTrader, EvaluationJob, SystemSettings
from pathlib import Path
from .tasks import run_training_job_task, stop_celery_task, run_meta_trainer_task, run_paper_trader_task, \
    run_evaluation_task


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
        # Create a new job from the form submission
        job = TrainingJob.objects.create(
            training_mode=request.POST.get('training_mode'),
            initial_cash=request.POST.get('initial_cash', 100000),
            num_episodes=request.POST.get('num_episodes', 200),
            target_equity=request.POST.get('target_equity', 200000)
        )
        run_training_job_task.delay(job.id)
        return redirect('training')

    all_jobs = TrainingJob.objects.all().order_by('-id')
    meta_jobs = MetaTrainingJob.objects.all().order_by('-id')
    context = {'training_jobs': all_jobs,
               'meta_training_jobs': meta_jobs, }
    return render(request, 'training.html', context)


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
    model_files = _get_available_models()
    trader, created = PaperTrader.objects.get_or_create(id=1)
    context = {'model_files': model_files, 'trader_status': trader.status}
    return render(request, 'papertrading.html', context)


@login_required
def start_trader_view(request):
    if request.method == 'POST':
        model_file = request.POST.get('model_file')
        trader, created = PaperTrader.objects.get_or_create(id=1)

        if trader.status == 'STOPPED' and model_file:
            task = run_paper_trader_task.delay(trader.id, model_file)
            trader.status = 'RUNNING'
            trader.model_file = model_file
            trader.celery_task_id = task.id
            trader.save()
    return redirect('papertrading')


@login_required
def stop_trader_view(request):
    if request.method == 'POST':
        trader = get_object_or_404(PaperTrader, id=1)
        if trader.celery_task_id:
            stop_celery_task.delay(trader.celery_task_id)
            trader.status = 'STOPPED'  # The task will also set this, but we do it here for immediate feedback
            trader.celery_task_id = ''
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


@login_required
def job_status_api(request):
    training_jobs = TrainingJob.objects.all()
    meta_jobs = MetaTrainingJob.objects.all()

    # Prepare the data in a simple format
    data = {
        'training_jobs': [
            {
                'id': job.id,
                'status': job.status,
                'progress': job.progress,
                'best_reward': round(job.best_reward, 2)
            } for job in training_jobs
        ],
        'meta_training_jobs': [
            {
                'id': job.id,
                'status': job.status,
                'progress': job.progress,
                # Safely get the sharpe ratio from the results JSON
                'best_sharpe_ratio': job.results.get('sharpe_ratio', 0.0) if job.results else 0.0
            } for job in meta_jobs
        ]
        # We can add PaperTrader status here later
    }

    return JsonResponse(data)


@login_required
def trader_status_api(request):
    try:
        broker = Broker()
        account_info = broker.api.get_account()
        positions = broker.api.list_positions()

        trader_model, created = PaperTrader.objects.get_or_create(id=1)

        data = {
            "status": trader_model.status,
            "equity": float(account_info.equity),
            "buying_power": float(account_info.buying_power),
            "positions": [
                {
                    "symbol": pos.symbol,
                    "qty": float(pos.qty),
                    "market_value": float(pos.market_value),
                    "unrealized_pl": float(pos.unrealized_pl),
                } for pos in positions
            ]
        }
        return JsonResponse(data)
    except Exception as e:
        return JsonResponse({"status": "ERROR", "message": str(e)})

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
    Provides the detailed, real-time activity of a running task
    by checking the Celery task's custom state.
    """
    # We assume a single paper trader instance with id=1
    trader, created = PaperTrader.objects.get_or_create(id=1)

    # Check if the bot is supposed to be running and has a task ID
    if trader.status == 'RUNNING' and trader.celery_task_id:
        # Use the task ID to get the task's result/state from Celery
        task_result = AsyncResult(trader.celery_task_id)

        # The 'info' attribute holds the custom metadata we set with update_state()
        if task_result.info and isinstance(task_result.info, dict):
            return JsonResponse(task_result.info)

    # If the bot isn't running or has no status, return a default message
    return JsonResponse({'activity': 'Not Running'})
