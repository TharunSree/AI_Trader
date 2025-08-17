# control_panel/views.py

from django.shortcuts import render, redirect, get_object_or_404
import json
from .models import TrainingJob, MetaTrainingJob

from .tasks import run_training_job_task, stop_celery_task, run_meta_trainer_task


def dashboard_view(request):
    """The view for the main dashboard page."""

    # This is all SAMPLE data. We will replace it with real data later.
    context = {
        'account_equity': "105,432.10", 'equity_change_pct': "+2.5", 'trader_status': "STOPPED",
        'training_job_status': "COMPLETED", 'training_job_id': 4, 'training_job_progress': 100
    }
    context['positions'] = [
        {'symbol': 'AAPL', 'qty': 10, 'market_value': '1,500.00'},
        {'symbol': 'TSLA', 'qty': 5, 'market_value': '3,500.00'}
    ]
    context['recent_jobs'] = [
        {'id': 1, 'status': 'COMPLETED', 'performance': '15.7'},
        {'id': 2, 'status': 'COMPLETED', 'performance': '22.1'},
    ]
    chart_data = {
        'categories': ["2025-08-01", "2025-08-02", "2025-08-03", "2025-08-04", "2025-08-05", "2025-08-06"],
        'data': [100000, 101500, 101200, 102500, 103000, 105432]
    }
    context['equity_chart_data'] = {
        'categories': json.dumps(chart_data['categories']),
        'data': json.dumps(chart_data['data'])
    }
    return render(request, 'dashboard.html', context)


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
    context = {'training_jobs': all_jobs}
    return render(request, 'training.html', context)


def stop_job_view(request, job_id):
    if request.method == 'POST':
        job = get_object_or_404(TrainingJob, id=job_id)
        if job.celery_task_id:
            stop_celery_task.delay(job.celery_task_id)
            job.status = 'STOPPED'
            job.save()
    return redirect('training')

def start_meta_job_view(request):
    if request.method == 'POST':
        # Create a new job object
        meta_job = MetaTrainingJob.objects.create(status='PENDING')
        # Launch the background task
        run_meta_trainer_task.delay(meta_job.id)
    return redirect('training') # Redirect back to the training page


def papertrading_view(request):
    return render(request, 'papertrading.html', {})


def realtrading_view(request):
    return render(request, 'realtrading.html', {})


def settings_view(request):
    return render(request, 'settings.html', {})