# control_panel/admin.py

from django.contrib import admin
from .models import TrainingJob, PaperTrader
from .tasks import run_training_job_task, stop_celery_task

@admin.action(description='Start Selected Training Job(s)')
def start_training(modeladmin, request, queryset):
    for job in queryset:
        if job.status in ['PENDING', 'STOPPED']:
            run_training_job_task.delay(job.id)

@admin.action(description='Stop Selected Job(s)')
def stop_task(modeladmin, request, queryset):
    for job in queryset:
        if job.celery_task_id:
            stop_celery_task.delay(job.celery_task_id)

class TrainingJobAdmin(admin.ModelAdmin):
    list_display = ('id', 'status', 'num_episodes', 'target_equity', 'progress')
    actions = [start_training, stop_task]

admin.site.register(TrainingJob, TrainingJobAdmin)
admin.site.register(PaperTrader)