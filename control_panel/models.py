# control_panel/models.py

from django.db import models


class TrainingJob(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Pending'), ('RUNNING', 'Running'), ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'), ('STOPPED', 'Stopped'),
    ]
    # --- NEW: Add training mode choice ---
    TRAINING_MODE_CHOICES = [
        ('EPISODES', 'Train for N Episodes'),
        ('TARGET', 'Train to Target Equity'),
    ]

    # --- Configuration ---
    training_mode = models.CharField(max_length=10, choices=TRAINING_MODE_CHOICES, default='EPISODES')
    initial_cash = models.FloatField(default=100000.0, help_text="The starting principal for the simulation.")
    num_episodes = models.IntegerField(default=200)
    target_equity = models.FloatField(default=200000.0)

    # ... (rest of the model is the same)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='PENDING')
    progress = models.IntegerField(default=0)
    celery_task_id = models.CharField(max_length=255, null=True, blank=True, editable=False)

    def __str__(self): return f"Training Job #{self.id} - {self.status}"


class MetaTrainingJob(models.Model):
    STATUS_CHOICES = [('PENDING', 'Pending'), ('RUNNING', 'Running'), ('COMPLETED', 'Completed'), ('FAILED', 'Failed')]
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='PENDING')
    start_time = models.DateTimeField(null=True, blank=True)
    end_time = models.DateTimeField(null=True, blank=True)

    # Store the results
    best_strategy_details = models.JSONField(null=True, blank=True)
    celery_task_id = models.CharField(max_length=255, null=True, blank=True, editable=False)

    def __str__(self):
        return f"Meta-Training Job #{self.id} - {self.status}"


class PaperTrader(models.Model):
    STATUS_CHOICES = [('STOPPED', 'Stopped'), ('RUNNING', 'Running')]
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='STOPPED')
    celery_task_id = models.CharField(max_length=255, null=True, blank=True, editable=False)
    def __str__(self): return f"Paper Trading Bot - {self.status}"