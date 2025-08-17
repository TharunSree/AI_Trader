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
    best_reward = models.FloatField(default=0.0)
    progress = models.IntegerField(default=0)
    celery_task_id = models.CharField(max_length=255, null=True, blank=True, editable=False)

    def __str__(self): return f"Training Job #{self.id} - {self.status}"


class MetaTrainingJob(models.Model):
    STATUS_CHOICES = [('PENDING', 'Pending'), ('RUNNING', 'Running'), ('COMPLETED', 'Completed'), ('FAILED', 'Failed')]
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='PENDING')
    start_time = models.DateTimeField(null=True, blank=True)
    end_time = models.DateTimeField(null=True, blank=True)
    initial_cash = models.FloatField(default=100000.0)
    target_equity = models.FloatField(default=200000.0)
    results = models.JSONField(null=True, blank=True)
    celery_task_id = models.CharField(max_length=255, null=True, blank=True, editable=False)

    def __str__(self): return f"Meta-Training Job #{self.id} - {self.status}"


class PaperTrader(models.Model):
    STATUS_CHOICES = [
        ('STOPPED', 'Stopped'),
        ('RUNNING', 'Running'),
    ]
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='STOPPED')
    # --- NEW: Field to store the selected model file ---
    model_file = models.CharField(max_length=255, null=True, blank=True, help_text="e.g., ppo_agent_advanced.pth")
    celery_task_id = models.CharField(max_length=255, null=True, blank=True, editable=False)

    def __str__(self):
        return f"Paper Trading Bot - Status: {self.status}"


class EvaluationJob(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Pending'), ('RUNNING', 'Running'), ('COMPLETED', 'Completed'), ('FAILED', 'Failed'),
    ]

    # Configuration
    model_file = models.CharField(max_length=255)
    start_date = models.DateField()
    end_date = models.DateField()

    # Status
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='PENDING')
    celery_task_id = models.CharField(max_length=255, null=True, blank=True, editable=False)

    # Results
    results = models.JSONField(null=True, blank=True)  # Store metrics like Sharpe, return, etc.

    def __str__(self):
        return f"Evaluation Job #{self.id} for {self.model_file}"



class SystemSettings(models.Model):
    CURRENCY_CHOICES = [('USD', 'USD ($)'), ('INR', 'INR (₹)')]
    singleton_id = models.IntegerField(primary_key=True, default=1, editable=False)
    display_currency = models.CharField(max_length=3, choices=CURRENCY_CHOICES, default='USD')

    def save(self, *args, **kwargs):
        self.pk = 1
        super(SystemSettings, self).save(*args, **kwargs)

    @classmethod
    def load(cls):
        obj, created = cls.objects.get_or_create(pk=1)
        return obj
