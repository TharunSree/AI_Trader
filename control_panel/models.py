# control_panel/models.py
import uuid

from django.db import models


class TrainingJob(models.Model):
    STATUS_CHOICES = [('PENDING', 'Pending'), ('RUNNING', 'Running'), ('COMPLETED', 'Completed'), ('FAILED', 'Failed'),
                      ('STOPPED', 'Stopped')]
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='PENDING')

    # Configuration from playbook
    feature_set_key = models.CharField(max_length=50, default='all_in')
    hyperparameter_key = models.CharField(max_length=50, default='balanced')
    window_size = models.IntegerField(default=10)
    initial_cash = models.FloatField(default=100000.0)

    # Status Tracking
    progress = models.IntegerField(default=0)
    best_reward = models.FloatField(default=0.0)
    error_message = models.TextField(null=True, blank=True)
    celery_task_id = models.CharField(max_length=255, null=True, blank=True, editable=False)

    def __str__(self): return f"Training Job #{self.id}"


class MetaTrainingJob(models.Model):
    STATUS_CHOICES = [('PENDING', 'Pending'), ('RUNNING', 'Running'), ('COMPLETED', 'Completed'), ('FAILED', 'Failed')]
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='PENDING')
    start_time = models.DateTimeField(null=True, blank=True)
    end_time = models.DateTimeField(null=True, blank=True)

    initial_cash = models.FloatField(default=100000.0)
    target_equity = models.FloatField(default=200000.0)

    # --- NEW: Add progress field ---
    progress = models.IntegerField(default=0)

    results = models.JSONField(null=True, blank=True)  # This will store the best strategy info
    celery_task_id = models.CharField(max_length=255, null=True, blank=True, editable=False)

    error_message = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"Meta-Training Job #{self.id} - {self.status}"


class PaperTrader(models.Model):
    STATUS_CHOICES = [
        ('STOPPED', 'Stopped'),
        ('RUNNING', 'Running'),
        ('FAILED', 'Failed'),
    ]
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='STOPPED')
    model_file = models.CharField(max_length=255, null=True, blank=True, help_text="e.g., best_agent.pth")
    celery_task_id = models.CharField(max_length=255, null=True, blank=True, editable=False)
    error_message = models.TextField(null=True, blank=True)
    initial_cash = models.DecimalField(max_digits=15, decimal_places=2, default=100000.00)

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

    error_message = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"Evaluation Job #{self.id} for {self.model_file}"


class TradeLog(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    timestamp = models.DateTimeField(auto_now_add=True)
    symbol = models.CharField(max_length=10)
    action = models.CharField(max_length=4)  # BUY or SELL
    quantity = models.DecimalField(max_digits=15, decimal_places=8)
    price = models.DecimalField(max_digits=15, decimal_places=2)
    notional_value = models.DecimalField(max_digits=15, decimal_places=2)
    trader = models.ForeignKey(PaperTrader, on_delete=models.CASCADE, related_name='trades')

    def __str__(self):
        return f"{self.timestamp} - {self.action} {self.quantity} {self.symbol} @ {self.price}"


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
