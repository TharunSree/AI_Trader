# control_panel/models.py
import uuid

from django.db import models


class TrainingJob(models.Model):
    MAX_STORED_MODELS = 20
    STATUS_CHOICES = [('PENDING', 'Pending'), ('RUNNING', 'Running'), ('COMPLETED', 'Completed'), ('FAILED', 'Failed'),
                      ('STOPPED', 'Stopped')]
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='PENDING')
    name = models.CharField(max_length=100, default='Untitled Model')

    # Configuration from playbook
    feature_set_key = models.CharField(max_length=50, default='all_in')
    hyperparameter_key = models.CharField(max_length=50, default='balanced')
    ticker = models.CharField(max_length=20, default='SPY')
    window_size = models.IntegerField(default=10)
    initial_cash = models.FloatField(default=100000.0)

    # Status Tracking
    progress = models.IntegerField(default=0)
    best_reward = models.FloatField(default=0.0)
    is_live_trading_ready = models.BooleanField(default=False, help_text="Checked after successful evaluation")
    error_message = models.TextField(null=True, blank=True)
    model_weights = models.BinaryField(null=True, blank=True)
    celery_task_id = models.CharField(max_length=255, null=True, blank=True, editable=False)

    def save(self, *args, **kwargs):
        update_fields = kwargs.get('update_fields')
        should_enforce_retention = self.model_weights and (
            update_fields is None or 'model_weights' in update_fields
        )

        super().save(*args, **kwargs)

        if should_enforce_retention:
            self.enforce_model_retention_limit(exclude_id=self.id)

    @classmethod
    def enforce_model_retention_limit(cls, exclude_id=None):
        retained_ids = list(
            cls.objects.filter(model_weights__isnull=False)
            .order_by('-id')
            .values_list('id', flat=True)[:cls.MAX_STORED_MODELS]
        )
        if exclude_id and exclude_id not in retained_ids:
            retained_ids = [exclude_id] + retained_ids[:cls.MAX_STORED_MODELS - 1]

        queryset = cls.objects.filter(model_weights__isnull=False)
        if retained_ids:
            queryset = queryset.exclude(id__in=retained_ids)

        queryset.update(model_weights=None)

    @property
    def model_reference(self):
        return f"db:{self.id}"

    def __str__(self):
        return self.name or f"Training Job #{self.id}"


class MetaTrainingJob(models.Model):
    STATUS_CHOICES = [('PENDING', 'Pending'), ('RUNNING', 'Running'), ('COMPLETED', 'Completed'), ('FAILED', 'Failed')]
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='PENDING')
    start_time = models.DateTimeField(null=True, blank=True)
    end_time = models.DateTimeField(null=True, blank=True)

    ticker = models.CharField(max_length=20, default='SPY')

    initial_cash = models.FloatField(default=100000.0)
    target_equity = models.FloatField(default=200000.0)

    # --- NEW: Add progress field ---
    progress = models.IntegerField(default=0)

    results = models.JSONField(null=True, blank=True)  # This will store the best strategy info
    celery_task_id = models.CharField(max_length=255, null=True, blank=True, editable=False)

    error_message = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"Meta-Training Job #{self.id} - {self.status}"



class BrokerAccount(models.Model):
    name = models.CharField(max_length=100, help_text="e.g. 'Main Alpaca Account'")
    is_live = models.BooleanField(default=False, help_text="Distinguish real-money accounts from Sandbox")
    api_key = models.CharField(max_length=255)
    secret_key = models.CharField(max_length=255)
    base_url = models.CharField(max_length=255, default='https://paper-api.alpaca.markets')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class PaperTrader(models.Model):
    STATUS_CHOICES = [
        ('STOPPED', 'Stopped'),
        ('RUNNING', 'Running'),
        ('PAUSED', 'Paused'),
        ('FAILED', 'Failed'),
    ]
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='STOPPED')
    model_file = models.CharField(max_length=255, null=True, blank=True, help_text="e.g., best_agent.pth")
    celery_task_id = models.CharField(max_length=255, null=True, blank=True, editable=False)
    error_message = models.TextField(null=True, blank=True)
    initial_cash = models.DecimalField(max_digits=15, decimal_places=2, default=100000.00)
    goal_amount = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True)
    stop_loss_amount = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True)
    high_water_mark = models.DecimalField(max_digits=15, decimal_places=2, default=0.00)
    is_live = models.BooleanField(default=False)
    account = models.ForeignKey(BrokerAccount, on_delete=models.SET_NULL, null=True, blank=True, related_name='instances')
    live_net_profit = models.FloatField(default=0.0)

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
    sentiment_score = models.FloatField(default=0.0, null=True, blank=True, help_text="FinBERT Sentiment Score at time of trade")
    trader = models.ForeignKey(PaperTrader, on_delete=models.CASCADE, related_name='trades')

    def __str__(self):
        return f"{self.timestamp} - {self.action} {self.quantity} {self.symbol} @ {self.price}"


class SystemSettings(models.Model):
    CURRENCY_CHOICES = [('USD', 'USD ($)'), ('INR', 'INR (₹)')]
    singleton_id = models.IntegerField(primary_key=True, default=1, editable=False)
    display_currency = models.CharField(max_length=3, choices=CURRENCY_CHOICES, default='USD')
    notify_eod = models.BooleanField(default=True)
    notify_sos = models.BooleanField(default=True)
    notify_ab_test = models.BooleanField(default=True)
    notify_start_stop = models.BooleanField(default=False)
    
    # Security Settings
    lockscreen_password = models.CharField(max_length=128, blank=True, null=True, help_text="Hashed PIN or Password for lockscreen")
    idle_lock_minutes = models.IntegerField(default=5, help_text="Minutes of inactivity before showing lockscreen (0 to disable)")
    idle_logout_minutes = models.IntegerField(default=30, help_text="Minutes of inactivity before forcefully ending session")

    def save(self, *args, **kwargs):
        self.pk = 1
        super(SystemSettings, self).save(*args, **kwargs)

    @classmethod
    def load(cls):
        obj, created = cls.objects.get_or_create(pk=1)
        return obj

class TradingReport(models.Model):
    REPORT_TYPES = [
        ('DAILY', 'Daily Summary'),
        ('WEEKLY', 'Weekly Rolling Analysis'),
        ('MONTHLY', 'Monthly Performance'),
    ]
    report_type = models.CharField(max_length=15, choices=REPORT_TYPES)
    timestamp = models.DateTimeField(auto_now_add=True)
    markdown_path = models.CharField(max_length=255)
    pdf_path = models.CharField(max_length=255, null=True, blank=True)
    total_revenue = models.DecimalField(max_digits=15, decimal_places=2, default=0.0)
    total_trades = models.IntegerField(default=0)
    win_rate = models.FloatField(default=0.0)

    def __str__(self):
        return f"{self.report_type} Report - {self.timestamp.strftime('%Y-%m-%d')}"

class SystemAlert(models.Model):
    ALERT_TYPES = [
        ('INFO', 'Information'),
        ('WARNING', 'Warning'),
        ('CRITICAL', 'Critical Level'),
        ('AB_SWAP', 'A/B Test Swap Request'),
    ]
    level = models.CharField(max_length=15, choices=ALERT_TYPES, default='INFO')
    title = models.CharField(max_length=255)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)
    
    # Used for ping persistence in A/B Engine
    insist_count = models.IntegerField(default=0)
    related_model_reference = models.CharField(max_length=255, null=True, blank=True)
    
    def __str__(self):
        return f"[{self.level}] {self.title}"
