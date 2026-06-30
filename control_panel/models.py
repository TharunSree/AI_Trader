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
    training_duration_days = models.IntegerField(default=365)
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
    training_duration_days = models.IntegerField(default=365)

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

    @property
    def name(self):
        return f"Paper Bot #{self.id}"

    @property
    def symbol(self):
        return "BTC/USD"

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

    # API Keys
    alpaca_api_key = models.CharField(max_length=255, blank=True, default='')
    alpaca_secret_key = models.CharField(max_length=255, blank=True, default='')
    broker_endpoint = models.CharField(max_length=255, blank=True, default='')
    gemini_api_key = models.CharField(max_length=255, blank=True, default='')
    openai_api_key = models.CharField(max_length=255, blank=True, default='')
    anthropic_api_key = models.CharField(max_length=255, blank=True, default='')

    # Remote Gaming Rig (Windows Host) Launch Settings
    gaming_rig_ip = models.CharField(max_length=255, blank=True, default='', help_text="Local IP or hostname of the Windows gaming machine")
    gaming_rig_ssh_username = models.CharField(max_length=255, blank=True, default='', help_text="Windows user account name for SSH connection")
    gaming_rig_ssh_password = models.CharField(max_length=255, blank=True, default='', help_text="Windows user password or SSH key path")
    steam_username = models.CharField(max_length=255, blank=True, default='', help_text="Steam ID64 or custom profile URL name for auto-syncing library playtimes")

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
    RUN_MODES = [
        ('EVOLUTION', 'Evolution Model'),
        ('PAPER', 'Paper Trading'),
        ('LIVE', 'Live Trading'),
    ]
    report_type = models.CharField(max_length=15, choices=REPORT_TYPES)
    run_mode = models.CharField(max_length=20, choices=RUN_MODES, default='PAPER')
    timestamp = models.DateTimeField(auto_now_add=True)
    markdown_path = models.CharField(max_length=255)
    pdf_path = models.CharField(max_length=255, null=True, blank=True)
    total_revenue = models.DecimalField(max_digits=15, decimal_places=2, default=0.0)
    total_trades = models.IntegerField(default=0)
    win_rate = models.FloatField(default=0.0)

    def __str__(self):
        return f"{self.report_type} Report ({self.run_mode}) - {self.timestamp.strftime('%Y-%m-%d')}"

class SystemAlert(models.Model):
    ALERT_TYPES = [
        ('INFO', 'Information'),
        ('WARNING', 'Warning'),
        ('CRITICAL', 'Critical Level'),
        ('EVOLUTION', 'Neural Evolution Promotion'),
    ]
    level = models.CharField(max_length=15, choices=ALERT_TYPES, default='INFO')
    title = models.CharField(max_length=255)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)
    
    # Used for ping persistence in Evolution Engine
    insist_count = models.IntegerField(default=0)
    related_model_reference = models.CharField(max_length=255, null=True, blank=True)
    
    def __str__(self):
        return f"[{self.level}] {self.title}"


class ModelVariant(models.Model):
    """
    Neural Evolution Engine: Tracks each mutated model variant.
    Each variant runs in a virtual paper trading sandbox, competing
    against the active production model for promotion.
    """
    STATUS_CHOICES = [
        ('TESTING', 'Testing'),      # Running virtual paper trade
        ('PENDING', 'Pending'),      # Won comparison, awaiting human approval
        ('PROMOTED', 'Promoted'),    # Became the live model
        ('FAILED', 'Failed'),        # Lost the comparison or was rejected
        ('QUEUED', 'Queued'),        # Waiting for a test slot (spawn guard)
    ]
    name = models.CharField(max_length=150, default='Unnamed Variant')
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='TESTING')
    parent_variant = models.ForeignKey(
        'self', null=True, blank=True, on_delete=models.SET_NULL,
        related_name='children', help_text="The variant this was mutated from"
    )
    parent_trader = models.ForeignKey(
        PaperTrader, null=True, blank=True, on_delete=models.SET_NULL,
        related_name='spawned_variants',
        help_text="The live PaperTrader this variant is benchmarked against"
    )

    # The mutated agent code (stored as text, NOT overwriting ppo_agent.py)
    agent_code = models.TextField(help_text="Full ppo_agent.py source code for this variant")
    model_weights = models.BinaryField(null=True, blank=True, help_text="Trained .pth weights if applicable")

    # Capital snapshot at fork time
    starting_cash = models.DecimalField(
        max_digits=15, decimal_places=2, default=100.00,
        help_text="Inherited from parent model's balance at fork time"
    )

    # Test window
    test_start = models.DateTimeField(auto_now_add=True)
    test_duration_days = models.IntegerField(default=14)

    # Virtual Paper Trading Results (updated every cycle by the virtual engine)
    virtual_balance = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    virtual_trades_count = models.IntegerField(default=0)
    virtual_pnl = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    virtual_pnl_pct = models.FloatField(default=0.0, help_text="PnL as percentage of starting_cash")
    sharpe_ratio = models.FloatField(default=0.0)
    max_drawdown_pct = models.FloatField(default=0.0)
    win_rate = models.FloatField(default=0.0)

    # Mutation metadata
    mutation_reasoning = models.TextField(blank=True, help_text="AI reasoning for the code changes")
    diff_summary = models.TextField(blank=True, help_text="Unified diff of code changes")

    # Process tracking
    celery_task_id = models.CharField(max_length=255, null=True, blank=True, editable=False)
    error_message = models.TextField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"[{self.status}] {self.name} (PnL: {self.virtual_pnl_pct:+.2f}%)"

    @property
    def is_test_expired(self):
        """True if the test window has elapsed."""
        from django.utils import timezone as tz
        from datetime import timedelta
        if not self.test_start:
            return False
        return tz.now() >= self.test_start + timedelta(days=self.test_duration_days)

    @property
    def days_remaining(self):
        """Days left in the test window."""
        from django.utils import timezone as tz
        from datetime import timedelta
        if not self.test_start:
            return self.test_duration_days
        end = self.test_start + timedelta(days=self.test_duration_days)
        remaining = (end - tz.now()).total_seconds() / 86400
        return max(0, round(remaining, 1))


class VirtualTrade(models.Model):
    """
    Neural Evolution Engine: Trade log for virtual paper trading.
    These trades NEVER hit Alpaca — they are pure simulations using live market prices.
    """
    variant = models.ForeignKey(ModelVariant, on_delete=models.CASCADE, related_name='virtual_trades')
    timestamp = models.DateTimeField(auto_now_add=True)
    symbol = models.CharField(max_length=20)
    action = models.CharField(max_length=4)  # BUY or SELL
    quantity = models.DecimalField(max_digits=15, decimal_places=8)
    price = models.DecimalField(max_digits=15, decimal_places=4)
    notional_value = models.DecimalField(max_digits=15, decimal_places=2)
    virtual_balance_after = models.DecimalField(
        max_digits=15, decimal_places=2, default=0,
        help_text="Running balance after this trade"
    )

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.timestamp:%H:%M} {self.action} {self.quantity} {self.symbol} @ ${self.price}"


class OnlineLearningLog(models.Model):
    EVENT_TYPES = [
        ('ENTRY', 'Entry Recorded'),
        ('EXIT', 'Exit Recorded'),
        ('UPDATE', 'Micro-Update'),
        ('CHECKPOINT', 'Checkpoint Saved'),
        ('INIT', 'Learner Initialized'),
        ('MANUAL', 'Manual Edit'),
        ('DECISION', 'Model Decision Check'),
    ]
    trader = models.ForeignKey(PaperTrader, on_delete=models.CASCADE, null=True, blank=True)
    event_type = models.CharField(max_length=15, choices=EVENT_TYPES)
    symbol = models.CharField(max_length=20, blank=True)
    details = models.JSONField(default=dict)  # pnl, reward, win/loss, etc.
    reason = models.TextField(blank=True, default='')  # Reason for weight change or event description
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"[{self.event_type}] {self.timestamp:%Y-%m-%d %H:%M} - {self.reason or self.event_type}"


class Game(models.Model):
    name = models.CharField(max_length=150)
    steam_app_id = models.CharField(max_length=50, blank=True, null=True, help_text="Steam App ID if it is a Steam game")
    local_path = models.CharField(max_length=500, blank=True, null=True, help_text="Path to local executable for non-Steam games")
    hours_played = models.FloatField(default=0.0)
    cover_image_url = models.CharField(max_length=500, blank=True, null=True)
    animated_bg_url = models.CharField(max_length=500, blank=True, null=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class GameGuide(models.Model):
    game = models.ForeignKey(Game, on_delete=models.CASCADE, related_name='guides')
    title = models.CharField(max_length=250)
    source_url = models.URLField(blank=True, null=True)
    content_markdown = models.TextField(help_text="Extracted guide content styled in markdown")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.game.name} - {self.title}"


class GameVideo(models.Model):
    game = models.ForeignKey(Game, on_delete=models.CASCADE, related_name='videos')
    title = models.CharField(max_length=250)
    youtube_url = models.URLField()
    video_id = models.CharField(max_length=50, blank=True, null=True, help_text="Extracted YouTube Video ID")
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        if self.youtube_url and not self.video_id:
            import re
            match = re.search(r'(?:v=|\/v\/|embed\/|youtu\.be\/|\/shorts\/)([^#\&\?]+)', self.youtube_url)
            if match:
                self.video_id = match.group(1)
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.game.name} - {self.title}"
