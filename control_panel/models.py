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
    steam_api_key = models.CharField(max_length=255, blank=True, default='', help_text="Steam Web API Key for reliable full library sync")
    global_wishlist_budget = models.FloatField(default=1000.0, help_text="Default target budget in INR for discount watchlist games")

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
    playtime_offset = models.FloatField(default=0.0, help_text="Manual playtime offset to add to Steam hours")
    years_played = models.FloatField(default=1.0, help_text="Number of years this game has been played to normalize rankings")
    cover_image_url = models.CharField(max_length=500, blank=True, null=True)
    animated_bg_url = models.CharField(max_length=500, blank=True, null=True)
    is_active = models.BooleanField(default=True)
    watch_beta_recruitment = models.BooleanField(default=False, help_text="Watch this game for beta recruitment signups")
    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def normalized_hours_per_year(self):
        if self.years_played and self.years_played > 0:
            return self.hours_played / self.years_played
        return self.hours_played

    def has_active_beta(self):
        return self.beta_infos.filter(is_active=True).exists()

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


class GameBetaInfo(models.Model):
    game = models.ForeignKey(Game, on_delete=models.CASCADE, related_name='beta_infos')
    title = models.CharField(max_length=250)
    signup_link = models.TextField()
    is_active = models.BooleanField(default=True)
    discovered_at = models.DateTimeField(auto_now_add=True)
    notified = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.game.name} - {self.title}"


class WatchlistGame(models.Model):
    BUSINESS_MODELS = [
        ('F2P', 'Free to Play'),
        ('P2P', 'Pay to Play'),
        ('UNKNOWN', 'Unknown'),
    ]
    name = models.CharField(max_length=150)
    steam_app_id = models.CharField(max_length=50, blank=True, null=True, help_text="Steam App ID if available")
    header_image = models.URLField(max_length=500, blank=True, null=True, help_text="Horizontal banner image URL")
    expected_release_date = models.CharField(max_length=100, blank=True, null=True)
    business_model = models.CharField(max_length=50, choices=BUSINESS_MODELS, default='UNKNOWN')
    price_estimate = models.CharField(max_length=50, blank=True, null=True)
    system_requirements = models.TextField(blank=True, null=True)
    official_website = models.URLField(blank=True, null=True)
    current_status = models.CharField(max_length=100, blank=True, null=True)
    check_interval_days = models.IntegerField(default=7)
    last_checked_at = models.DateTimeField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

    def scout_details(self):
        import urllib.request
        import urllib.parse
        import json
        import re
        from django.utils import timezone
        from datetime import datetime
        
        # Search CheapShark first to find steamAppID if not set
        if not self.steam_app_id:
            encoded_title = urllib.parse.quote(self.name)
            search_url = f"https://www.cheapshark.com/api/1.0/games?title={encoded_title}"
            try:
                req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=5) as res:
                    search_data = json.loads(res.read().decode('utf-8'))
                if search_data:
                    matched = None
                    for r in search_data:
                        if r.get('external', '').lower() == self.name.lower():
                            matched = r
                            break
                    if not matched:
                        matched = search_data[0]
                    steam_id = matched.get('steamAppID')
                    if steam_id and steam_id != 'None':
                        self.steam_app_id = steam_id
                        self.save()
            except Exception:
                pass
                
        # Search Steam Store API fallback to find steamAppID if still not set
        if not self.steam_app_id:
            try:
                encoded_title = urllib.parse.quote(self.name)
                steam_search_url = f"https://store.steampowered.com/api/storesearch/?term={encoded_title}&l=english&cc=in"
                req = urllib.request.Request(steam_search_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=5) as res:
                    steam_data = json.loads(res.read().decode('utf-8'))
                items = steam_data.get('items', [])
                if items:
                    matched = None
                    for item in items:
                        if item.get('name', '').lower() == self.name.lower():
                            matched = item
                            break
                    if not matched:
                        matched = items[0]
                    self.steam_app_id = str(matched.get('id'))
                    self.save()
            except Exception:
                pass
                
        # If we have steam_app_id, query Steam Store API
        if self.steam_app_id:
            url = f"https://store.steampowered.com/api/appdetails?appids={self.steam_app_id}&cc=in"
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=6) as res:
                    response_data = json.loads(res.read().decode('utf-8'))
                
                game_data_outer = response_data.get(str(self.steam_app_id), {})
                if game_data_outer.get('success'):
                    data = game_data_outer.get('data', {})
                    
                    # 1. Official Website
                    website = data.get('website')
                    if website:
                        self.official_website = website
                        
                    # 2. System Requirements (Minimum)
                    pc_reqs = data.get('pc_requirements', {}).get('minimum', '')
                    if pc_reqs:
                        # Convert HTML lists and linebreaks into clean markdown newlines
                        html = pc_reqs
                        html = html.replace('<li>', '\n - ')
                        html = html.replace('</li>', '')
                        html = html.replace('<br>', '\n')
                        html = html.replace('<br/>', '\n')
                        html = html.replace('<br />', '\n')
                        clean_reqs = re.sub('<[^<]+?>', '', html)
                        lines = [line.strip() for line in clean_reqs.split('\n') if line.strip()]
                        clean_reqs = '\n'.join(lines)
                        clean_reqs = clean_reqs.replace('OS *:', 'OS:')
                        self.system_requirements = clean_reqs
                        
                    # 3. Price Estimate
                    price_overview = data.get('price_overview')
                    if price_overview:
                        final_formatted = price_overview.get('final_formatted', '')
                        if final_formatted:
                            self.price_estimate = final_formatted.replace('₹', '').strip()
                        else:
                            price_val = price_overview.get('final', 0) / 100
                            self.price_estimate = str(int(price_val))
                            
                    # 4. Release Date
                    rel_data = data.get('release_date', {})
                    if rel_data and not rel_data.get('coming_soon'):
                        date_str = rel_data.get('date', '')
                        if date_str:
                            try:
                                parsed_date = datetime.strptime(date_str, "%d %b, %Y").date()
                                self.expected_release_date = parsed_date.strftime("%Y-%m-%d")
                            except Exception:
                                try:
                                    parsed_date = datetime.strptime(date_str, "%b %d, %Y").date()
                                    self.expected_release_date = parsed_date.strftime("%Y-%m-%d")
                                except Exception:
                                    self.expected_release_date = date_str
                    elif rel_data and rel_data.get('coming_soon'):
                        self.expected_release_date = rel_data.get('date', 'Coming Soon')
                        
                    # 5. Business Model
                    is_free = data.get('is_free', False)
                    genres = [g.get('description', '').lower() for g in data.get('genres', [])]
                    if is_free or 'free to play' in genres:
                        self.business_model = 'F2P'
                    else:
                        self.business_model = 'P2P'
                        
                    self.save()
                    return True
            except Exception:
                pass

        # CheapShark price fallback for non-Steam games (Epic, GOG, Humble Store)
        if not self.price_estimate:
            try:
                # search_data is already loaded from lines 488 if CheapShark search succeeded
                if 'search_data' in locals() and search_data:
                    cheapest_usd = float(search_data[0].get('cheapest', 0.0))
                    if cheapest_usd > 0:
                        usd_to_inr = 83.5
                        try:
                            rate_url = "https://open.er-api.com/v6/latest/USD"
                            ex_req = urllib.request.Request(rate_url, headers={'User-Agent': 'Mozilla/5.0'})
                            with urllib.request.urlopen(ex_req, timeout=3) as ex_res:
                                rate_data = json.loads(ex_res.read().decode('utf-8'))
                                usd_to_inr = float(rate_data.get('rates', {}).get('INR', 83.5))
                        except Exception:
                            pass
                        self.price_estimate = str(int(cheapest_usd * usd_to_inr))
                        if self.business_model == 'UNKNOWN':
                            self.business_model = 'P2P'
                        self.save()
                        return True
            except Exception:
                pass

        # 1. Steam Header construction
        if self.steam_app_id and not self.header_image:
            self.header_image = f"https://cdn.cloudflare.steamstatic.com/steam/apps/{self.steam_app_id}/header.jpg"
            self.save()
            
        # 2. Local game cover fallback
        if not self.header_image:
            try:
                local_g = Game.objects.filter(name__iexact=self.name).first()
                if local_g and local_g.cover_image_url:
                    self.header_image = local_g.cover_image_url
                    self.save()
            except Exception:
                pass
                
        # 3. Wikipedia cover lookup fallback
        if not self.header_image:
            try:
                encoded = urllib.parse.quote(self.name)
                wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={encoded}&prop=pageimages&format=json&pithumbsize=600"
                wiki_req = urllib.request.Request(wiki_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(wiki_req, timeout=5) as wiki_res:
                    wiki_data = json.loads(wiki_res.read().decode('utf-8'))
                pages = wiki_data.get('query', {}).get('pages', {})
                for pid, pdata in pages.items():
                    thumb = pdata.get('thumbnail', {})
                    if thumb and thumb.get('source'):
                        self.header_image = thumb.get('source')
                        self.save()
                        break
            except Exception:
                pass

        # 4. DuckDuckGo System Requirements Crawler (Failsafe/Non-Steam fallback)
        if not self.system_requirements:
            try:
                query = urllib.parse.quote(f"{self.name} PC minimum system requirements GPU CPU RAM")
                url = f"https://html.duckduckgo.com/html/?q={query}"
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
                })
                with urllib.request.urlopen(req, timeout=6) as res:
                    html = res.read().decode('utf-8')
                snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
                
                keywords = ["ram", "gb", "intel", "i5", "i7", "gtx", "rtx", "gpu", "cpu", "processor", "windows", "storage", "ryzen", "amd", "nvidia"]
                valid_snippets = []
                for s in snippets:
                    clean_s = re.sub(r'<[^>]+>', ' ', s)
                    clean_s = ' '.join(clean_s.split())
                    matches = sum(1 for kw in keywords if kw in clean_s.lower())
                    if matches >= 2:
                        if clean_s not in valid_snippets:
                            valid_snippets.append(clean_s)
                if valid_snippets:
                    # Format snippets line-by-line beautifully
                    spec_lines = []
                    for vs in valid_snippets[:3]:
                        # Split by common separator punctuation
                        parts = re.split(r'[;.|]', vs)
                        for part in parts:
                            part_clean = part.strip()
                            if len(part_clean) > 8 and any(kw in part_clean.lower() for kw in keywords):
                                spec_lines.append(f"- {part_clean}")
                    unique_lines = []
                    for line in spec_lines:
                        if line not in unique_lines:
                            unique_lines.append(line)
                    self.system_requirements = "\n".join(unique_lines)
                    self.save()
            except Exception:
                pass

        # 5. DuckDuckGo Price Crawler (Failsafe/Non-Steam fallback)
        if not self.price_estimate:
            try:
                query = urllib.parse.quote(f"{self.name} game official price cost USD")
                url = f"https://html.duckduckgo.com/html/?q={query}"
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
                })
                with urllib.request.urlopen(req, timeout=6) as res:
                    html = res.read().decode('utf-8')
                snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
                
                prices = []
                for s in snippets:
                    clean_s = re.sub(r'<[^>]+>', ' ', s)
                    clean_s = ' '.join(clean_s.split())
                    matches = re.findall(r'\$\s?(\d{2,3}(?:\.\d{2})?)', clean_s)
                    for m in matches:
                        val = float(m)
                        if 29 <= val <= 160:
                            prices.append(val)
                if prices:
                    cheapest_usd = prices[0]
                    usd_to_inr = 83.5
                    try:
                        rate_url = "https://open.er-api.com/v6/latest/USD"
                        ex_req = urllib.request.Request(rate_url, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(ex_req, timeout=3) as ex_res:
                            rate_data = json.loads(ex_res.read().decode('utf-8'))
                            usd_to_inr = float(rate_data.get('rates', {}).get('INR', 83.5))
                    except Exception:
                        pass
                    self.price_estimate = str(int(cheapest_usd * usd_to_inr))
                    if self.business_model == 'UNKNOWN':
                        self.business_model = 'P2P'
                    self.save()
            except Exception:
                pass
        return False


class BudgetWatchlistGame(models.Model):
    name = models.CharField(max_length=150)
    steam_app_id = models.CharField(max_length=50, blank=True, null=True)
    target_budget = models.FloatField(default=1000.0, help_text="Target budget in INR (Rupees)")
    check_steam = models.BooleanField(default=True)
    check_epic = models.BooleanField(default=True)
    check_xbox = models.BooleanField(default=True)
    current_price = models.FloatField(blank=True, null=True, help_text="Current lowest price in INR (Rupees)")
    lowest_platform = models.CharField(max_length=50, blank=True, null=True)
    buy_link = models.TextField(blank=True, null=True, help_text="Direct link to purchase the deal")
    last_checked_at = models.DateTimeField(blank=True, null=True)
    notified_under_budget = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

    def scout_price(self):
        import urllib.request
        import urllib.parse
        import json
        from django.utils import timezone
        
        # Get exchange rate
        usd_to_inr = 83.5
        try:
            url = "https://open.er-api.com/v6/latest/USD"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=4) as res:
                ex_data = json.loads(res.read().decode('utf-8'))
                usd_to_inr = ex_data.get('rates', {}).get('INR', 83.5)
        except Exception:
            pass

        encoded_title = urllib.parse.quote(self.name)
        search_url = f"https://www.cheapshark.com/api/1.0/games?title={encoded_title}"
        try:
            req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=6) as res:
                search_data = json.loads(res.read().decode('utf-8'))
            
            if search_data:
                # Match hierarchy (Steam ID -> Exact name -> Substring -> First result)
                matched_result = None
                if self.steam_app_id:
                    for res in search_data:
                        if str(res.get('steamAppID')) == str(self.steam_app_id):
                            matched_result = res
                            break
                if not matched_result:
                    for res in search_data:
                        if res.get('external', '').lower() == self.name.lower():
                            matched_result = res
                            break
                if not matched_result:
                    for res in search_data:
                        if self.name.lower() in res.get('external', '').lower():
                            matched_result = res
                            break
                if not matched_result:
                    matched_result = search_data[0]

                game_id = matched_result.get('gameID')
                details_url = f"https://www.cheapshark.com/api/1.0/games?id={game_id}"
                
                req_details = urllib.request.Request(details_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req_details, timeout=6) as d_res:
                    details_data = json.loads(d_res.read().decode('utf-8'))
                
                deals = details_data.get('deals', [])
                lowest_price_usd = None
                lowest_deal_id = None
                store_map = {
                    "1": "Steam",
                    "2": "GamersGate",
                    "3": "GreenManGaming",
                    "7": "Funstock",
                    "11": "Humble Store",
                    "15": "Fanatical",
                    "18": "GOG",
                    "21": "Nintendo eShop",
                    "25": "Epic Games",
                    "27": "Xbox Store",
                    "30": "PlayStation Store"
                }
                lowest_store = None
                
                for deal in deals:
                    store_id = deal.get('storeID')
                    price_usd = float(deal.get('price', 999.0))
                    deal_id = deal.get('dealID')
                    
                    # Strict platform filtering
                    is_steam_deal = store_id in ["1", "2", "3", "11", "15", "18"]
                    is_epic_deal = store_id == "25"
                    is_xbox_deal = store_id == "27"
                    
                    if is_steam_deal and not self.check_steam: continue
                    if is_epic_deal and not self.check_epic: continue
                    if is_xbox_deal and not self.check_xbox: continue
                    
                    if not is_steam_deal and not is_epic_deal and not is_xbox_deal:
                        continue
                    
                    if lowest_price_usd is None or price_usd < lowest_price_usd:
                        lowest_price_usd = price_usd
                        lowest_deal_id = deal_id
                        lowest_store = store_map.get(store_id, f"Store #{store_id}")
                
                if lowest_price_usd is not None:
                    self.current_price = lowest_price_usd * usd_to_inr
                    self.lowest_platform = lowest_store
                    if lowest_deal_id:
                        self.buy_link = f"https://www.cheapshark.com/redirect?dealID={lowest_deal_id}"
                    else:
                        self.buy_link = None
            
            self.last_checked_at = timezone.now()
            self.save()
            return True
        except Exception:
            return False


class GamePlaytimeSession(models.Model):
    game = models.ForeignKey(Game, on_delete=models.CASCADE, related_name='sessions')
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(blank=True, null=True)
    limit_minutes = models.IntegerField(blank=True, null=True)
    duration_seconds = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.game.name} Session ({self.start_time})"

