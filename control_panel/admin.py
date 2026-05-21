from django.contrib import admin
from .models import (
    TrainingJob, MetaTrainingJob, PaperTrader, EvaluationJob,
    SystemSettings, ModelVariant, VirtualTrade
)

@admin.register(TrainingJob)
class TrainingJobAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'status', 'progress', 'best_reward',
                    'feature_set_key', 'hyperparameter_key', 'window_size', 'initial_cash')
    list_filter = ('status',)
    search_fields = ('id', 'name')
    exclude = ('model_weights',)

@admin.register(MetaTrainingJob)
class MetaTrainingJobAdmin(admin.ModelAdmin):
    list_display = ('id', 'status', 'progress')

@admin.register(PaperTrader)
class PaperTraderAdmin(admin.ModelAdmin):
    list_display = ('id', 'status', 'model_file')

@admin.register(EvaluationJob)
class EvaluationJobAdmin(admin.ModelAdmin):
    list_display = ('id', 'model_file', 'status')

@admin.register(SystemSettings)
class SystemSettingsAdmin(admin.ModelAdmin):
    list_display = ('singleton_id', 'display_currency')

@admin.register(ModelVariant)
class ModelVariantAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'status', 'starting_cash', 'virtual_pnl_pct', 'sharpe_ratio', 'days_remaining', 'created_at')
    list_filter = ('status',)
    search_fields = ('name',)
    readonly_fields = ('created_at', 'updated_at', 'test_start')
    exclude = ('agent_code', 'model_weights')  # Too large for admin view

@admin.register(VirtualTrade)
class VirtualTradeAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'variant', 'symbol', 'action', 'quantity', 'price', 'notional_value')
    list_filter = ('action', 'symbol')
