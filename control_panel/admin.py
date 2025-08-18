from django.contrib import admin
from .models import TrainingJob, MetaTrainingJob, PaperTrader, EvaluationJob, SystemSettings

@admin.register(TrainingJob)
class TrainingJobAdmin(admin.ModelAdmin):
    list_display = ('id', 'status', 'progress', 'best_reward',
                    'feature_set_key', 'hyperparameter_key', 'window_size', 'initial_cash')
    list_filter = ('status',)
    search_fields = ('id',)

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