import sys, os
sys.path.insert(0, r'D:\AI_Trader')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trader_project.settings')

import django
django.setup()

from control_panel.models import PaperTrader

# Check all trader details
for t in PaperTrader.objects.all():
    print(f'=== Trader {t.id}: {t.name} ===')
    print(f'  Status: {t.status}')
    print(f'  Model File: {t.model_file}')
    print(f'  Initial Cash: {t.initial_cash}')
    print(f'  Error: {t.error_message}')
    trades = t.trades.all().order_by('-timestamp')
    print(f'  Trade Count: {trades.count()}')
    for trade in trades[:5]:
        print(f'    [{trade.timestamp}] {trade.action} {trade.symbol} qty={trade.quantity} price={trade.price}')
    print()

# Check if there's any other trading session
from django.apps import apps
all_models = apps.get_models()
for model in all_models:
    name = model.__name__
    if 'trade' in name.lower() or 'position' in name.lower() or 'order' in name.lower():
        try:
            count = model.objects.count()
            print(f'{name}: {count} records')
        except:
            pass
