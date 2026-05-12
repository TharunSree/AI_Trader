import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trader_project.settings')
django.setup()
from control_panel.models import TrainingJob
jobs = TrainingJob.objects.all()
for j in jobs:
    print(f"ID: {j.id}, Name: {j.name}, Status: {j.status}, Ready: {j.is_live_trading_ready}")
