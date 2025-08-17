# trader_project/celery.py

import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trader_project.settings')

app = Celery('trader_project')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()