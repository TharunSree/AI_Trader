import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trader_project.settings')
django.setup()
from control_panel.models import PaperTrader
traders = PaperTrader.objects.all()
for t in traders:
    print(f"ID: {t.id}, Status: {t.status}, is_live: {t.is_live}")
