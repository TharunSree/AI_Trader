import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trader_project.settings')
django.setup()
from src.execution.broker import Broker
b = Broker()
clock = b.api.get_clock()
print(f"Server Clock Time: {clock.timestamp}")
print(f"Is Open: {clock.is_open}")
print(f"Next Open: {clock.next_open}")
print(f"Next Close: {clock.next_close}")
