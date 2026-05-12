import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trader_project.settings')
django.setup()
from alpaca_trade_api.rest import REST
from django.conf import settings

api = REST(key_id=settings.ALPACA_API_KEY, secret_key=settings.ALPACA_SECRET_KEY, base_url='https://api.alpaca.markets', api_version='v2')
clock = api.get_clock()
print(f"Live Server Clock Time: {clock.timestamp}")
print(f"Live Is Open: {clock.is_open}")
print(f"Live Next Open: {clock.next_open}")
print(f"Live Next Close: {clock.next_close}")
