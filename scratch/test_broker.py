import os
import sys
import django

sys.path.insert(0, r"d:\AI_Trader")
from dotenv import load_dotenv
load_dotenv(r"d:\AI_Trader\.env")
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trader_project.settings')
django.setup()

from src.execution.broker import Broker

broker = Broker()

for symbol in ["BTC/USD", "ETH/USD"]:
    try:
        p = broker.get_current_price(symbol)
        print(f"Current Price of {symbol}: ${p}")
    except Exception as e:
        print(f"Failed to fetch {symbol}: {e}")

    try:
        df = broker.get_historical_daily_bars(symbol, limit=10)
        print(f"Fetched {len(df)} historical bars for {symbol}")
        if not df.empty:
            print(df.tail(2))
    except Exception as e:
        print(f"Failed to fetch bars for {symbol}: {e}")
