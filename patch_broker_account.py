import re

with open('src/execution/broker.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''class Broker:
    def __init__(self):
        api_key = getattr(settings, 'ALPACA_API_KEY', None) or getattr(settings, 'API_KEY', None)
        secret_key = (
                getattr(settings, 'ALPACA_SECRET_KEY', None)
                or getattr(settings, 'SECRET_KEY_ALPACA', None)
                or getattr(settings, 'SECRET_KEY', None)
        )
        raw_base = getattr(settings, 'BASE_URL', 'paper')
        if not api_key or not secret_key:
            raise ValueError("Missing Alpaca credentials (API_KEY / SECRET_KEY).")'''

replacement = '''class Broker:
    def __init__(self, account=None):
        if account:
            api_key = account.api_key
            secret_key = account.secret_key
            raw_base = account.base_url
        else:
            api_key = getattr(settings, 'ALPACA_API_KEY', None) or getattr(settings, 'API_KEY', None)
            secret_key = (
                    getattr(settings, 'ALPACA_SECRET_KEY', None)
                    or getattr(settings, 'SECRET_KEY_ALPACA', None)
                    or getattr(settings, 'SECRET_KEY', None)
            )
            raw_base = getattr(settings, 'BASE_URL', 'paper')
            
        if not api_key or not secret_key:
            raise ValueError("Missing Alpaca credentials (API_KEY / SECRET_KEY).")'''

text = text.replace(target, replacement)

with open('src/execution/broker.py', 'w', encoding='utf-8') as f:
    f.write(text)
