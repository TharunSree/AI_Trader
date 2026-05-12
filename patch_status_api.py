import re

with open('control_panel/views.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''    try:
        broker = Broker()
        clock_data = broker.get_market_clock()
        equity = _to_float(broker.get_equity())
        buying_power = _to_float(broker.get_buying_power())
        positions_raw = broker.get_positions()
        positions = []
        for p in positions_raw:
            positions.append({
                "symbol": getattr(p, 'symbol', ''),
                "qty": _to_float(getattr(p, 'qty', 0)),
                "market_value": _to_float(getattr(p, 'market_value', 0)),
                "unrealized_pl": _to_float(getattr(p, 'unrealized_pl', 0)),
            })
        positions.sort(key=lambda x: x['market_value'], reverse=True)'''

replacement = '''    try:
        active_accounts = list({t.account for t in traders if t.status == 'RUNNING' and t.account})
        positions = []
        equity = 0.0
        buying_power = 0.0
        clock_data = {}

        if not active_accounts:
            try:
                broker = Broker()
                clock_data = broker.get_market_clock()
                equity = _to_float(broker.get_equity())
                buying_power = _to_float(broker.get_buying_power())
                for p in broker.get_positions():
                    positions.append({
                        "symbol": getattr(p, 'symbol', ''),
                        "qty": _to_float(getattr(p, 'qty', 0)),
                        "market_value": _to_float(getattr(p, 'market_value', 0)),
                        "unrealized_pl": _to_float(getattr(p, 'unrealized_pl', 0)),
                    })
            except Exception:
                pass # Default .env unavailable
        else:
            for acc in active_accounts:
                try:
                    broker = Broker(account=acc)
                    if not clock_data:
                        clock_data = broker.get_market_clock()
                    equity += _to_float(broker.get_equity())
                    buying_power += _to_float(broker.get_buying_power())
                    for p in broker.get_positions():
                        # Group up positions with same symbol if needed, but alpaca isolated accounts won't overlap usually
                        positions.append({
                            "symbol": getattr(p, 'symbol', ''),
                            "qty": _to_float(getattr(p, 'qty', 0)),
                            "market_value": _to_float(getattr(p, 'market_value', 0)),
                            "unrealized_pl": _to_float(getattr(p, 'unrealized_pl', 0)),
                        })
                except Exception:
                    pass

        positions.sort(key=lambda x: x['market_value'], reverse=True)'''

text = text.replace(target, replacement)

with open('control_panel/views.py', 'w', encoding='utf-8') as f:
    f.write(text)
