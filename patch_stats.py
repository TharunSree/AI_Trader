with open('control_panel/views.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''        'trade_count': trade_count,
        'total_notional': float(total_notional or 0),
        'last_trade_at': last_trade.timestamp.isoformat() if last_trade else '','''

replacement = '''        'trade_count': trade_count,
        'total_notional': float(total_notional or 0),
        'live_net_profit': float(trader.live_net_profit or 0.0),
        'last_trade_at': last_trade.timestamp.isoformat() if last_trade else '','''

text = text.replace(target, replacement)

with open('control_panel/views.py', 'w', encoding='utf-8') as f:
    f.write(text)
