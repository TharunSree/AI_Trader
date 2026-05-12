import re

with open('control_panel/views.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''    total_notional = trades.aggregate(
        total=Coalesce(Sum('notional_value'), Value(0, output_field=DecimalField(max_digits=20, decimal_places=2)))
    )['total']'''

replacement = '''    total_notional = trades.aggregate(
        total=Coalesce(Sum('notional_value'), Value(0, output_field=DecimalField(max_digits=20, decimal_places=2)))
    )['total']
    
    buy_notional = trades.filter(action='BUY').aggregate(total=Coalesce(Sum('notional_value'), Value(0, output_field=DecimalField(max_digits=20, decimal_places=2))))['total']
    sell_notional = trades.filter(action='SELL').aggregate(total=Coalesce(Sum('notional_value'), Value(0, output_field=DecimalField(max_digits=20, decimal_places=2))))['total']
    
    active_principal = max(0.0, float(trader.initial_cash or 0.0) - float(buy_notional - sell_notional))'''

text = text.replace(target, replacement)

target2 = '''        'initial_cash': float(trader.initial_cash or 0),'''
replacement2 = '''        'initial_cash': float(trader.initial_cash or 0),
        'active_principal': float(active_principal),'''

text = text.replace(target2, replacement2)

with open('control_panel/views.py', 'w', encoding='utf-8') as f:
    f.write(text)
