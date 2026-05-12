import re

with open('control_panel/views.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Fix get_trader_stats to return initial_cash and account so the frontend Modal works
target = "'goal_amount': float(trader.goal_amount or 0.0),\n"
replacement = target + "        'initial_cash': float(trader.initial_cash or 0.0),\n        'account': trader.account,\n"

text = text.replace(target, replacement)

with open('control_panel/views.py', 'w', encoding='utf-8') as f:
    f.write(text)
