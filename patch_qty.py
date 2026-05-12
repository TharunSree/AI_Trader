import re

with open('src/core/async_engine.py', 'r', encoding='utf-8') as f:
    text = f.read()

replacement = '''                inventory = {}
                for t in qs:
                    inventory[t.symbol] = inventory.get(t.symbol, 0.0) + (float(t.quantity) if t.action == 'BUY' else -float(t.quantity))'''

text = text.replace(
'''                inventory = {}
                for t in qs:
                    inventory[t.symbol] = inventory.get(t.symbol, 0.0) + (float(t.qty) if t.action == 'BUY' else -float(t.qty))''', 
replacement)

with open('src/core/async_engine.py', 'w', encoding='utf-8') as f:
    f.write(text)
