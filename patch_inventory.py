import re

with open('src/core/async_engine.py', 'r', encoding='utf-8') as f:
    text = f.read()

replacement = '''                inventory = {}
                for t in qs:
                    inventory[t.symbol] = inventory.get(t.symbol, 0.0) + (float(t.qty) if t.action == 'BUY' else -float(t.qty))'''

text = text.replace("                inventory = await asyncio.to_thread(self.broker.get_virtual_inventory, self.trader_id)", replacement)

with open('src/core/async_engine.py', 'w', encoding='utf-8') as f:
    f.write(text)
