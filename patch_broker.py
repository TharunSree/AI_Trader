import re

with open('src/core/async_engine.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = "broker = await asyncio.to_thread(Broker)"
replacement = "broker = self.broker"

if target in text:
    text = text.replace(target, replacement)
    with open('src/core/async_engine.py', 'w', encoding='utf-8') as f:
        f.write(text)

