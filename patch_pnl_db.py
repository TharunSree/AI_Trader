with open('src/core/async_engine.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''                net_profit = (cap_rec - cap_spent) + unrealized_val
                
                # High Water Mark / Trailing Logic'''

replacement = '''                net_profit = (cap_rec - cap_spent) + unrealized_val
                
                # [TELEMETRY HOOK] Continual structural background sync of Unrealized Profit Metrics
                await asyncio.to_thread(lambda: PaperTrader.objects.filter(id=self.trader_id).update(live_net_profit=net_profit))
                
                # High Water Mark / Trailing Logic'''

if target in text:
    text = text.replace(target, replacement)
    with open('src/core/async_engine.py', 'w', encoding='utf-8') as f:
        f.write(text)
