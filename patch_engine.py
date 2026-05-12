with open('src/core/async_engine.py', 'r', encoding='utf-8') as f:
    text = f.read()

target1 = '''        if side == 'buy' and active_principal < (current_price * 0.15):
            logger.warning(f"BUDGET LOCK: Holding BUY intention. Partition Liquidity (\) exhausted.")
            return'''

replacement1 = '''        # [FRACTIONAL OVERRIDE] Physical share bounding removed for Micro-Accounts'''

target2 = '''        # Hard mathematical cap so it can NEVER breach its partition
        trade_size_usd = min(trade_size_usd, active_principal)
        
        if trade_size_usd < 1.0 and side == 'buy':
            return # Block fractional dusting rejections on Alpaca'''

replacement2 = '''        # Hard mathematical cap so it can NEVER breach its partition
        trade_size_usd = min(trade_size_usd, active_principal)
        
        # [FRACTIONAL OVERRIDE] Force base .00 minimum bounds dynamically
        trade_size_usd = max(1.0, trade_size_usd) if active_principal >= 1.0 else active_principal
        
        if trade_size_usd < 1.0 and side == 'buy':
            logger.warning(f"Insufficient active principal (\) for Alpaca  trades. Blocked.")
            return # Block fractional dusting rejections on Alpaca'''

text = text.replace(target1, replacement1)
text = text.replace(target2, replacement2)

with open('src/core/async_engine.py', 'w', encoding='utf-8') as f:
    f.write(text)
