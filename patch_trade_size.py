with open('src/core/async_engine.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''        trade_size_usd = active_principal * (action_confidence * 0.15)
        
        # Hard mathematical cap so it can NEVER breach its partition
        trade_size_usd = min(trade_size_usd, active_principal)
        
        # [FRACTIONAL OVERRIDE] Force base .00 minimum bounds dynamically
        trade_size_usd = max(1.0, trade_size_usd) if active_principal >= 1.0 else active_principal
        
        if trade_size_usd < 1.0 and side == 'buy':
            logger.warning(f"Insufficient active principal (${active_principal:.2f}) for Alpaca  trades. Blocked.")
            return # Block fractional dusting rejections on Alpaca
        
        qty = trade_size_usd / current_price
        
        # Ensure we don't try to sell shares we don't own
        if side == 'sell':
            qty = min(qty, held_qty)
            if action == -1.0:
                qty = held_qty  # Dump entire inventory on 1-cent override'''

replacement = '''        if side == 'buy':
            trade_size_usd = active_principal * (action_confidence * 0.15)
            trade_size_usd = min(trade_size_usd, active_principal)
            trade_size_usd = max(1.0, trade_size_usd) if active_principal >= 1.0 else active_principal
            
            if trade_size_usd < 1.0:
                logger.warning(f"Insufficient active principal (${active_principal:.2f}) for Alpaca  trades. Blocked.")
                return 
            qty = trade_size_usd / current_price
        else:
            held_notional = held_qty * current_price
            trade_size_usd = held_notional * (action_confidence * 0.15)
            trade_size_usd = min(trade_size_usd, held_notional)
            
            if action == -1.0:
                qty = held_qty
            else:
                trade_size_usd = max(1.0, trade_size_usd) if held_notional >= 1.0 else held_notional
                if trade_size_usd < 1.0 and held_notional < 1.0:
                    qty = held_qty
                else:
                    qty = trade_size_usd / current_price
                    qty = min(qty, held_qty)'''

text = text.replace(target, replacement)
with open('src/core/async_engine.py', 'w', encoding='utf-8') as f:
    f.write(text)
