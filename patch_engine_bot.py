import re

with open('src/core/async_engine.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''        # [FRACTIONAL OVERRIDE] Force base .00 minimum bounds dynamically
        trade_size_usd = max(1.0, trade_size_usd) if active_principal >= 1.0 else active_principal
        
        if trade_size_usd < 1.0 and side == 'buy':
            logger.warning(f"Insufficient active principal (${active_principal:.2f}) for Alpaca  trades. Blocked.")
            return # Block fractional dusting rejections on Alpaca'''

replacement = '''        # --- PHASE 4 DYNAMIC AGGRESSION & HOUSE MONEY SCALING ---
        # "Aggressive or not but not make loss also"
        house_money = max(0.0, net_profit)
        baseline_risk_factor = 0.05
        
        if house_money > 0:
            # We have a profit cushion! We can dynamically scale up risk, strictly using house money.
            expanded_risk = min(trade_size_usd + house_money * 0.5, active_principal * 0.25)
            trade_size_usd = expanded_risk
        else:
            # Drop to defensive base to prevent tapping into initial principal too aggressively
            trade_size_usd = active_principal * (action_confidence * baseline_risk_factor)

        # Hard mathematical cap so it can NEVER breach its partition
        trade_size_usd = min(trade_size_usd, active_principal)
        
        # [FRACTIONAL OVERRIDE] Micro-Sizing .00 Clamp.
        # If principal >= 1.00 but calculated trade_size < 1.00, automatically bump it to .00 so  limits actually trade.
        if active_principal >= 1.00 and trade_size_usd < 1.00:
            trade_size_usd = 1.00
            
        if trade_size_usd < 1.0 and side == 'buy':
            logger.warning(f"Insufficient active principal (${active_principal:.2f}) for Alpaca $1 trades. Blocked.")
            return # Block fractional dusting rejections on Alpaca'''

if 'house_money' not in text:
    text = text.replace(target, replacement)
    
goal_target = '''                # Absolute Hard Ceiling Goal Logic
                if getattr(trader, 'goal_amount', None):
                    goal_val = float(trader.goal_amount)
                    if net_profit >= goal_val:
                        logger.warning(f"ABSOLUTE GOAL REACHED! Net Profit ( ${net_profit:,.2f} ) hit the absolute threshold of ${goal_val:,.2f}. SECURING PROFITS AND HALTING!")
                        await asyncio.to_thread(lambda: PaperTrader.objects.filter(id=self.trader_id).update(status='STOPPED', error_message=f"Absolute Goal Achieved at ${net_profit:,.2f}!"))
                        
                        # Emergency liquidation of isolated bot inventory to crystallize the float'''

goal_repl = '''                # Absolute Hard Ceiling Goal Logic
                if getattr(trader, 'goal_amount', None):
                    goal_val = float(trader.goal_amount)
                    if net_profit >= goal_val:
                        logger.warning(f"ABSOLUTE GOAL REACHED! Net Profit ( ${net_profit:,.2f} ) hit the absolute threshold of ${goal_val:,.2f}. SECURING PROFITS AND HALTING!")
                        await asyncio.to_thread(lambda: PaperTrader.objects.filter(id=self.trader_id).update(status='ACHIEVED', error_message=f"Absolute Goal Achieved at ${net_profit:,.2f}!"))
                        
                        # Emergency liquidation of isolated bot inventory to crystallize the float'''

if 'status=\'ACHIEVED\'' not in text:
    text = text.replace(goal_target, goal_repl)
    with open('src/core/async_engine.py', 'w', encoding='utf-8') as f:
        f.write(text)

