with open('src/core/async_engine.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''                # High Water Mark / Trailing Logic
                if getattr(trader, 'goal_amount', None):
                    goal_val = float(trader.goal_amount)
                    high_water = float(trader.high_water_mark)
                    if net_profit > goal_val and net_profit > high_water:
                        await asyncio.to_thread(lambda: PaperTrader.objects.filter(id=self.trader_id).update(high_water_mark=net_profit))
                        logger.info(f"Trailing Stop Target Exceeded! Locked new High Water Mark: ${net_profit:,.2f}")
                    if high_water >= goal_val and net_profit <= goal_val:
                        logger.warning(f"Trailing Stop Triggered! Equity bled back to secure threshold of ${goal_val:,.2f}. SECURING PROFITS.")
                        await asyncio.to_thread(lambda: PaperTrader.objects.filter(id=self.trader_id).update(status='STOPPED', error_message=f"Goal Reached & Trailing Threshold Tripped at ${goal_val:,.2f}!"))
                        break'''

replacement = '''                # Absolute Hard Ceiling Goal Logic
                if getattr(trader, 'goal_amount', None):
                    goal_val = float(trader.goal_amount)
                    if net_profit >= goal_val:
                        logger.warning(f"ABSOLUTE GOAL REACHED! Net Profit ( ${net_profit:,.2f} ) hit the absolute threshold of ${goal_val:,.2f}. SECURING PROFITS AND HALTING!")
                        await asyncio.to_thread(lambda: PaperTrader.objects.filter(id=self.trader_id).update(status='STOPPED', error_message=f"Absolute Goal Achieved at ${net_profit:,.2f}!"))
                        
                        # Emergency liquidation of isolated bot inventory to crystallize the float
                        for sym, sqty in inventory.items():
                            if sqty > 0:
                                logger.info(f"Liquidating {sqty} shares of {sym} to secure target profit.")
                                try:
                                    await asyncio.to_thread(self.broker.api.submit_order,
                                        symbol=sym,
                                        qty=sqty,
                                        side='sell',
                                        type='market',
                                        time_in_force='ioc'
                                    )
                                    await asyncio.sleep(0.5)
                                except Exception as err:
                                    logger.error(f"Failed to cleanly liquidate {sym}: {err}")
                        
                        break'''

text = text.replace(target, replacement)
with open('src/core/async_engine.py', 'w', encoding='utf-8') as f:
    f.write(text)
