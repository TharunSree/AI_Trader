with open('control_panel/views.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''    active_amount_recovered = sum(float(trader.trades.filter(action='SELL').aggregate(total=Coalesce(Sum('notional_value'), Value(0, output_field=DecimalField())))['total']) for trader in running_traders)

    assumed_base = 200000.0 if float(live_equity) > 150000 else 100000.0
    active_profit_made = float(live_equity) - assumed_base'''

replacement = '''    active_amount_recovered = sum(float(trader.trades.filter(action='SELL').aggregate(total=Coalesce(Sum('notional_value'), Value(0, output_field=DecimalField())))['total']) for trader in running_traders)

    # Calculate active_profit_made by aggregating the isolated live_net_profit blocks from all running nodes
    active_profit_made = sum(float(trader.live_net_profit) for trader in running_traders)'''

text = text.replace(target, replacement)

# Make start_trader_view parse the broker account from the UI dropdown
target2 = '''        raw_goal = request.POST.get('goal_amount')
        goal_amount = float(raw_goal) if raw_goal else None

        stop_loss_amount = None # Auto-calculated in async_engine'''

replacement2 = '''        raw_goal = request.POST.get('goal_amount')
        goal_amount = float(raw_goal) if raw_goal else None

        account_id = request.POST.get('broker_account_id')
        from .models import BrokerAccount
        account = BrokerAccount.objects.filter(id=account_id).first() if account_id else None

        stop_loss_amount = None # Auto-calculated in async_engine'''

text = text.replace(target2, replacement2)

target3 = '''            trader = PaperTrader.objects.create(
                model_file=model_file,
                initial_cash=initial_cash,
                goal_amount=goal_amount,
                stop_loss_amount=stop_loss_amount,
                status='STOPPED',
                error_message='',
            )'''

replacement3 = '''            trader = PaperTrader.objects.create(
                model_file=model_file,
                initial_cash=initial_cash,
                goal_amount=goal_amount,
                stop_loss_amount=stop_loss_amount,
                status='STOPPED',
                error_message='',
                account=account,
            )'''

text = text.replace(target3, replacement3)

# And add broker accounts to context in papertrading_view
target4 = '''    context = {
        'traders': traders,
        'trader_rows': [_get_trader_stats(trader) for trader in traders],'''
        
replacement4 = '''    from .models import BrokerAccount
    context = {
        'broker_accounts': BrokerAccount.objects.all(),
        'traders': traders,
        'trader_rows': [_get_trader_stats(trader) for trader in traders],'''

text = text.replace(target4, replacement4)

with open('control_panel/views.py', 'w', encoding='utf-8') as f:
    f.write(text)
