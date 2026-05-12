import re

with open('control_panel/urls.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = "path('paper-trading/<int:trader_id>/pause/', views.pause_trader_view, name='pause_trader'),"
replacement = target + "\n    path('paper-trading/<int:trader_id>/edit/', views.edit_trader_view, name='edit_trader_api'),"
if 'edit_trader_api' not in text:
    text = text.replace(target, replacement)
    with open('control_panel/urls.py', 'w', encoding='utf-8') as f:
        f.write(text)

with open('control_panel/views.py', 'r', encoding='utf-8') as f:
    text = f.read()

view_func = '''
@login_required
def edit_trader_view(request, trader_id):
    if request.method == 'POST':
        trader = get_object_or_404(PaperTrader, id=trader_id)
        
        initial_cash = request.POST.get('initial_cash')
        goal_amount = request.POST.get('goal_amount')
        account_id = request.POST.get('account_id')
        
        if initial_cash:
            trader.initial_cash = float(initial_cash)
        
        if goal_amount:
            trader.goal_amount = float(goal_amount)
            
        if account_id:
            from .models import BrokerAccount
            acc = BrokerAccount.objects.filter(id=account_id).first()
            if acc:
                trader.account = acc
                
        trader.save()
    return redirect('papertrading')
'''

if 'def edit_trader_view' not in text:
    with open('control_panel/views.py', 'a', encoding='utf-8') as f:
        f.write(view_func)
