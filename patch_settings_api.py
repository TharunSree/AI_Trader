import re

with open('control_panel/views.py', 'r', encoding='utf-8') as f:
    text = f.read()

new_views = '''
@login_required
def add_broker_account_api(request):
    if request.method == 'POST':
        from .models import BrokerAccount
        BrokerAccount.objects.create(
            name=request.POST.get('name', 'Unnamed env'),
            api_key=request.POST.get('api_key', ''),
            secret_key=request.POST.get('secret_key', ''),
            base_url=request.POST.get('base_url', 'https://paper-api.alpaca.markets')
        )
        messages.success(request, "Broker environment successfully attached.")
    return redirect('settings')

@login_required
def delete_broker_account_api(request, account_id):
    if request.method == 'POST':
        from .models import BrokerAccount
        acc = BrokerAccount.objects.filter(id=account_id).first()
        if acc:
            acc.delete()
            messages.info(request, "Broker environment securely detached.")
    return redirect('settings')
'''

target_settings = "    settings = SystemSettings.load()\n    context = {"
replacement_settings = '''    settings = SystemSettings.load()
    from .models import BrokerAccount
    context = {
        'broker_accounts': BrokerAccount.objects.all(),'''

text = text.replace(target_settings, replacement_settings)
text += new_views

with open('control_panel/views.py', 'w', encoding='utf-8') as f:
    f.write(text)
