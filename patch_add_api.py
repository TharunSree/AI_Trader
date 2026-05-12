with open('control_panel/views.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''        BrokerAccount.objects.create(
            name=request.POST.get('name', 'Unnamed env'),
            api_key=request.POST.get('api_key', ''),
            secret_key=request.POST.get('secret_key', ''),
            base_url=request.POST.get('base_url', 'https://paper-api.alpaca.markets')
        )'''

replacement = '''        BrokerAccount.objects.create(
            name=request.POST.get('name', 'Unnamed env'),
            api_key=request.POST.get('api_key', ''),
            secret_key=request.POST.get('secret_key', ''),
            base_url=request.POST.get('base_url', 'https://paper-api.alpaca.markets'),
            is_live=request.POST.get('is_live') == 'on'
        )'''

text = text.replace(target, replacement)

with open('control_panel/views.py', 'w', encoding='utf-8') as f:
    f.write(text)
