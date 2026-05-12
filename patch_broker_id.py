import re

with open('control_panel/views.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace("account_id = request.POST.get('broker_account_id')", "account_id = request.POST.get('account_id')")

with open('control_panel/views.py', 'w', encoding='utf-8') as f:
    f.write(text)
