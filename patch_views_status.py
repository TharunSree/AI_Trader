import re

with open('control_panel/views.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = "active_accounts = list({t.account for t in traders if t.status == 'RUNNING' and t.account})"
replacement = "active_accounts = list({t.account for t in traders if t.account})"

if target in text:
    text = text.replace(target, replacement)
    with open('control_panel/views.py', 'w', encoding='utf-8') as f:
        f.write(text)

