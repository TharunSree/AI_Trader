with open('control_panel/views.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''trader = PaperTrader.objects.filter(id=1).first()'''
replacement = '''trader = PaperTrader.objects.filter(status='RUNNING').first() or PaperTrader.objects.first()'''

if target in text:
    text = text.replace(target, replacement)
    with open('control_panel/views.py', 'w', encoding='utf-8') as f:
        f.write(text)
