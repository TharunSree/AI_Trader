with open('templates/settings.html', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''<i class="fas fa-trash"></i>'''
replacement = '''<span class="font-mono text-xs font-bold tracking-widest">[REMOVE]</span>'''

if target in text:
    text = text.replace(target, replacement)
    with open('templates/settings.html', 'w', encoding='utf-8') as f:
        f.write(text)
