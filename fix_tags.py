import re

filepath = r'd:\AI_Trader\templates\backtest_lab.html'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix broken {{ \n  tag \n }}
# {{ followed by newline and optional spaces
content = re.sub(r'\{\{\n\s*', '{{ ', content)
# newline and optional spaces followed by }}
content = re.sub(r'\n\s*\}\}', ' }}', content)

# Fix broken {% \n  tag \n %}
content = re.sub(r'\{%\n\s*', '{% ', content)
content = re.sub(r'\n\s*%\}', ' %}', content)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed tags!")
