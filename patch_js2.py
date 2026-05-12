import re

with open('templates/partials/jarvis_loader.html', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace(
    'el.style.transition = \\'opacity 0.1s, clip-path 0.9s cubic-bezier(0.8, 0, 0.2, 1), transform 1.2s cubic-bezier(0.16, 1, 0.3, 1), filter 1.5s linear\\';',
    'el.style.transition = \\'opacity 0.1s, clip-path 0.9s cubic-bezier(0.8, 0, 0.2, 1), transform 1.2s cubic-bezier(0.16, 1, 0.3, 1), filter 1.5s linear, box-shadow 1s linear\\';'
)

with open('templates/partials/jarvis_loader.html', 'w', encoding='utf-8') as f:
    f.write(text)
