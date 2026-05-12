import re

with open('templates/papertrading_fleet.html', 'r', encoding='utf-8') as f:
    text = f.read()

# Fix the URL
text = text.replace("if(form) form.action = '/control/paper-trading/' + id + '/edit/';", "if(form) form.action = '/dashboard/paper-trading/' + id + '/edit/';")

# Replace FontAwesome Empty Circle with beautifully crafted inline SVG warning icon
svg_icon = '''<svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
          <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>'''

text = text.replace('<i class="fas fa-exclamation-triangle"></i>', svg_icon)

with open('templates/papertrading_fleet.html', 'w', encoding='utf-8') as f:
    f.write(text)
