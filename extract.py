import re

with open('templates/login.html', 'r', encoding='utf-8') as f:
    text = f.read()

# Match the HTML of jarvis loader
html_match = re.search(r'(<!-- Jarvis Loader Overlay \(Iron Man Mk 2 HUD Cinematic\) -->.*?<!-- Initialize Particles & Logic -->)', text, re.DOTALL)
if html_match:
    with open('templates/partials/jarvis_loader.html', 'w', encoding='utf-8') as f:
        f.write(html_match.group(1).replace('<!-- Initialize Particles & Logic -->', ''))
    print('Extracted HTML')
    
    # Remove HTML from login.html
    new_text = text.replace(html_match.group(1).replace('<!-- Initialize Particles & Logic -->', ''), '')
    with open('templates/login.html', 'w', encoding='utf-8') as f:
        f.write(new_text)

# We will refactor the JS by replacement tools instead
