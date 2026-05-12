import re

with open('templates/login.html', 'r', encoding='utf-8') as f:
    text = f.read()

# Match the HTML of jarvis loader
text = re.sub(r'<!-- Jarvis Loader Overlay \(Iron Man Mk 2 HUD Cinematic\) -->.*?<!-- Initialize Particles & Logic -->', '<!-- Initialize Particles & Logic -->', text, flags=re.DOTALL)

with open('templates/login.html', 'w', encoding='utf-8') as f:
    f.write(text)
