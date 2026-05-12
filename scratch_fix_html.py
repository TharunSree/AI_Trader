import os, glob, re

# 1. Restore all html files to original state
os.system("git checkout 91801885ec2193c0f0eaebb08b4718a5077c0850 -- d:\\AI_Trader\\templates")

# 2. Safely fix Django tags
for f in glob.glob(r'd:\AI_Trader\templates\**\*.html', recursive=True):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Fix block tags: { % block % } -> {% block %}
    content = re.sub(r'\{\s+%', '{%', content)
    content = re.sub(r'%\s+\}', '%}', content)
    
    # Fix variable tags conditionally (only if it looks like a django var, not arbitrary JS)
    # Actually, the user's IDE put spaces like `{ { variable } }`
    # Let's just fix the opening `{ {` and closing `} }` but ONLY if the opening is present.
    # Wait, JS doesn't usually have `{ {`.
    content = re.sub(r'\{\s+\{', '{{', content)
    # For closing, we only replace `} }` if it's preceded by `{{` somewhere? 
    # Just be careful: re.sub(r'\}\s+\}', '}}', ...) broke JS.
    # Let's match `{{ ... } }` and fix the closing part.
    content = re.sub(r'(\{\{.*?)\}\s+\}', r'\1}}', content, flags=re.DOTALL)
    
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)

# 3. Apply ui_patch_2.py safe CSS scrubber
os.system(r"d:\AI_Trader\.venv\Scripts\python.exe d:\AI_Trader\ui_patch_2.py")
