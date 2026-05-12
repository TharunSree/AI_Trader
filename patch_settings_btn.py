import re

with open('templates/settings.html', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''<button type="submit" class="text-red-400/50 hover:text-red-400 transition-colors p-2 rounded hover:bg-red-400/10" title="Detach Account">
                  <span class="font-mono text-xs font-bold tracking-widest">[REMOVE]</span>
              </button>'''

replacement = '''<button type="submit" class="bg-red-500/10 text-red-400 border border-red-500/30 px-3 py-1.5 flex items-center justify-center rounded-lg hover:bg-red-500/20 hover:border-red-500/50 hover:text-red-300 transition-all shadow-sm" title="Detach Account">
                  <span class="font-mono text-xs font-bold tracking-widest uppercase">Delete</span>
              </button>'''

if target in text:
    text = text.replace(target, replacement)
    with open('templates/settings.html', 'w', encoding='utf-8') as f:
        f.write(text)
