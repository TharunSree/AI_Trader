import re

content = open('templates/papertrading_fleet.html', 'r', encoding='utf-8').read()

# Replace variables
content = content.replace('Live Terminal', 'Live Production Terminal')
content = content.replace('Multi-Model Paper Trading Matrix', 'Live Production Trading Matrix')
content = content.replace('papertrading', 'realtrading')
content = content.replace('Paper Trading', 'Real Money Trading')
content = content.replace('Live Trading Fleet', 'PRODUCTION MONEY TRADING')

# Toned down warning banner
warning_banner = """
    <div class="mb-6 rounded-xl border border-red-500 bg-slate-900/50 p-4 relative overflow-hidden">
        <div class="absolute top-0 left-0 w-1 h-full bg-red-500 animate-pulse"></div>
        <div class="flex items-center gap-3 ml-2">
            <svg class="w-6 h-6 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
            <div>
                <h4 class="font-bold uppercase tracking-widest text-slate-200 text-sm">Live Fire Zone</h4>
                <p class="text-xs font-mono text-slate-400 mt-0.5">WARNING: Actions on this page execute against REAL MONEY. Models must be certified to launch.</p>
            </div>
        </div>
    </div>
"""

content = content.replace('{% block page_content %}', '{% block page_content %}\n' + warning_banner)

# Filter for Certified Models
content = content.replace(
    '''{% for model in model_files %}''',
    '''{% for model in model_files %}\n          <option value="{{ model.value }}" {% if not model.is_certified %}disabled{% endif %}>{{ model.label }} {% if not model.is_certified %}(Uncertified){% endif %}</option>'''
)
# remove the line that has <option value="{{ model.value }}">{{ model.label }}</option> because I replaced it inline but wait, let me just do a safe regex.

content = re.sub(r'<option value="\{\{ model\.value \}\}">(.*?)<\/option>', '', content) # remove the original inside loop if replaced improperly, but actually I didn't replace the original so it's duplicated. Let me rewrite the replace accurately:

with open('templates/papertrading_fleet.html', 'r', encoding='utf-8') as f:
    orig = f.read()

content = orig.replace('Live Terminal', 'Live Production Terminal')
content = content.replace('Multi-Model Paper Trading Matrix', 'Live Production Trading Matrix')
content = content.replace('papertrading', 'realtrading')
content = content.replace('start_trader', 'start_real_trader') # Ensure action URL points to live start view
content = content.replace('Paper Trading Bot', 'Live Money Bot')
content = content.replace('Paper Trading', 'Real Money Trading')
content = content.replace('Live Trading Fleet', 'PRODUCTION MONEY TRADING')
content = content.replace('{% block page_content %}', '{% block page_content %}\n' + warning_banner)

# Enforce certified model option logic
model_opt_old = '<option value="{{ model.value }}">{{ model.label }}</option>'
model_opt_new = '<option value="{{ model.value }}" {% if not model.is_certified %}disabled class="text-slate-600 bg-slate-900"{% endif %}>{{ model.label }} {% if not model.is_certified %}[UNCERTIFIED]{% endif %}</option>'
content = content.replace(model_opt_old, model_opt_new)

# Force the Broker dropdown to only iterate over LIVE accounts visually
content = content.replace('{% for acc in broker_accounts %}', '{% for acc in broker_accounts %}{% if acc.is_live %}')
content = content.replace('{% endfor %}\n          </select>', '{% endif %}{% endfor %}\n          </select>')

with open('templates/realtrading.html', 'w', encoding='utf-8') as f:
    f.write(content)

print('Toned down realtrading UI generated!')
