with open('templates/papertrading_fleet.html', 'r', encoding='utf-8') as f:
    text = f.read()

target1 = '''          <select name="model_file" class="flex-1 w-full md:w-auto min-w-[200px] bg-slate-900/70 border border-slate-700 text-brand-primary rounded-xl px-4 py-3 font-mono outline-none focus:border-brand-primary truncate">
            {% for model in model_files %}
            <option value="{{ model.value }}">{{ model.label }}</option>
            {% endfor %}
          </select>'''

replacement1 = '''          <select name="model_file" class="flex-1 w-full md:w-auto min-w-[150px] bg-slate-900/70 border border-slate-700 text-brand-primary rounded-xl px-4 py-3 font-mono outline-none focus:border-brand-primary truncate">
            {% for model in model_files %}
            <option value="{{ model.value }}">{{ model.label }}</option>
            {% endfor %}
          </select>
          
          <select name="broker_account_id" class="flex-1 w-full md:w-auto min-w-[150px] bg-slate-900/70 border border-slate-700 text-brand-primary rounded-xl px-4 py-3 font-mono outline-none focus:border-brand-primary truncate">
            <option value="">Default Environment</option>
            {% for account in broker_accounts %}
            <option value="{{ account.id }}">{{ account.name }}</option>
            {% endfor %}
          </select>'''

text = text.replace(target1, replacement1)

target2 = '''<div class="text-xs uppercase tracking-[0.24em] text-slate-500 font-mono mb-2">Active Limit</div>'''
replacement2 = '''<div class="text-xs uppercase tracking-[0.24em] text-slate-500 font-mono mb-2">Active Fleet Profit</div>'''

text = text.replace(target2, replacement2)

with open('templates/papertrading_fleet.html', 'w', encoding='utf-8') as f:
    f.write(text)
