import os

with open('d:\\AI_Trader\\templates\\realtrading.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace Live Fire Zone with Sandbox Environment
content = content.replace('bg-red-50 dark:bg-red-900/10', 'bg-blue-50 dark:bg-blue-900/10')
content = content.replace('border-red-500/50', 'border-blue-500/50')
content = content.replace('bg-red-500 animate-pulse', 'bg-blue-500')
content = content.replace('text-red-500 ml-2', 'text-blue-500 ml-2')
content = content.replace('text-red-700 dark:text-red-400', 'text-blue-700 dark:text-blue-400')
content = content.replace('Live Fire Zone', 'Sandbox Environment')
content = content.replace('text-red-600 dark:text-red-300', 'text-blue-600 dark:text-blue-300')
content = content.replace('Actions execute against REAL MONEY. Models must be certified to launch.', 'Safe execution zone. Actions do not affect real balances.')
content = content.replace('fa-exclamation-triangle', 'fa-flask')
content = content.replace('Live Production Terminal', 'Paper Trading Fleet')

# Replace start_real_trader with start_trader
content = content.replace('start_real_trader', 'start_trader')

# Replace account filter
content = content.replace('{% if acc.is_live %}\n                <option value="{{ acc.id }}">{{ acc.name }} [PRODUCTION]</option>\n                {% endif %}', '<option value="{{ acc.id }}">{{ acc.name }} {% if acc.is_live %}[PRODUCTION]{% else %}[SANDBOX]{% endif %}</option>')

# Replace the is_live=true API fetch
content = content.replace('?is_live=true&_=', '?is_live=false&_=')

with open('d:\\AI_Trader\\templates\\papertrading_fleet.html', 'w', encoding='utf-8') as f:
    f.write(content)
print('papertrading_fleet.html updated!')
