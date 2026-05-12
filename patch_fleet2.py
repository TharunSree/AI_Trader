with open('templates/papertrading_fleet.html', 'r', encoding='utf-8') as f:
    text = f.read()

target1 = 'Vol / PnL'
replacement1 = 'Live PnL'

target2 = '<span class="text-white font-mono px-3 py-1 bg-slate-800/50 rounded-lg text-sm border border-slate-700/50">${{ row.total_notional|floatformat:2 }}</span>'
replacement2 = '<span class="font-mono px-3 py-1 bg-slate-800/50 rounded-lg text-sm border border-slate-700/50 {% if row.live_net_profit >= 0 %}text-brand-accent{% else %}text-red-400{% endif %}">{% if row.live_net_profit > 0 %}+{% endif %}${{ row.live_net_profit|floatformat:2 }}</span>'

text = text.replace(target1, replacement1)
text = text.replace(target2, replacement2)

with open('templates/papertrading_fleet.html', 'w', encoding='utf-8') as f:
    f.write(text)
