import sys

text = open('templates/model_detail.html', 'r', encoding='utf-8').read()

target = '<div id="dyn-live-pnl" class="text-2xl font-bold tracking-tight drop-shadow-md {% if trader.live_net_profit >= 0 %}text-brand-accent{% else %}text-red-400{% endif %}">{% if trader.live_net_profit > 0 %}+{% elif trader.live_net_profit < 0 %}-{% endif %}${{ trader.live_net_profit|floatformat:2 }}</div>'

replacement = '''<div class="flex items-end gap-2">
<div id="dyn-live-pnl" class="text-2xl font-bold tracking-tight drop-shadow-md {% if trader.live_net_profit >= 0 %}text-brand-accent{% else %}text-red-400{% endif %}">{% if trader.live_net_profit > 0 %}+{% elif trader.live_net_profit < 0 %}-{% endif %}${{ trader.live_net_profit|floatformat:2 }}</div>
<div id="dyn-live-pnl-goal" class="text-xs font-mono font-bold tracking-widest text-slate-500 mb-1">{% if trader.goal_amount %} / ${{ trader.goal_amount|floatformat:2 }} GOAL{% endif %}</div>
</div>'''

text = text.replace(target, replacement)
open('templates/model_detail.html', 'w', encoding='utf-8').write(text)
