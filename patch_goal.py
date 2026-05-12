import sys
text = open('templates/model_detail.html', 'r', encoding='utf-8').read()

target = '<div class="text-[10px] uppercase tracking-widest text-slate-500 mb-1 font-semibold">Allocated Limit</div>'
replacement = '<div class="flex justify-between items-center"><div class="text-[10px] uppercase tracking-widest text-slate-500 mb-1 font-semibold">Allocated Limit</div> <span class="text-[10px] text-brand-accent font-mono">GOAL: {% if trader.goal_amount %}${{ trader.goal_amount|floatformat:2 }}{% else %}NONE{% endif %}</span></div>'

text = text.replace(target, replacement)
open('templates/model_detail.html', 'w', encoding='utf-8').write(text)
