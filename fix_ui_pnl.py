import re

with open('templates/model_detail.html', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Remove the text-[10px] GOAL element from Allocated Limit box
target_goal = ''' <span class="text-[10px] text-brand-accent font-mono">GOAL: {% if trader.goal_amount %}${{ trader.goal_amount|floatformat:2 }}{% else %}NONE{% endif %}</span>'''
text = text.replace(target_goal, '')

# 2. Fix Live PnL whitespace-nowrap and gap
target_pnl = '<div id="dyn-live-pnl" class="text-2xl font-bold tracking-tight drop-shadow-md'
repl_pnl = '<div id="dyn-live-pnl" class="whitespace-nowrap text-2xl font-bold tracking-tight drop-shadow-md'
text = text.replace(target_pnl, repl_pnl)

# 3. Fix the active principle structure that failed to match earlier
text = re.sub(
    r'<div class="text-2xl font-bold text-white tracking-tight drop-shadow-md">\s*?\${{\s*initial_cash\s*}}\s*?</div>',
    '''      <div class="flex items-end gap-2">
        <div id="dyn-allocated-limit" class="text-2xl font-bold text-white tracking-tight drop-shadow-md whitespace-nowrap">
          ${{ initial_cash }}
        </div>
        <div id="dyn-initial-cash-goal" class="text-xs font-mono font-bold tracking-widest text-slate-500 mb-1 whitespace-nowrap">
          / ${{ initial_cash }}
        </div>
      </div>''',
    text
)

with open('templates/model_detail.html', 'w', encoding='utf-8') as f:
    f.write(text)
