with open('templates/model_detail.html', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''      <div class="text-2xl font-bold text-white tracking-tight drop-shadow-md">
        ${{ initial_cash }}
      </div>'''

replacement = '''      <div class="flex items-end gap-2">
        <div id="dyn-allocated-limit" class="text-2xl font-bold text-white tracking-tight drop-shadow-md">
          ${{ initial_cash }}
        </div>
        <div id="dyn-initial-cash-goal" class="text-xs font-mono font-bold tracking-widest text-slate-500 mb-1">
          / ${{ initial_cash }}
        </div>
      </div>'''
      
text = text.replace(target, replacement)

js_target = '''                    const goalEl = document.getElementById("dyn-live-pnl-goal");'''
js_replacement = '''                    const allocEl = document.getElementById("dyn-allocated-limit");
                    if (allocEl && traderData.active_principal !== undefined) {
                        allocEl.textContent = `$${parseFloat(traderData.active_principal).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
                    }
                    const initCashEl = document.getElementById("dyn-initial-cash-goal");
                    if (initCashEl && traderData.initial_cash) {
                        initCashEl.textContent = ` / $${parseFloat(traderData.initial_cash).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
                    }

                    const goalEl = document.getElementById("dyn-live-pnl-goal");'''

text = text.replace(js_target, js_replacement)

with open('templates/model_detail.html', 'w', encoding='utf-8') as f:
    f.write(text)
