import re

with open('templates/model_detail.html', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace Status
text = re.sub(
    r'<div class="text-2xl font-bold tracking-tight drop-shadow-md.*?\{\{\s*trader\.status\s*\}\}.*?</div>',
    '<div id="dyn-status" class="text-2xl font-bold tracking-tight drop-shadow-md {% if trader.status == \'RUNNING\' %}text-brand-accent{% elif trader.status == \'FAILED\' %}text-red-400{% else %}text-slate-400{% endif %}">{{ trader.status }}</div>',
    text,
    flags=re.DOTALL
)

# Replace Live Net PnL
text = re.sub(
    r'<div class="text-2xl font-bold tracking-tight drop-shadow-md \{\% if trader\.live_net_profit[^>]*>.*?</div>',
    '<div id="dyn-live-pnl" class="text-2xl font-bold tracking-tight drop-shadow-md {% if trader.live_net_profit >= 0 %}text-brand-accent{% else %}text-red-400{% endif %}">{% if trader.live_net_profit > 0 %}+{% elif trader.live_net_profit < 0 %}-{% endif %}${{ trader.live_net_profit|floatformat:2 }}</div>',
    text,
    flags=re.DOTALL
)

# Replace Trade Count
text = re.sub(
    r'<div class="text-[0-9a-zA-Z\-\s]+">\s*\{\{\s*trades\|length\s*\}\}\s*Actions\s*</div>',
    '<div id="dyn-trade-count" class="text-2xl font-bold text-brand-primary tracking-tight drop-shadow-md">{{ trades|length }} Actions</div>',
    text,
    flags=re.DOTALL
)

# Replace Active positions tbody if not already having an ID
text = re.sub(r'<tbody class="divide-y divide-slate-800">', '<tbody id="dyn-active-positions" class="divide-y divide-slate-800">', text)


js_injection = '''
    // Polling structure for dynamic telemetry 
    const currentTraderId = {{ trader.id }};
    function pollTraderStatus() {
        fetch("{% url 'trader_status_api' %}?_=" + Date.now(), {cache: 'no-store'})
            .then(res => res.json())
            .then(data => {
                const traderData = (data.traders || []).find(t => t.id === currentTraderId);
                if (traderData) {
                    const statusEl = document.getElementById("dyn-status");
                    if (statusEl) {
                        statusEl.textContent = traderData.status;
                        statusEl.className = `text-2xl font-bold tracking-tight drop-shadow-md ${traderData.status === 'RUNNING' ? 'text-brand-accent' : (traderData.status === 'FAILED' ? 'text-red-400' : 'text-slate-400')}`;
                    }

                    const tradeCountEl = document.getElementById("dyn-trade-count");
                    if (tradeCountEl) tradeCountEl.textContent = `${traderData.trade_count} Actions`;

                    const livePnlEl = document.getElementById("dyn-live-pnl");
                    if (livePnlEl) {
                        const val = parseFloat(traderData.live_net_profit) || 0;
                        livePnlEl.className = `text-2xl font-bold tracking-tight drop-shadow-md ${val >= 0 ? 'text-brand-accent' : 'text-red-400'}`;
                        livePnlEl.textContent = `${val > 0 ? '+' : (val < 0 ? '-' : '')}$${Math.abs(val).toFixed(2)}`;
                    }
                }
                
                // Update active positions grid across cluster if available
                if (data.positions) {
                    const tbody = document.getElementById("dyn-active-positions");
                    if (tbody) {
                        // Normally trader_status_api aggregates ALL positions, but for UI sake we update it.
                        tbody.innerHTML = '';
                        if (data.positions.length === 0) {
                            tbody.innerHTML = `<tr><td colspan="4" class="py-6 text-center text-slate-500 font-mono text-sm">NO ACTIVE POSITIONS</td></tr>`;
                        } else {
                            data.positions.forEach(p => {
                                const row = `
                                <tr class="hover:bg-slate-800/50 transition-colors">
                                  <td class="py-3 px-4 font-mono text-brand-primary">${p.symbol}</td>
                                  <td class="py-3 px-4 font-mono text-slate-300">${p.qty.toFixed(4)}</td>
                                  <td class="py-3 px-4 font-mono text-slate-300">$${p.market_value.toFixed(2)}</td>
                                  <td class="py-3 px-4 font-mono ${p.unrealized_pl >= 0 ? 'text-brand-accent' : 'text-red-400'} text-right">
                                    ${p.unrealized_pl > 0 ? '+' : ''}$${p.unrealized_pl.toFixed(2)}
                                  </td>
                                </tr>`;
                                tbody.insertAdjacentHTML('beforeend', row);
                            });
                        }
                    }
                }
            }).catch(e => console.error("Telemetry Error", e));
    }
    
    // Auto sync dashboard data 
    setInterval(pollTraderStatus, 3000);
'''

# Identify block extra_javascript insertion point
if "pollTraderStatus()" not in text:
    text = text.replace('setInterval(pollTerminal, 2000);', 'setInterval(pollTerminal, 2000);\n' + js_injection)

with open('templates/model_detail.html', 'w', encoding='utf-8') as f:
    f.write(text)
