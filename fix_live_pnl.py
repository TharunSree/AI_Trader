import re
import os

# 1. Update views.py trader_status_api
with open("control_panel/views.py", "r", encoding="utf-8") as f:
    views_content = f.read()

api_old = """
    return JsonResponse({
        'positions': pos_list,
        'equity': equity,
        'buying_power': buying_power,
        'clock': clock_data,
        'traders': rows,
        'memory': memory_snapshot,
    })
"""
api_new = """
    # Calculate Active Fleet Profit for auto-update
    running_traders = [t for t in traders if t.status == 'RUNNING']
    active_starting_limit = 0.0
    active_amount_spent = 0.0
    active_amount_recovered = 0.0
    for t in running_traders:
        active_starting_limit += float(getattr(t, 'initial_cash', 0.0))
        for trade in t.trades.all():
            q = float(getattr(trade, 'quantity', 0))
            p = float(getattr(trade, 'price', 0))
            notional = float(getattr(trade, 'notional_value', q * p))
            if trade.action == 'BUY':
                active_amount_spent += notional
            elif trade.action == 'SELL':
                active_amount_recovered += notional
    
    assumed_base = 200000.0 if float(equity) > 150000 else 100000.0
    active_profit_made = float(equity) - assumed_base
    
    return JsonResponse({
        'positions': pos_list,
        'equity': equity,
        'buying_power': buying_power,
        'clock': clock_data,
        'traders': rows,
        'memory': memory_snapshot,
        'active_starting_limit': f"{active_starting_limit:,.2f}",
        'active_amount_spent': f"{active_amount_spent:,.2f}",
        'active_amount_recovered': f"{active_amount_recovered:,.2f}",
        'active_profit_made': f"{active_profit_made:,.2f}",
        'active_profit_raw': active_profit_made,
        'live_equity': f"{float(equity):,.2f}",
    })
"""
if api_old in views_content:
    views_content = views_content.replace(api_old, api_new)
    with open("control_panel/views.py", "w", encoding="utf-8") as f:
        f.write(views_content)
    print("views.py updated.")

# 2. Update papertrading_fleet.html & realtrading.html templates
def patch_template(filename, is_live=False):
    if not os.path.exists(filename): return
    with open(filename, "r", encoding="utf-8") as f:
        html = f.read()
    
    # Inject IDs into the 4 cards
    html = re.sub(
        r'<span class="({% if active_profit_raw[^>]+>)',
        r'<span id="active-fleet-profit" class="\1', html)
    html = re.sub(
        r'<div class="text-3xl font-semibold[^>]+>\s*\$\{\{ active_starting_limit \}\}\s*</div>',
        r'<div id="active-starting-limit" class="text-3xl font-semibold text-white tracking-tight drop-shadow-md">${{ active_starting_limit }}</div>', html)
    html = re.sub(
        r'<div class="text-xl font-bold text-slate-300[^>]+>\$\{\{ active_amount_spent \}\}<\/div>',
        r'<div id="active-amount-spent" class="text-xl font-bold text-slate-300 tracking-tight drop-shadow-md">${{ active_amount_spent }}</div>', html)
    html = re.sub(
        r'<div class="text-xl font-bold text-brand-primary[^>]+>\$\{\{ active_amount_recovered \}\}<\/div>',
        r'<div id="active-amount-recovered" class="text-xl font-bold text-brand-primary tracking-tight drop-shadow-md">${{ active_amount_recovered }}</div>', html)
    
    # Notice we replace text-brand-accent with correct CSS class, or simply ignore the color if realtrading
    equity_color = "red-100" if is_live else "brand-accent"
    html = re.sub(
        r'<div class="text-xl font-bold text-' + (equity_color if '/' not in equity_color else "orange-500") + r'[^>]+>\$\{\{ live_equity \}\}<\/div>',
        r'<div id="fleet-live-equity" class="text-xl font-bold text-' + ("orange-500" if is_live else "brand-accent") + r' tracking-tight drop-shadow-md">${{ live_equity }}</div>', html)
    
    # Inject JS updater
    js_inject = """
        updatePositions(data.positions || []);
        updateFleetRows(data.traders || []);
        
        // --- NEW: Dynamic PnL Updates ---
        if(document.getElementById('active-starting-limit')) document.getElementById('active-starting-limit').textContent = '$' + data.active_starting_limit;
        if(document.getElementById('active-amount-spent')) document.getElementById('active-amount-spent').textContent = '$' + data.active_amount_spent;
        if(document.getElementById('active-amount-recovered')) document.getElementById('active-amount-recovered').textContent = '$' + data.active_amount_recovered;
        if(document.getElementById('fleet-live-equity')) document.getElementById('fleet-live-equity').textContent = '$' + data.live_equity;
        
        let prof = document.getElementById('active-fleet-profit');
        if (prof) {
            let val = parseFloat(data.active_profit_raw || 0);
            let prefix = val > 0 ? '+' : (val < 0 ? '-' : '');
            let colorAccent = document.body.innerHTML.includes('Real Money') ? 'text-orange-500' : 'text-brand-accent';
            prof.className = (val >= 0 ? colorAccent : 'text-red-400') + ' font-bold font-mono tracking-widest text-sm';
            prof.textContent = prefix + '$' + Math.abs(val).toFixed(2);
        }
"""
    if "// --- NEW: Dynamic PnL Updates ---" not in html:
        html = html.replace("updatePositions(data.positions || []);", js_inject)
            
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"{filename} patched.")

patch_template("templates/papertrading_fleet.html")
patch_template("templates/realtrading.html", is_live=True)
