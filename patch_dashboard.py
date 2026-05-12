import sys

with open('templates/dashboard.html', 'r', encoding='utf-8') as f:
    text = f.read()

target = """                  <div class="text-2xl font-bold text-white drop-shadow-[0_0_8px_rgba(255,255,255,0.3)]"
                      id="trader-status-text">{% if trader and trader.status == 'RUNNING' %}ONLINE{% else %}STANDBY{% endif %}</div>
                  <div class="flex items-center mt-2 space-x-2">
                      <span id="trader-status-badge"
                          class="px-2 py-0.5 rounded text-xs font-bold tracking-widest {% if trader and trader.status == 'RUNNING' %}bg-green-500/20 text-green-400 border border-green-500/30{% else %}bg-yellow-500/20 text-yellow-500 border border-yellow-500/30{% endif %}">{{ trader.status|default:"OFFLINE" }}</span>
                  </div>"""

replacement = """                  <div class="text-2xl font-bold text-white drop-shadow-[0_0_8px_rgba(255,255,255,0.3)]"
                      id="trader-status-text">{% if running_count > 0 %}MATRIX ONLINE{% else %}STANDBY{% endif %}</div>
                  <div class="flex items-center mt-2 space-x-2">
                      <span id="trader-status-badge"
                          class="px-2 py-0.5 rounded text-xs font-bold tracking-widest {% if running_count > 0 %}bg-green-500/20 text-green-400 border border-green-500/30{% else %}bg-yellow-500/20 text-yellow-500 border border-yellow-500/30{% endif %}">{% if running_count > 0 %}{{ running_count }} ACTIVE NODES{% else %}OFFLINE{% endif %}</span>
                  </div>"""

if target in text:
    text = text.replace(target, replacement)
    
js_target = """                if (traderStatusBadge.innerText === 'RUNNING') {
                    traderStatusText.innerText = "DISCONNECTED";
                    traderStatusBadge.innerText = "OFFLINE";
                    traderStatusBadge.className = "px-2 py-0.5 rounded text-xs font-bold tracking-widest bg-red-500/20 text-red-400 border border-red-500/30";
                }"""

js_replacement = """                if (traderStatusBadge.innerText.includes('ACTIVE NODES')) {
                    traderStatusText.innerText = "DISCONNECTED";
                    traderStatusBadge.innerText = "OFFLINE";
                    traderStatusBadge.className = "px-2 py-0.5 rounded text-xs font-bold tracking-widest bg-red-500/20 text-red-400 border border-red-500/30";
                }"""

if js_target in text:
    text = text.replace(js_target, js_replacement)

with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(text)
