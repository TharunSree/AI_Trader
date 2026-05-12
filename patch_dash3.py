import re

with open('templates/dashboard.html', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace the text inside trader-status-text
text = re.sub(
    r'id="trader-status-text">.*?</div>',
    'id="trader-status-text">{% if running_count > 0 %}MATRIX ONLINE{% else %}STANDBY{% endif %}</div>',
    text
)

# Replace the trader-status-badge completely!
# To do this safely, we will look for <span id="trader-status-badge"...</span>
text = re.sub(
    r'<span id="trader-status-badge"[^>]*>.*?</span>',
    '''<span id="trader-status-badge"
                          class="px-2 py-0.5 rounded text-xs font-bold tracking-widest {% if running_count > 0 %}bg-green-500/20 text-green-400 border border-green-500/30{% else %}bg-yellow-500/20 text-yellow-500 border border-yellow-500/30{% endif %}">{% if running_count > 0 %}{{ running_count }} ACTIVE NODES{% else %}OFFLINE{% endif %}</span>''',
    text,
    flags=re.DOTALL
)

# Replace JS
text = text.replace("if (traderStatusBadge.innerText === 'RUNNING') {", "if (traderStatusBadge.innerText.includes('ACTIVE NODES')) {")

with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(text)
