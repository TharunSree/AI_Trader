with open('templates/settings.html', 'r', encoding='utf-8') as f:
    text = f.read()

broker_ui = '''
  <!-- Broker Accounts Configuration -->
  <div class="glass rounded-2xl p-6 relative overflow-hidden flex flex-col md:col-span-2 mt-6">
    <h3 class="text-sm font-medium text-slate-400 mb-6 tracking-widest uppercase">Broker Account Modules</h3>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
      {% for account in broker_accounts %}
      <div class="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 flex justify-between items-center relative overflow-hidden group">
          {% if account.is_live %}
          <div class="absolute top-0 right-0 px-2 py-0.5 bg-red-500/20 text-red-400 text-[10px] font-bold tracking-widest uppercase rounded-bl-lg border-b border-l border-red-500/30">LIVE</div>
          {% endif %}
          <div>
              <div class="text-brand-primary font-mono text-sm mb-1">{{ account.name }}</div>
              <div class="text-xs text-slate-500 truncate max-w-[200px]">Key: {{ account.api_key|slice:":8" }}...</div>
          </div>
          <form method="post" action="{% url 'delete_broker_account_api' account.id %}" class="inline relative z-10">
              {% csrf_token %}
              <button type="submit" class="text-red-400/50 hover:text-red-400 transition-colors p-2 rounded hover:bg-red-400/10" title="Detach Account">
                  <i class="fas fa-trash"></i>
              </button>
          </form>
      </div>
      {% empty %}
      <div class="col-span-1 md:col-span-2 text-slate-500 font-mono text-sm text-center py-4 bg-slate-900/30 rounded-xl border border-slate-800/50">
          No secondary broker accounts linked. Operations will completely fail without .env fallback.
      </div>
      {% endfor %}
    </div>

    <h4 class="text-white font-medium mb-4 text-sm">Attach New Broker Environment</h4>
    <form method="post" action="{% url 'add_broker_account_api' %}" class="grid grid-cols-1 md:grid-cols-2 gap-4">
      {% csrf_token %}
      <div>
        <label class="block text-xs text-slate-400 uppercase tracking-widest mb-2 font-mono">Environment Alias</label>
        <input type="text" name="name" required placeholder="e.g. Testbed Alpha" class="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-4 py-2 text-white font-mono focus:outline-none focus:border-brand-primary text-sm transition-colors">
      </div>
      <div>
        <label class="block text-xs text-slate-400 uppercase tracking-widest mb-2 font-mono">Broker Base URL</label>
        <input type="text" name="base_url" required value="https://paper-api.alpaca.markets" class="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-4 py-2 text-brand-accent font-mono focus:outline-none focus:border-brand-primary text-sm transition-colors">
      </div>
      <div>
        <label class="block text-xs text-slate-400 uppercase tracking-widest mb-2 font-mono">API Key ID</label>
        <input type="text" name="api_key" required class="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-4 py-2 text-white font-mono focus:outline-none focus:border-brand-primary text-sm transition-colors">
      </div>
      <div>
        <label class="block text-xs text-slate-400 uppercase tracking-widest mb-2 font-mono">Secret Key</label>
        <input type="password" name="secret_key" required class="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-4 py-2 text-white font-mono focus:outline-none focus:border-brand-primary text-sm transition-colors">
      </div>
      <div class="col-span-1 md:col-span-2 flex items-center gap-3 mt-2 mb-4">
        <input type="checkbox" name="is_live" id="is_live" class="h-4 w-4 bg-slate-900 border-slate-700 rounded text-red-500 focus:ring-red-500 focus:ring-offset-slate-900 cursor-pointer">
        <label for="is_live" class="text-sm font-mono text-slate-300 cursor-pointer">Mark as <span class="text-red-400 font-bold">PRODUCTION (LIVE TRADING)</span> Account</label>
      </div>
      <div class="col-span-1 md:col-span-2">
        <button type="submit" class="w-full bg-brand-primary/20 hover:bg-brand-primary/40 border border-brand-primary/50 text-brand-primary px-6 py-3 rounded-lg font-mono tracking-widest uppercase text-xs transition-all">
          <i class="fas fa-link mr-2"></i>Link Broker Account
        </button>
      </div>
    </form>
  </div>
'''

target = "{% endblock page_content %}"
if "Broker Accounts Configuration" not in text:
    text = text.replace(target, broker_ui + "\n" + target)
    with open('templates/settings.html', 'w', encoding='utf-8') as f:
        f.write(text)
