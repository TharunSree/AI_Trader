with open('templates/papertrading_fleet.html', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''            <!-- Model Selection dropdown -->
            <div class="col-span-12 sm:col-span-8">'''

replacement = '''            <div class="col-span-12 mb-2">
              <label class="block text-[10px] uppercase font-mono tracking-widest text-[#94a3b8] mb-2 font-semibold">Select Broker Environment</label>
              <select name="account_id" class="w-full bg-[#1e293b] border border-[#334155] rounded-xl px-4 py-3 text-white text-sm focus:outline-none focus:border-brand-primary placeholder-[#64748b] transition-all disabled:opacity-50 font-mono focus:ring-1 focus:ring-brand-primary" required>
                <option value="">Attach Sandbox or Production Node</option>
                {% for acc in broker_accounts %}
                <option value="{{ acc.id }}">{{ acc.name }} {% if acc.is_live %}[PRODUCTION]{% else %}[SANDBOX]{% endif %}</option>
                {% endfor %}
              </select>
            </div>

            <!-- Model Selection dropdown -->
            <div class="col-span-12 sm:col-span-8">'''

text = text.replace(target, replacement)

with open('templates/papertrading_fleet.html', 'w', encoding='utf-8') as f:
    f.write(text)
