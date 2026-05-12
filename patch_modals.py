import re

with open('templates/papertrading_fleet.html', 'r', encoding='utf-8') as f:
    text = f.read()

# 2. Add Modal trigger inside the Table Actions
actions_start = '''<td class="py-3 px-4">
            <div class="flex justify-end gap-2">'''

new_edit_btn = '''
              <button onclick="openEditModal({{ row.id }}, '{{ row.initial_cash }}', '{{ row.goal_amount|default_if_none:"" }}', '{% if row.account %}{{row.account.id}}{% endif %}')" class="px-3 py-1.5 rounded-lg border border-blue-500/30 text-blue-400 hover:bg-blue-500/10 transition-all text-[10px] font-mono uppercase tracking-[0.18em] flex items-center">Edit</button>
'''

if 'openEditModal' not in text and actions_start in text:
    text = text.replace(actions_start, actions_start + new_edit_btn)

# 3. Replace Kill Button onsubmit
kill_form_old = '''<form method="post" action="{% url 'delete_trader_api' row.id %}" onsubmit="return confirm('Wipe instance entirely? Trade logs will be destroyed.');">'''
kill_form_new = '''<form id="kill-form-{{ row.id }}" method="post" action="{% url 'delete_trader_api' row.id %}">'''

if kill_form_old in text:
    text = text.replace(kill_form_old, kill_form_new)

kill_btn_old = '''<button type="submit" class="px-3 py-1.5 rounded-lg bg-red-600/10 border border-red-600/50 text-red-500 hover:bg-red-600/30 transition-all text-[10px] font-mono uppercase tracking-[0.18em] font-bold">Kill</button>'''
kill_btn_new = '''<button type="button" onclick="openKillModal({{ row.id }})" class="px-3 py-1.5 rounded-lg bg-red-600/10 border border-red-600/50 text-red-500 hover:bg-red-600/30 transition-all text-[10px] font-mono uppercase tracking-[0.18em] font-bold">Kill</button>'''

if kill_btn_old in text:
    text = text.replace(kill_btn_old, kill_btn_new)


# 4. Inject Modals and Toast HTML at the end of content
modals_html = '''
<!-- Edit Trader Modal -->
<div id="edit-trader-modal" class="fixed inset-0 z-50 flex items-center justify-center hidden bg-black/60 backdrop-blur-sm">
  <div class="glass border border-slate-700/50 rounded-2xl w-full max-w-md overflow-hidden transform scale-95 opacity-0 transition-all duration-300 shadow-2xl shadow-blue-900/20" id="edit-modal-content">
    <div class="px-6 py-4 border-b border-slate-700/50 flex justify-between items-center bg-slate-800/50">
      <h3 class="text-lg font-semibold text-white tracking-wide">Configure Instance <span id="edit-modal-id-label" class="text-blue-400 font-mono text-sm ml-2"></span></h3>
      <button onclick="closeEditModal()" class="text-slate-400 hover:text-white transition-colors"><i class="fas fa-times"></i></button>
    </div>
    <div class="p-6">
      <form id="edit-trader-form" method="post" action="">
        {% csrf_token %}
        <div class="space-y-4">
          <div>
            <label class="block text-xs uppercase tracking-widest text-slate-400 mb-2 font-mono">Bound Environment</label>
            <select name="account_id" id="edit-modal-account" class="w-full bg-slate-900/70 border border-slate-700 text-brand-primary rounded-xl px-4 py-3 font-mono text-sm outline-none focus:border-brand-primary focus:ring-1 focus:ring-brand-primary/50 transition duration-300">
              <option value="">[.ENV FALLBACK]</option>
              {% for acc in broker_accounts %}
              <option value="{{ acc.id }}">{{ acc.name }}</option>
              {% endfor %}
            </select>
          </div>
          <div>
            <label class="block text-xs uppercase tracking-widest text-slate-400 mb-2 font-mono">Initial Cash ($)</label>
            <input type="number" name="initial_cash" id="edit-modal-cash" step="0.01" class="w-full bg-slate-900/70 border border-slate-700 text-white rounded-xl px-4 py-3 font-mono text-sm outline-none focus:border-brand-primary focus:ring-1 focus:ring-brand-primary/50 transition duration-300">
          </div>
          <div>
            <label class="block text-xs uppercase tracking-widest text-slate-400 mb-2 font-mono">Profit Cap / Goal ($)</label>
            <input type="number" name="goal_amount" id="edit-modal-goal" step="0.01" class="w-full bg-slate-900/70 border border-slate-700 text-white rounded-xl px-4 py-3 font-mono text-sm outline-none focus:border-brand-primary focus:ring-1 focus:ring-brand-primary/50 transition duration-300 placeholder-slate-600" placeholder="(Optional)">
          </div>
        </div>
        <div class="mt-8 flex justify-end gap-3">
          <button type="button" onclick="closeEditModal()" class="px-5 py-2.5 rounded-xl border border-slate-600 text-slate-300 hover:bg-slate-700 hover:text-white transition-all font-medium text-sm">Cancel</button>
          <button type="submit" class="px-5 py-2.5 rounded-xl bg-blue-600/20 border border-blue-500/50 text-blue-400 hover:bg-blue-600 hover:text-white transition-all font-medium text-sm shadow-lg shadow-blue-900/20">Apply Changes</button>
        </div>
      </form>
    </div>
  </div>
</div>

<!-- Kill Confirmation Modal -->
<div id="kill-trader-modal" class="fixed inset-0 z-50 flex items-center justify-center hidden bg-black/60 backdrop-blur-sm">
  <div class="glass border border-red-500/30 rounded-2xl w-full max-w-sm overflow-hidden transform scale-95 opacity-0 transition-all duration-300 shadow-2xl shadow-red-900/20" id="kill-modal-content">
    <div class="p-6 text-center">
      <div class="w-16 h-16 rounded-full bg-red-500/10 border border-red-500/30 flex items-center justify-center mx-auto mb-4 text-red-500 text-2xl">
        <i class="fas fa-exclamation-triangle"></i>
      </div>
      <h3 class="text-xl font-semibold text-white mb-2">Destroy Node?</h3>
      <p class="text-sm text-slate-400 mb-6">Are you sure you want to kill this instance? All historic trade logs associated with it will be permanently obliterated.</p>
      <div class="flex gap-3 justify-center">
        <button type="button" onclick="closeKillModal()" class="flex-1 px-4 py-2.5 rounded-xl border border-slate-600 text-slate-300 hover:bg-slate-700 transition-all font-medium text-sm">Cancel</button>
        <button type="button" onclick="confirmKill()" class="flex-1 px-4 py-2.5 rounded-xl bg-red-600 text-white hover:bg-red-500 transition-all font-medium text-sm shadow-lg shadow-red-900/50">Yes, Kill It</button>
      </div>
    </div>
  </div>
</div>

<!-- Global Toast Notification -->
<div id="global-toast" class="fixed bottom-6 right-6 z-50 transform translate-y-24 opacity-0 transition-all duration-300 pointer-events-none">
  <div class="glass border border-green-500/30 bg-gray-900/90 rounded-2xl p-4 flex items-center gap-4 shadow-2xl shadow-green-900/20">
    <div class="w-8 h-8 rounded-full bg-green-500/20 flex items-center justify-center text-green-400">
      <i class="fas fa-check"></i>
    </div>
    <div>
      <h4 class="text-slate-200 text-sm font-semibold tracking-wide" id="toast-title">Copy Success</h4>
      <p class="text-slate-400 text-xs mt-0.5" id="toast-message">Value copied to clipboard.</p>
    </div>
  </div>
</div>
'''

block_content_end = '{% endblock page_content %}'
if 'edit-trader-modal' not in text:
    text = text.replace(block_content_end, modals_html + '\n' + block_content_end)

# 5. Inject JS logic
js_logic = '''
let targetKillId = null;

function openEditModal(id, cash, goal, acc) {
  const modal = document.getElementById('edit-trader-modal');
  const content = document.getElementById('edit-modal-content');
  const form = document.getElementById('edit-trader-form');
  
  document.getElementById('edit-modal-id-label').textContent = '#T' + id;
  document.getElementById('edit-modal-cash').value = cash || '';
  document.getElementById('edit-modal-goal').value = goal || '';
  document.getElementById('edit-modal-account').value = acc || '';
  
  form.action = '/control/paper-trading/' + id + '/edit/';
  
  modal.classList.remove('hidden');
  setTimeout(() => {
    content.classList.remove('scale-95', 'opacity-0');
    content.classList.add('scale-100', 'opacity-100');
  }, 10);
}

function closeEditModal() {
  const modal = document.getElementById('edit-trader-modal');
  const content = document.getElementById('edit-modal-content');
  content.classList.remove('scale-100', 'opacity-100');
  content.classList.add('scale-95', 'opacity-0');
  setTimeout(() => { modal.classList.add('hidden'); }, 300);
}

function openKillModal(id) {
  targetKillId = id;
  const modal = document.getElementById('kill-trader-modal');
  const content = document.getElementById('kill-modal-content');
  modal.classList.remove('hidden');
  setTimeout(() => {
    content.classList.remove('scale-95', 'opacity-0');
    content.classList.add('scale-100', 'opacity-100');
  }, 10);
}

function closeKillModal() {
  targetKillId = null;
  const modal = document.getElementById('kill-trader-modal');
  const content = document.getElementById('kill-modal-content');
  content.classList.remove('scale-100', 'opacity-100');
  content.classList.add('scale-95', 'opacity-0');
  setTimeout(() => { modal.classList.add('hidden'); }, 300);
}

function confirmKill() {
  if (targetKillId) {
    document.getElementById('kill-form-' + targetKillId).submit();
  }
}

function showToast(title, message) {
  const toast = document.getElementById('global-toast');
  document.getElementById('toast-title').textContent = title;
  document.getElementById('toast-message').textContent = message;
  
  toast.classList.remove('translate-y-24', 'opacity-0');
  
  setTimeout(() => {
    toast.classList.add('translate-y-24', 'opacity-0');
  }, 3000);
}

// Override window.alert for copies if we can, or manually patch templates that copy.
window.showToast = showToast;
'''

block_js_end = '{% endblock extra_js %}'
if 'function openEditModal' not in text:
    text = text.replace(block_js_end, js_logic + '\n' + block_js_end)
    with open('templates/papertrading_fleet.html', 'w', encoding='utf-8') as f:
        f.write(text)

