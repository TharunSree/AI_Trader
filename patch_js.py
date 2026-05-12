import re

with open('templates/papertrading_fleet.html', 'r', encoding='utf-8') as f:
    text = f.read()

js_logic = '''
<script>
let targetKillId = null;

function openEditModal(id, cash, goal, acc) {
  const modal = document.getElementById('edit-trader-modal');
  const content = document.getElementById('edit-modal-content');
  const form = document.getElementById('edit-trader-form');
  
  document.getElementById('edit-modal-id-label').textContent = '#T' + id;
  document.getElementById('edit-modal-cash').value = cash || '';
  document.getElementById('edit-modal-goal').value = goal || '';
  // Ensure we select correct environment or fallback to empty string
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

window.showToast = showToast;
</script>
'''

block = '{% endblock extra_javascript %}'
if 'openEditModal' not in text:
    text = text.replace(block, js_logic + '\n' + block)
    with open('templates/papertrading_fleet.html', 'w', encoding='utf-8') as f:
        f.write(text)

