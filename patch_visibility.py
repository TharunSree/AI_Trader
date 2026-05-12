import re

with open('templates/papertrading_fleet.html', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace Tailwind modal hiding with inline style logic
text = text.replace('class="fixed inset-0 z-50 flex items-center justify-center hidden bg-black/60 backdrop-blur-sm"', 'class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" style="display: none;"')

# Replace exact script functions to use style.display
script_old = '''function openEditModal(id, cash, goal, acc) {
  const modal = document.getElementById('edit-trader-modal');
  const content = document.getElementById('edit-modal-content');
  const form = document.getElementById('edit-trader-form');
  
  if(document.getElementById('edit-modal-id-label')) document.getElementById('edit-modal-id-label').textContent = '#T' + id;
  if(document.getElementById('edit-modal-cash')) document.getElementById('edit-modal-cash').value = cash || '';
  if(document.getElementById('edit-modal-goal')) document.getElementById('edit-modal-goal').value = goal || '';
  if(document.getElementById('edit-modal-account')) document.getElementById('edit-modal-account').value = acc || '';
  
  if(form) form.action = '/control/paper-trading/' + id + '/edit/';
  
  if(modal) modal.classList.remove('hidden');'''

script_new = '''function openEditModal(id, cash, goal, acc) {
  const modal = document.getElementById('edit-trader-modal');
  const content = document.getElementById('edit-modal-content');
  const form = document.getElementById('edit-trader-form');
  
  if(document.getElementById('edit-modal-id-label')) document.getElementById('edit-modal-id-label').textContent = '#T' + id;
  if(document.getElementById('edit-modal-cash')) document.getElementById('edit-modal-cash').value = cash || '';
  if(document.getElementById('edit-modal-goal')) document.getElementById('edit-modal-goal').value = goal || '';
  if(document.getElementById('edit-modal-account')) document.getElementById('edit-modal-account').value = acc || '';
  
  if(form) form.action = '/paper-trading/' + id + '/edit/';
  
  if(modal) modal.style.display = 'flex';'''

text = text.replace(script_old, script_new)
text = text.replace('if(modal) modal.classList.add(\'hidden\');', 'if(modal) modal.style.display = \'none\';')
text = text.replace('modal.classList.remove(\'hidden\');', 'modal.style.display = \'flex\';')

# And toast fallback
text = text.replace('toast.classList.remove(\'translate-y-24\', \'opacity-0\');', 'toast.style.transform = \'translateY(0)\'; toast.style.opacity = \'1\';')
text = text.replace('toast.classList.add(\'translate-y-24\', \'opacity-0\');', 'toast.style.transform = \'translateY(6rem)\'; toast.style.opacity = \'0\';')

with open('templates/papertrading_fleet.html', 'w', encoding='utf-8') as f:
    f.write(text)
