import re

with open('templates/model_detail.html', 'r', encoding='utf-8') as f:
    text = f.read()

toast_html = '''
<div id="local-toast" class="fixed bottom-6 right-6 z-50 transform translate-y-24 opacity-0 transition-all duration-300 pointer-events-none">
  <div class="glass border border-green-500/30 bg-gray-900/90 rounded-2xl p-4 flex items-center gap-4 shadow-2xl shadow-green-900/20">
    <div class="w-8 h-8 rounded-full bg-green-500/20 flex items-center justify-center text-green-400">
      <i class="fas fa-check"></i>
    </div>
    <div>
      <h4 class="text-slate-200 text-sm font-semibold tracking-wide">Copy Success</h4>
      <p class="text-slate-400 text-xs mt-0.5" id="local-toast-msg">Trade history copied to clipboard.</p>
    </div>
  </div>
</div>
'''

if 'local-toast' not in text:
    text = text.replace('{% endblock page_content %}', toast_html + '\n{% endblock page_content %}')

js_toast = '''
function showLocalToast(msg) {
  const toast = document.getElementById('local-toast');
  if(toast) {
    document.getElementById('local-toast-msg').textContent = msg;
    toast.classList.remove('translate-y-24', 'opacity-0');
    setTimeout(() => { toast.classList.add('translate-y-24', 'opacity-0'); }, 3000);
  }
}
'''
if 'showLocalToast' not in text:
    text = text.replace('{% endblock extra_js %}', js_toast + '\n{% endblock extra_js %}')

text = text.replace("if(window.showToast) { window.showToast('Notification', 'Copied to clipboard'); } else { alert(", "showLocalToast('Copied to clipboard'); //")

with open('templates/model_detail.html', 'w', encoding='utf-8') as f:
    f.write(text)

