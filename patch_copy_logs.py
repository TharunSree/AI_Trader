import re

with open('templates/model_detail.html', 'r', encoding='utf-8') as f:
    text = f.read()

# Modify button HTML to pass 	his and give it an ID
text = text.replace('onclick="copyLogs()"', 'id="copy-logs-btn" onclick="copyLogs(this)"')

# Rewrite the JS logic to change the button color to green and text to COPIED!
script_old = '''window.copyLogs = function() {
        navigator.clipboard.writeText(rawLogBuffer).then(() => {
            showLocalToast('Copied to clipboard'); //'Trace buffer copied to clipboard!');
        });
    };'''

script_new = '''window.copyLogs = function(btn) {
        navigator.clipboard.writeText(rawLogBuffer).then(() => {
            if(!btn) btn = document.getElementById('copy-logs-btn');
            if(btn) {
                const originalHtml = btn.innerHTML;
                const originalColor = btn.className;
                
                // Set green text
                btn.innerHTML = '<svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg> COPIED';
                btn.className = "text-xs uppercase tracking-widest text-emerald-400 font-mono flex items-center gap-1 transition-colors";
                
                setTimeout(() => {
                    btn.innerHTML = originalHtml;
                    btn.className = originalColor;
                }, 2000);
            }
        });
    };'''

text = text.replace(script_old, script_new)

with open('templates/model_detail.html', 'w', encoding='utf-8') as f:
    f.write(text)
