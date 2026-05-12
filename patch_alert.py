import re

with open('templates/model_detail.html', 'r', encoding='utf-8') as f:
    text = f.read()

target = "alert('Log Copied!');"
replacement = "if(window.showToast) { window.showToast('Log Copied', 'Trade history copied to clipboard'); } else { alert('Log Copied!'); }"

if target in text:
    text = text.replace(target, replacement)
elif "alert('Copied" in text:
    text = re.sub(r"alert\('[^']+'\);", "if(window.showToast) { window.showToast('Copied', 'Data copied to clipboard'); } else { alert('Copied!'); }", text)
else:
    # Let's search for alert
    text = text.replace("alert(", "if(window.showToast) { window.showToast('Notification', 'Copied to clipboard'); } else { alert(")

with open('templates/model_detail.html', 'w', encoding='utf-8') as f:
    f.write(text)

