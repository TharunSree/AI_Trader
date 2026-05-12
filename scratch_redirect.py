with open('control_panel/views.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace("return redirect('papertrading')", "return redirect(request.META.get('HTTP_REFERER', 'papertrading'))")

with open('control_panel/views.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Redirects replaced.')
