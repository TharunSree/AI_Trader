import re

with open('templates/papertrading_fleet.html', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace("if(form) form.action = '/paper-trading/' + id + '/edit/';", "if(form) form.action = '/dashboard/paper-trading/' + id + '/edit/';")

with open('templates/papertrading_fleet.html', 'w', encoding='utf-8') as f:
    f.write(text)
