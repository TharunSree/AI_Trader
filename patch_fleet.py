with open('templates/papertrading_fleet.html', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''        <form method="post" action="{% url 'start_trader' %}" class="w-full flex flex-wrap items-center gap-3" 
id="primary-launch-form" data-jarvis-handled="false">
          {% csrf_token %}'''

# Sometimes formatting makes it fail! I'll use regex.
import re
text = re.sub(
    r'<form method="post" action="\{%\s*url\s+\'start_trader\'\s*%\}".*?>\s*\{%\s*csrf_token\s*%\}',
    '''<form method="post" action="{% url 'start_trader' %}" class="w-full flex flex-wrap items-center gap-3" id="primary-launch-form" data-jarvis-handled="false">
          {% csrf_token %}
          <select name="account_id" class="flex-1 w-full md:w-auto min-w-[200px] bg-slate-900/70 border border-slate-700 text-brand-accent rounded-xl px-4 py-3 font-mono outline-none focus:border-brand-primary truncate" required>
            <option value="">Attach Environment...</option>
            {% for acc in broker_accounts %}
            <option value="{{ acc.id }}">{{ acc.name }} {% if acc.is_live %}[PRODUCTION]{% else %}[SANDBOX]{% endif %}</option>
            {% endfor %}
          </select>''',
    text,
    flags=re.DOTALL | re.MULTILINE
)

with open('templates/papertrading_fleet.html', 'w', encoding='utf-8') as f:
    f.write(text)
