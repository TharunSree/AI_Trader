with open('templates/papertrading.html', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''<form id="start-form" method="post" action="{% url 'start_trader' %}" class="flex flex-col gap-3 mb-3">
        {% csrf_token %}'''

replacement = '''<form id="start-form" method="post" action="{% url 'start_trader' %}" class="flex flex-col gap-3 mb-3">
        {% csrf_token %}
        <select name="account_id" class="bg-slate-800 border border-slate-700 text-brand-primary text-sm rounded block w-full p-2.5 outline-none font-mono mb-2" required {% if trader_status == 'RUNNING' %}disabled{% endif %}>
          <option value="">Select Isolated Broker Account</option>
          {% for acc in broker_accounts %}
          <option value="{{ acc.id }}">{{ acc.name }} {% if acc.is_live %}[LIVE]{% else %}[SANDBOX]{% endif %}</option>
          {% endfor %}
        </select>'''

if "name=\"account_id\"" not in text:
    text = text.replace(target, replacement)
    with open('templates/papertrading.html', 'w', encoding='utf-8') as f:
        f.write(text)
