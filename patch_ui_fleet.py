import re

with open('templates/papertrading_fleet.html', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''                <div class="flex items-center gap-2 fleet-start-stop-actions">
                  <a href="{% url 'model_detail' t.id %}" class="bg-blue-500/20 text-blue-400 hover:bg-blue-500/30 font-medium px-4 py-2 rounded-xl transition shadow hover:shadow-lg hover:shadow-brand-primary/20">
                    <i class="fas fa-chart-line mr-2"></i>Diagnostics
                  </a>'''

# Ensure we wrap the whole Row in a relative container so the form handles well. Wait, we'll just inject an Edit button that opens a generic prompt, or better yet, since we have an API, we can trigger a JS prompt().
# To keep UI simple and premium without building a massive full-screen React modal, we can use a simple HTML form dropdown or JS prompt to alter the value and post it!

js_script = '''
<script>
function configureTrader(traderId, currentCash, currentGoal, currentAcc) {
    let newCash = prompt("Enter new Initial Cash (Number):", currentCash);
    if(newCash === null) return;
    
    let newGoal = prompt("Enter new Goal Amount (Number or empty):", currentGoal);
    if(newGoal === null) return;
    
    let newAcc = prompt("Enter Database BrokerAccount ID (e.g., 1, 2) or leave empty for .ENV:", currentAcc);
    if(newAcc === null) return;

    let form = document.createElement("form");
    form.method = "POST";
    form.action = "/control/paper-trading/" + traderId + "/edit/";
    
    let csrfInput = document.createElement("input");
    csrfInput.type = "hidden";
    csrfInput.name = "csrfmiddlewaretoken";
    csrfInput.value = document.querySelector('[name=csrfmiddlewaretoken]').value;
    form.appendChild(csrfInput);
    
    let cashInput = document.createElement("input");
    cashInput.type = "hidden";cashInput.name = "initial_cash";cashInput.value = newCash;
    form.appendChild(cashInput);
    
    let goalInput = document.createElement("input");
    goalInput.type = "hidden";goalInput.name = "goal_amount";goalInput.value = newGoal;
    form.appendChild(goalInput);
    
    let accInput = document.createElement("input");
    accInput.type = "hidden";accInput.name = "account_id";accInput.value = newAcc;
    form.appendChild(accInput);
    
    document.body.appendChild(form);
    form.submit();
}
</script>
'''

target_search = '<a href="{% url \'model_detail\' t.id %}" class="bg-blue-500/20 text-blue-400 hover:bg-blue-500/30'
if target_search in text and 'configureTrader' not in text:
    # Inject Edit Button
    btn = '''<button onclick="configureTrader('{{ t.id }}', '{{ t.initial_cash }}', '{{ t.goal_amount|default_if_none:\"\" }}', '{% if t.account %}{{ t.account.id }}{% endif %}')" class="bg-slate-700/50 text-slate-300 hover:bg-slate-700 hover:text-white font-medium px-4 py-2 rounded-xl transition">
    <i class="fas fa-cog mr-2"></i>Edit
</button>\n                  '''
    text = text.replace(target_search, btn + target_search)
    text = text.replace('{% block extra_js %}', '{% block extra_js %}\n' + js_script)

with open('templates/papertrading_fleet.html', 'w', encoding='utf-8') as f:
    f.write(text)

