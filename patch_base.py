import re

with open('templates/base.html', 'r', encoding='utf-8') as f:
    text = f.read()

messages_html = """
            <!-- Django Messages -->
            {% if messages %}
            <div class="mb-4 space-y-2 z-50 relative">
                {% for message in messages %}
                <div class="px-4 py-3 rounded-lg border {% if message.tags == 'error' %}bg-red-500/10 border-red-500/50 text-red-400{% elif message.tags == 'success' %}bg-green-500/10 border-green-500/50 text-green-400{% else %}bg-brand-primary/10 border-brand-primary/50 text-brand-primary{% endif %} flex justify-between items-center shadow-[0_0_15px_rgba(0,0,0,0.5)] fade-in">
                    <span class="font-mono text-sm">{{ message }}</span>
                    <button type="button" class="text-slate-400 hover:text-white transition-colors" onclick="this.parentElement.remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                {% endendfor %}
            </div>
            {% endif %}
            
            <!-- Decorative Glows -->"""

# Replace the decorative glows to inject messages HTML just above them
text = text.replace("<!-- Decorative Glows -->", messages_html)

# Wait! There's a typo in django template syntax: {% endendfor %} should be {% endfor %}! Let me fix it immediately.
text = text.replace("{% endendfor %}", "{% endfor %}")

with open('templates/base.html', 'w', encoding='utf-8') as f:
    f.write(text)
