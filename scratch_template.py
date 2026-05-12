import os

content = open('templates/papertrading_fleet.html', 'r', encoding='utf-8').read()

# Replace variables and brand colors
content = content.replace('Live Terminal', 'Live Production Terminal')
content = content.replace('Multi-Model Paper Trading Matrix', 'Live Production Trading Matrix')
content = content.replace('papertrading', 'realtrading')
content = content.replace('Paper Trading', 'Real Money Trading')
content = content.replace('Live Trading Fleet', 'PRODUCTION MONEY TRADING')
content = content.replace('brand-primary', 'red-500')
content = content.replace('brand-accent', 'orange-500')
content = content.replace('text-white font-semibold tracking-wide', 'text-red-100 font-bold tracking-widest uppercase text-shadow')

# Add Safety Interlocks overlay warning at top
warning_banner = """
    <div class="alert alert-danger bg-red-900/50 text-red-200 border-2 border-red-500 rounded-xl p-4 mb-6 sticky top-0 z-50 shadow-[0_0_20px_rgba(239,68,68,0.5)]">
        <div class="flex items-center gap-3">
            <svg class="w-8 h-8 text-red-500 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
            <div>
                <h4 class="font-bold uppercase tracking-widest text-red-400">Live Fire Zone</h4>
                <p class="text-sm font-mono mt-1">WARNING: THIS IS NOT A DRILL. Actions on this page execute against REAL MONEY. You are responsible for all losses.</p>
            </div>
        </div>
    </div>
"""

# Insert banner right inside the page_content
content = content.replace('{% block page_content %}', '{% block page_content %}\n' + warning_banner)

with open('templates/realtrading.html', 'w', encoding='utf-8') as f:
    f.write(content)

print('Template swapped and rewritten.')
