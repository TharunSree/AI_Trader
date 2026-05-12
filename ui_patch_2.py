import os
import re

target_files = [
    r"d:\AI_Trader\templates\training.html",
    r"d:\AI_Trader\templates\backtest_lab.html",
    r"d:\AI_Trader\templates\model_detail.html",
    r"d:\AI_Trader\templates\evaluation_report.html",
    r"d:\AI_Trader\templates\trader_report.html",
]

for file_path in target_files:
    if not os.path.exists(file_path):
        continue
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Rounded corners
    content = re.sub(r'rounded-2xl', 'rounded', content)
    content = re.sub(r'rounded-xl', 'rounded', content)
    content = re.sub(r'rounded-lg', 'rounded', content)

    # Shadows and gradients
    content = re.sub(r'shadow-\[inset[^\]]+\]', '', content)
    content = re.sub(r'drop-shadow-md', '', content)
    content = re.sub(r'drop-shadow-xl', '', content)
    content = re.sub(r'shadow-sm', '', content)
    content = re.sub(r'hover:shadow-md', '', content)
    content = re.sub(r'hover:scale-\[1\.01\]', '', content)
    content = re.sub(r'bg-gradient-to-br from-brand-[a-zA-Z]+/5 to-transparent', '', content)
    content = re.sub(r'<div class="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity"></div>', '', content)
    content = re.sub(r'<div class="absolute inset-0 transition-opacity"></div>', '', content)
    
    # Borders
    content = re.sub(r'hover:border-brand-primary/[0-9]+', '', content)
    content = re.sub(r'border-brand-primary/[0-9]+', 'border-slate-800', content)
    content = re.sub(r'border-brand-accent/[0-9]+', 'border-slate-800', content)
    content = re.sub(r'border-slate-700/40', 'border-slate-800', content)
    content = re.sub(r'border-slate-700/50', 'border-slate-800', content)
    content = re.sub(r'border-slate-700/60', 'border-slate-800', content)
    content = re.sub(r'border-slate-700', 'border-slate-800', content)

    # Brand Colors to neutral/dense scheme
    # Primary -> Emerald (used for live/success), Accent -> Blue (used for meta/training)
    content = re.sub(r'text-brand-primary', 'text-slate-800 dark:text-slate-200', content)
    content = re.sub(r'text-brand-accent', 'text-slate-800 dark:text-slate-200', content)
    content = re.sub(r'bg-brand-primary/20', 'bg-slate-100 dark:bg-slate-800', content)
    content = re.sub(r'bg-brand-accent/20', 'bg-slate-100 dark:bg-slate-800', content)
    content = re.sub(r'bg-brand-primary/10', 'bg-slate-50 dark:bg-slate-800/50', content)
    content = re.sub(r'bg-brand-accent/10', 'bg-slate-50 dark:bg-slate-800/50', content)
    content = re.sub(r'hover:bg-brand-primary/30', 'hover:bg-slate-200 dark:hover:bg-slate-700', content)
    content = re.sub(r'hover:bg-brand-accent/30', 'hover:bg-slate-200 dark:hover:bg-slate-700', content)
    content = re.sub(r'hover:bg-brand-primary/10', 'hover:bg-slate-100 dark:hover:bg-slate-800', content)
    content = re.sub(r'hover:bg-brand-accent/10', 'hover:bg-slate-100 dark:hover:bg-slate-800', content)
    content = re.sub(r'hover:bg-brand-primary', 'hover:bg-slate-200 dark:hover:bg-slate-700', content)
    content = re.sub(r'hover:bg-brand-accent', 'hover:bg-slate-200 dark:hover:bg-slate-700', content)
    content = re.sub(r'bg-brand-primary', 'bg-slate-800', content)
    content = re.sub(r'bg-brand-accent', 'bg-slate-800', content)

    # Convert large tracking to standard tight text
    content = re.sub(r'tracking-\[0\.2[0-9]em\]', 'tracking-widest', content)
    content = re.sub(r'tracking-\[0\.3[0-9]em\]', 'tracking-widest', content)
    content = re.sub(r'text-xl sm:text-2xl', 'text-sm font-mono uppercase tracking-widest', content)
    content = re.sub(r'text-2xl', 'text-sm font-mono', content)
    
    # Fix dark mode colors that might have been missed or broken
    content = re.sub(r'bg-\[\#0f172a\]', 'bg-slate-950', content)
    content = re.sub(r'bg-black/60', 'bg-slate-950', content)
    content = re.sub(r'bg-black/80', 'bg-slate-950', content)
    
    # Text resizing
    content = re.sub(r'text-lg', 'text-[10px] uppercase tracking-widest', content)
    content = re.sub(r'text-base', 'text-[10px] uppercase tracking-widest', content)

    # Clean up double classes
    content = content.replace('class=" "', 'class=""')

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"Patched {os.path.basename(file_path)}")

print("UI patch complete.")
