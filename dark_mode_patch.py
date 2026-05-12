import os
import glob
import re

template_dir = r"d:\AI_Trader\templates"
html_files = glob.glob(os.path.join(template_dir, "**/*.html"), recursive=True)

# Skip files we already manually perfected with dark mode
skip_files = ['base.html', 'dashboard.html']

for file_path in html_files:
    filename = os.path.basename(file_path)
    if filename in skip_files:
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content

    # --- Inject Dark Mode Classes safely ---
    
    # Backgrounds
    content = re.sub(r'bg-white(?!\s+dark:)', 'bg-white dark:bg-slate-900', content)
    content = re.sub(r'bg-slate-50(?!\s+dark:)', 'bg-slate-50 dark:bg-slate-800/50', content)
    content = re.sub(r'bg-slate-100(?!\s+dark:)', 'bg-slate-100 dark:bg-slate-800', content)
    
    # Borders
    content = re.sub(r'border-slate-200(?!\s+dark:)', 'border-slate-200 dark:border-slate-800', content)
    
    # Text Colors
    content = re.sub(r'text-slate-800(?!\s+dark:)', 'text-slate-800 dark:text-slate-200', content)
    content = re.sub(r'text-slate-900(?!\s+dark:)', 'text-slate-900 dark:text-white', content)
    content = re.sub(r'text-slate-700(?!\s+dark:)', 'text-slate-700 dark:text-slate-300', content)
    content = re.sub(r'text-slate-600(?!\s+dark:)', 'text-slate-600 dark:text-slate-400', content)
    content = re.sub(r'text-slate-500(?!\s+dark:)', 'text-slate-500 dark:text-slate-400', content)
    content = re.sub(r'text-slate-400(?!\s+dark:)', 'text-slate-400 dark:text-slate-500', content)

    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Patched Dark Mode in {filename}")

print("Dark mode injection complete.")
