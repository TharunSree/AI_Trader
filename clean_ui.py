import os
import glob
import re

template_dir = r"d:\AI_Trader\templates"
html_files = glob.glob(os.path.join(template_dir, "**/*.html"), recursive=True)

for file_path in html_files:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content

    # --- CSS Replacements (Converting Dark/Glass to Light/Solid) ---
    content = content.replace("class=\"glass", "class=\"bg-white border border-slate-200 shadow-sm")
    content = content.replace("glass-panel", "bg-white border border-slate-200 shadow-sm")
    
    # Text Colors
    content = content.replace("text-white", "text-slate-800")
    content = content.replace("text-slate-400", "text-slate-500")
    content = content.replace("text-slate-300", "text-slate-600")
    content = content.replace("text-slate-200", "text-slate-700")
    
    # Backgrounds
    content = content.replace("bg-slate-900/80", "bg-white")
    content = content.replace("bg-slate-900/50", "bg-slate-50")
    content = content.replace("bg-slate-800/80", "bg-white")
    content = content.replace("bg-slate-800", "bg-slate-100")
    content = content.replace("bg-slate-900", "bg-white")
    
    # Holographic Shadows & Effects
    content = re.sub(r'drop-shadow-\[0_0_[0-9]+px_rgba\([0-9,.]+\)\]', '', content)
    content = re.sub(r'hover:shadow-\[0_0_[0-9]+px_rgba\([0-9,.]+\)\]', 'hover:shadow-md', content)
    content = content.replace("shadow-[0_0_10px_rgba(16,185,129,0.8)]", "shadow-sm")
    content = content.replace("shadow-[0_0_10px_rgba(59,130,246,0.8)]", "shadow-sm")

    # --- Terminology Scrub ---
    content = content.replace("Jarvis Matrix", "QuantTrader Pro")
    content = content.replace("JARVIS MATRIX", "QUANTTRADER PRO")
    content = content.replace("Jarvis", "QuantTrader")
    content = content.replace("JARVIS", "QUANTTRADER")
    content = content.replace("Matrix", "System")
    content = content.replace("MATRIX", "SYSTEM")

    # --- Javascript Updates ---
    content = content.replace("launchJarvisOverlay", "launchOverlay")
    content = content.replace("launchJarvisShutdown", "launchOverlay")
    content = content.replace("data-jarvis-handled", "data-overlay-handled")
    content = content.replace("jarvisHandled", "overlayHandled")
    content = content.replace("confirmJarvisAction", "confirmAction")
    
    # Define a simple confirmAction for templates that used confirmJarvisAction
    if "confirmAction(" in content and "function confirmAction" not in content:
        # Just inject it at the bottom if needed, though base.html handles it
        pass

    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated {os.path.basename(file_path)}")

print("Scrub complete.")
