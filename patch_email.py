import re

with open('src/reporting/email_dispatcher.py', 'r', encoding='utf-8') as f:
    text = f.read()

old_func = '''def send_mutator_alert(status_message):
    subject = f"dY · JARVIS COGNITIVE MUTATOR ALERT"
    text_content = f"Mutator Diagnostic: {status_message}"
    
    html_content = f\"\"\"
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0f172a; padding: 40px 20px; color: #cbd5e1; line-height: 1.6;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #1e293b; border-radius: 16px; overflow: hidden; border: 1px solid #3b82f6; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);">
            <div style="padding: 35px 30px;">
                <div style="display: flex; align-items: center; margin-bottom: 20px;">
                    <div style="background-color: #8b5cf620; color: #c084fc; padding: 6px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;">
                        Cognitive Optimization
                    </div>
                </div>
                <p style="color: #94a3b8; font-size: 15px; margin-top: 0; margin-bottom: 30px; border-bottom: 1px solid #334155; padding-bottom: 25px;">The Gemini Mutator Engine has concluded an autonomous execution block.</p>
                <div style="background-color: #0f172a; border-radius: 10px; padding: 20px; text-align: center; border: 1px solid #334155;">
                    <div style="color: #f8fafc; font-weight: 600; font-size: 15px;">Diagnostic Result</div>
                    <div style="color: #a78bfa; font-size: 13px; margin-top: 8px;">{status_message}</div>
                </div>
            </div>
        </div>
    </div>
    \"\"\"'''

new_func = '''import html
def send_mutator_alert(status_message, diff_text=None):
    subject = f"dY · JARVIS COGNITIVE MUTATOR ALERT"
    text_content = f"Mutator Diagnostic: {status_message}"
    
    render_diff = ""
    if diff_text:
        # Escape HTML naturally but keep the line breaks
        safe_diff = html.escape(diff_text)
        render_diff = f"""
        <div style="margin-top: 25px; background-color: #020617; border-radius: 10px; padding: 20px; border: 1px solid #334155; text-align: left;">
            <div style="color: #38bdf8; font-weight: 600; font-size: 13px; margin-bottom: 15px; letter-spacing: 1px;">CODEBASE MUTATIONS APPLIED</div>
            <pre style="color: #a5b4fc; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 11px; overflow-x: auto; white-space: pre-wrap; line-height: 1.5; margin: 0;">{safe_diff}</pre>
        </div>"""
        
    html_content = f\"\"\"
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0f172a; padding: 40px 20px; color: #cbd5e1; line-height: 1.6;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #1e293b; border-radius: 16px; overflow: hidden; border: 1px solid #3b82f6; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);">
            <div style="padding: 35px 30px;">
                <div style="display: flex; align-items: center; margin-bottom: 20px;">
                    <div style="background-color: #8b5cf620; color: #c084fc; padding: 6px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;">
                        Cognitive Optimization
                    </div>
                </div>
                <p style="color: #94a3b8; font-size: 15px; margin-top: 0; margin-bottom: 30px; border-bottom: 1px solid #334155; padding-bottom: 25px;">The Gemini Mutator Engine has concluded an autonomous execution block.</p>
                <div style="background-color: #0f172a; border-radius: 10px; padding: 20px; text-align: center; border: 1px solid #334155;">
                    <div style="color: #f8fafc; font-weight: 600; font-size: 15px;">Diagnostic Result</div>
                    <div style="color: #a78bfa; font-size: 13px; margin-top: 8px;">{status_message}</div>
                </div>
                {render_diff}
            </div>
        </div>
    </div>
    \"\"\"'''

text = text.replace(old_func, new_func)
with open('src/reporting/email_dispatcher.py', 'w', encoding='utf-8') as f:
    f.write(text)
