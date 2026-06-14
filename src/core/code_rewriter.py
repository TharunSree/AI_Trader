import os
import json
import traceback
import argparse
import sys
import difflib
from pathlib import Path

# Django setup MUST happen before any Django imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trader_project.settings')

import django
django.setup()

# Now safe to import Django modules
from django.conf import settings
from django.utils import timezone
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from control_panel.models import TradingReport

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

try:
    import openai as openai_mod
except ImportError:
    openai_mod = None

try:
    import anthropic as anthropic_mod
except ImportError:
    anthropic_mod = None

class MutationReportPDF(FPDF):
    """Catppuccin Frappe themed mutation report with JetBrains Mono."""
    def __init__(self):
        super().__init__()
        font_dir = Path(settings.BASE_DIR) / "static" / "fonts" / "fonts" / "ttf"
        if (font_dir / "JetBrainsMono-Regular.ttf").exists():
            self.add_font("JBMono", "", str(font_dir / "JetBrainsMono-Regular.ttf"), uni=True)
            self.add_font("JBMono", "B", str(font_dir / "JetBrainsMono-Bold.ttf"), uni=True)
            self.add_font("JBMono", "I", str(font_dir / "JetBrainsMono-Italic.ttf"), uni=True)
            self.default_font = "JBMono"
        else:
            self.default_font = "helvetica"
    
    def add_page(self, *args, **kwargs):
        super().add_page(*args, **kwargs)
        # Catppuccin Frappe Base: #303446
        self.set_fill_color(48, 52, 70)
        self.rect(0, 0, 210, 297, 'F')

def _sanitize_for_pdf(text):
    """Strip Unicode characters that Helvetica can't render."""
    replacements = {
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '--', '\u2026': '...', '\u2022': '*',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode('latin-1', errors='replace').decode('latin-1')

def generate_mutation_pdf(model_name, reasoning, diff_str, simple_summary):
    pdf = MutationReportPDF()
    pdf.add_page()
    f = pdf.default_font
    
    # Catppuccin Frappe palette
    mauve    = (202, 158, 230)  # #ca9ee6
    blue     = (140, 170, 238)  # #8caaee
    teal     = (129, 200, 190)  # #81c8be
    green    = (166, 209, 137)  # #a6d189
    lavender = (186, 187, 241)  # #babbf1
    text_color = (198, 208, 245)  # #c6d0f5
    subtext  = (165, 173, 206)  # #a5adce
    surface0 = (65, 69, 89)    # #414559
    mantle   = (41, 44, 60)    # #292c3c
    
    def draw_hdr(title, color):
        pdf.set_font(f, "B", 12); pdf.set_text_color(*color)
        pdf.cell(0, 8, title, border="B", new_x=XPos.LMARGIN, new_y=YPos.NEXT); pdf.ln(3)
    
    # Title
    pdf.set_font(f, "B", 20)
    pdf.set_text_color(*mauve)
    pdf.cell(0, 12, f"Strategy Evolution Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font(f, "I", 10)
    pdf.set_text_color(*subtext)
    pdf.cell(0, 6, f"Model: {model_name} | {timezone.now().strftime('%B %d, %Y at %I:%M %p')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(8)
    
    # --- Section 1: What Changed (Plain English) ---
    draw_hdr("What Changed (In Plain English)", green)
    pdf.set_font(f, "", 10)
    pdf.set_text_color(*text_color)
    pdf.multi_cell(0, 5, _sanitize_for_pdf(simple_summary))
    pdf.ln(6)
    
    # --- Section 2: Why Jarvis Made These Changes ---
    draw_hdr("Why Jarvis Made These Changes", blue)
    pdf.set_font(f, "", 9)
    pdf.set_text_color(*text_color)
    
    # Clean up the reasoning - strip markdown formatting, make it readable
    clean_reasoning = reasoning.replace('#', '').replace('**', '').replace('```python', '').replace('```', '').strip()
    # Only take the reasoning part, not the full code output
    if 'import ' in clean_reasoning and 'class ' in clean_reasoning:
        # The raw_response likely includes the full code - extract just the reasoning
        parts = clean_reasoning.split('import ', 1)
        clean_reasoning = parts[0].strip() if parts[0].strip() else "Analysis complete. See code delta below for specific changes."
    
    pdf.multi_cell(0, 4.5, _sanitize_for_pdf(clean_reasoning[:2000]))
    pdf.ln(6)
    
    # --- Section 3: Code Changes (In a Box) ---
    draw_hdr("Code Changes (Technical Detail)", mauve)
    pdf.ln(2)
    
    # Draw a code box with Surface0 background
    x_start = pdf.get_x()
    y_start = pdf.get_y()
    
    # Calculate how much space is left on this page
    remaining = 270 - y_start
    
    pdf.set_font(f, "", 6.5)
    pdf.set_text_color(*lavender)
    
    # Draw the background box
    pdf.set_fill_color(*surface0)
    
    # Split diff into lines and limit to what fits
    diff_lines = _sanitize_for_pdf(diff_str).split('\n')
    
    # Render diff line by line inside the box
    box_x = pdf.l_margin
    box_w = pdf.w - pdf.l_margin - pdf.r_margin
    line_h = 3.2
    
    # First pass: calculate total height needed
    total_lines = min(len(diff_lines), 80)  # Cap at 80 lines
    box_h = (total_lines * line_h) + 8  # padding
    
    if y_start + box_h > 280:
        pdf.add_page()
        y_start = pdf.get_y()
    
    # Draw the box background
    pdf.set_fill_color(*surface0)
    pdf.rect(box_x, y_start, box_w, min(box_h, 240), 'F')
    
    # Draw a thin border
    pdf.set_draw_color(*lavender)
    pdf.rect(box_x, y_start, box_w, min(box_h, 240), 'D')
    
    # Position inside the box
    pdf.set_xy(box_x + 3, y_start + 3)
    
    for i, line in enumerate(diff_lines[:total_lines]):
        if pdf.get_y() > 275:
            pdf.add_page()
            pdf.set_fill_color(*surface0)
            pdf.rect(box_x, pdf.get_y(), box_w, 100, 'F')
            pdf.set_xy(box_x + 3, pdf.get_y() + 2)
        
        # Color code diff lines
        if line.startswith('+') and not line.startswith('+++'):
            pdf.set_text_color(*green)   # Added lines in green
        elif line.startswith('-') and not line.startswith('---'):
            pdf.set_text_color(231, 130, 132)  # Removed lines in red
        elif line.startswith('@@'):
            pdf.set_text_color(*mauve)   # Hunk headers in mauve
        else:
            pdf.set_text_color(*lavender)
        
        pdf.cell(box_w - 6, line_h, line[:120], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(box_x + 3)
    
    reports_dir = Path(settings.BASE_DIR) / "reports"
    reports_dir.mkdir(exist_ok=True)
    pdf_path = reports_dir / f"mutation_{model_name}_{timezone.now().strftime('%H%M%S')}.pdf"
    pdf.output(str(pdf_path))
    return str(pdf_path)

def _read_key(name):
    """Read an API key from SystemSettings database, fallback to env vars or Django settings."""
    def clean_val(v):
        if not v:
            return ''
        v_str = str(v).strip()
        if v_str.upper() in ['', 'YOUR_API_KEY', 'YOUR_GEMINI_API_KEY', 'YOUR_SECRET_KEY', 'YOUR_KEY', 'API_KEY_HERE']:
            return ''
        return v_str

    val = ''
    try:
        from control_panel.models import SystemSettings
        db_settings = SystemSettings.load()
        val = getattr(db_settings, name.lower(), '')
        val = clean_val(val)
        print(f"[DEBUG _read_key] Loaded from DB '{name.lower()}': {'SET' if val else 'EMPTY'}")
    except Exception as e:
        print(f"[DEBUG _read_key] SystemSettings DB error: {e}")
        pass
        
    if not val:
        val = clean_val(os.environ.get(name, ''))
    if not val:
        val = clean_val(getattr(settings, name, ''))
        
    return val


class _UnifiedAIClient:
    """Wraps Gemini / OpenAI / Anthropic behind a single .generate(prompt) interface."""

    def __init__(self):
        self.provider = None
        self._client = None

        # Try Gemini first
        gemini_key = _read_key("GEMINI_API_KEY")
        if gemini_key and genai:
            try:
                self._client = genai.Client(api_key=gemini_key)
                self.provider = 'gemini'
                print(f"[AI ENGINE] Using Gemini (google-genai)")
                return
            except Exception as e:
                print(f"[AI ENGINE] Gemini init failed: {e}")

        # Try OpenAI
        openai_key = _read_key("OPENAI_API_KEY")
        if openai_key and openai_mod:
            try:
                self._client = openai_mod.OpenAI(api_key=openai_key)
                self.provider = 'openai'
                print(f"[AI ENGINE] Using OpenAI")
                return
            except Exception as e:
                print(f"[AI ENGINE] OpenAI init failed: {e}")

        # Try Anthropic
        anthropic_key = _read_key("ANTHROPIC_API_KEY")
        if anthropic_key and anthropic_mod:
            try:
                self._client = anthropic_mod.Anthropic(api_key=anthropic_key)
                self.provider = 'anthropic'
                print(f"[AI ENGINE] Using Anthropic Claude")
                return
            except Exception as e:
                print(f"[AI ENGINE] Anthropic init failed: {e}")

        debug_msg = (
            f"Diagnostics: "
            f"Gemini(key={'SET' if gemini_key else 'EMPTY'}, mod={'LOADED' if genai else 'MISSING'}) | "
            f"OpenAI(key={'SET' if openai_key else 'EMPTY'}, mod={'LOADED' if openai_mod else 'MISSING'}) | "
            f"Anthropic(key={'SET' if anthropic_key else 'EMPTY'}, mod={'LOADED' if anthropic_mod else 'MISSING'})"
        )
        
        # REST API Fallback for Gemini if the package is missing but the key is set
        if gemini_key and not genai:
            self.provider = 'gemini'
            self._gemini_api_key = gemini_key
            print(f"[AI ENGINE] Using Gemini (REST API Fallback)")
            return
            
        raise ValueError(f"No AI API key or module found! {debug_msg}")

    def generate(self, prompt, temperature=0.4):
        """Generate text from any provider. Returns the raw response text."""
        if self.provider == 'gemini':
            if hasattr(self, '_gemini_api_key'):
                import requests
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self._gemini_api_key}"
                headers = {"Content-Type": "application/json"}
                data = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": temperature}
                }
                resp = requests.post(url, headers=headers, json=data)
                if resp.status_code != 200:
                    raise RuntimeError(f"Gemini API Error: {resp.text}")
                return resp.json()['candidates'][0]['content']['parts'][0]['text']
            else:
                response = self._client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(temperature=temperature),
                )
                return response.text

        elif self.provider == 'openai':
            response = self._client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content

        elif self.provider == 'anthropic':
            response = self._client.messages.create(
                model='claude-sonnet-4-20250514',
                max_tokens=8192,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        raise RuntimeError(f"Unknown provider: {self.provider}")


def get_ai_client():
    """Get a unified AI client — tries Gemini > OpenAI > Anthropic in order."""
    return _UnifiedAIClient()


# Legacy alias so existing callers still work
get_gemini_client = get_ai_client

def load_ppo_agent_code():
    agent_path = Path(__file__).resolve().parent.parent / 'models' / 'ppo_agent.py'
    if not agent_path.exists():
        raise FileNotFoundError(f"Could not find {agent_path}")
    return agent_path.read_text(encoding='utf-8')

def rewrite_agent_code(client, d_report, w_report, current_code, crash_log=None, eod_suggestions=None, failed_context=None):
    suggestions_block = ""
    if eod_suggestions:
        suggestions_block = f"""
    IMPORTANT - AI-GENERATED OPTIMIZATION DIRECTIVES FROM EOD ANALYSIS:
    The following suggestions were generated by the Jarvis EOD Diagnostic Engine. 
    You MUST implement these recommendations in your rewritten code:
    ---
    {eod_suggestions}
    ---
    """
    
    prompt = f"""
    You are an elite quantitative researcher and PyTorch engineer managing a live trading system.
    The system just completed an interval or encountered a fatal crash. You are acting as the Autonomous Cognitive Rewriter.
    
    Here is the daily performance report:
    {d_report}
    
    Here is the trailing 7-day performance report:
    {w_report}
    
    {suggestions_block}
    
    The trading logic is implemented in the PPOAgent class using PyTorch. 
    Analyze the performance, identify what isn't working (e.g. max drawdown is too high, it's trading too frequently, etc), and rewrite the PyTorch logic to optimize for the current market regime. 

    {'CRITICAL: THE SYSTEM JUST CRASHED. FIX THE FOLLOWING TRACEBACK IN YOUR NEW CODE:' if crash_log else ''}
    {crash_log if crash_log else ''}
    
    {failed_context if failed_context else ''}
    
    Only rewrite the hyperparameters or the get_action / predict functions. Do NOT change the neural architecture (input/output shape) or the system will crash.
    
    CRITICAL: If the trade reports show very high trade counts (e.g. > 50 trades/day) or a high concentration of day-trades, you MUST implement logic (like higher confidence thresholds or a time-based decay) to reduce churn and avoid Pattern Day Trading (PDT) rejections.
    
    CRITICAL IMPLEMENTATION RULES:
    1. Do NOT hardcode return values. Under no circumstances should the `predict()` method return a constant value (such as always returning `0.0`). It must be a fully dynamic PyTorch neural network forward-pass that computes and returns output actions based on the input state array and the current model weights.
    2. Ensure the state array is correctly converted to a PyTorch tensor, passed through the model's actor network, and clamped to [-1.0, 1.0].
    
    
    Here is the current code:
    ```python
    {current_code}
    ```
    
    Output ONLY valid Python code containing the full updated ppo_agent.py module. Do not use markdown wrappers if possible, but if you do, wrap the whole output exactly once.
    """
    
    raw_response = client.generate(prompt, temperature=0.4)
    
    code = raw_response
    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
        
    return code.strip() + "\n", raw_response

def _snapshot_parent_cash(parent_trader):
    """
    Get the parent trader's current computed balance (initial_cash + realized_profit).
    This is what the challenger variant inherits.
    """
    from collections import defaultdict
    if not parent_trader:
        return 100.0
    
    trades = parent_trader.trades.all()
    total_bought = sum(float(t.notional_value) for t in trades if t.action == 'BUY')
    total_sold = sum(float(t.notional_value) for t in trades if t.action == 'SELL')
    realized = total_sold - total_bought
    initial = float(parent_trader.initial_cash or 0)
    
    current_balance = max(0.0, initial + realized)
    print(f"[NEURAL EVOLUTION] Parent T#{parent_trader.id} balance snapshot: "
          f"initial=${initial:.2f} + realized=${realized:+.2f} = ${current_balance:.2f}")
    return current_balance


def _enforce_spawn_guard(max_active=3):
    """
    Spawn Guard: cap concurrent testing variants at max_active.
    If at capacity, terminate the worst performer (lowest virtual_pnl_pct).
    Returns True if a slot is available after enforcement.
    """
    from control_panel.models import ModelVariant
    import psutil
    import os
    import signal
    
    # 1. Self-healing/clean-up: Prune dead testing processes from database status
    testing_variants = ModelVariant.objects.filter(status='TESTING')
    for v in testing_variants:
        if v.celery_task_id:
            try:
                pid = int(v.celery_task_id)
                if not psutil.pid_exists(pid):
                    print(f"[NEURAL EVOLUTION] Detected dead virtual engine process (PID {pid}) for Variant #{v.id}. Pruning slot.")
                    
                    # Create failure alert
                    try:
                        from control_panel.models import SystemAlert
                        alert_msg = f"**Variant Name**: {v.name}\n"
                        alert_msg += f"**Failure Type**: PROCESS DEAD (Crashed or Terminated)\n"
                        alert_msg += f"**Error**: PID {pid} is no longer running.\n\n"
                        
                        alert_msg += "### 📊 Trade Performance Summary\n"
                        alert_msg += f"- **Total Trades**: {v.virtual_trades_count}\n"
                        alert_msg += f"- **Win Rate**: {v.win_rate:.1f}%\n"
                        alert_msg += f"- **Starting Balance**: ${float(v.starting_cash):,.2f}\n"
                        alert_msg += f"- **Final Balance**: ${float(v.virtual_balance):,.2f}\n"
                        alert_msg += f"- **Net P/L**: {v.virtual_pnl_pct:+.2f}%\n"
                        alert_msg += f"- **Sharpe Ratio**: {v.sharpe_ratio:.4f}\n"
                        alert_msg += f"- **Max Drawdown**: {v.max_drawdown_pct:.2f}%\n\n"
                        
                        if v.mutation_reasoning:
                            alert_msg += f"### 💡 Attempted Rationale\n{v.mutation_reasoning}\n\n"
                        if v.agent_code:
                            alert_msg += f"### 💻 Agent Code\n```python\n{v.agent_code}\n```\n"
                            
                        SystemAlert.objects.create(
                            level='WARNING',
                            title=f'🧬 Variant #{v.id} Failed: Process Dead',
                            message=alert_msg,
                            related_model_reference=str(v.id)
                        )
                    except Exception as alert_err:
                        print(f"[NEURAL EVOLUTION] Failed to log dead process alert: {alert_err}")
                        
                    v.delete()
            except Exception as e:
                print(f"[NEURAL EVOLUTION] Error checking process status for Variant #{v.id}: {e}")

    active = ModelVariant.objects.filter(status='TESTING').order_by('virtual_pnl_pct')
    count = active.count()
    
    if count < max_active:
        print(f"[NEURAL EVOLUTION] Spawn Guard: {count}/{max_active} slots occupied. Slot available.")
        return True
    
    # 2. Free up slot: Kill the worst performer to make room
    worst = active.first()
    if worst:
        print(f"[NEURAL EVOLUTION] Spawn Guard: {count}/{max_active} slots full. "
              f"Terminating worst performer: Variant #{worst.id} '{worst.name}' "
              f"(PnL: {worst.virtual_pnl_pct:+.2f}%)")
              
        # Stop its running process
        if worst.celery_task_id:
            try:
                pid = int(worst.celery_task_id)
                if psutil.pid_exists(pid):
                    os.kill(pid, signal.SIGTERM)
                    print(f"[NEURAL EVOLUTION] Successfully terminated PID {pid} for Variant #{worst.id}")
            except Exception as kill_err:
                print(f"[NEURAL EVOLUTION] Failed to kill PID {worst.celery_task_id} for Variant #{worst.id}: {kill_err}")
                
        # Create termination alert
        try:
            from control_panel.models import SystemAlert
            alert_msg = f"**Variant Name**: {worst.name}\n"
            alert_msg += f"**Failure Type**: TERMINATED BY SPAWN GUARD\n"
            alert_msg += f"**Reason**: Terminated by spawn guard to make room for a new variant (worst performance: {worst.virtual_pnl_pct:+.2f}%).\n\n"
            
            alert_msg += "### 📊 Trade Performance Summary\n"
            alert_msg += f"- **Total Trades**: {worst.virtual_trades_count}\n"
            alert_msg += f"- **Win Rate**: {worst.win_rate:.1f}%\n"
            alert_msg += f"- **Starting Balance**: ${float(worst.starting_cash):,.2f}\n"
            alert_msg += f"- **Final Balance**: ${float(worst.virtual_balance):,.2f}\n"
            alert_msg += f"- **Net P/L**: {worst.virtual_pnl_pct:+.2f}%\n"
            alert_msg += f"- **Sharpe Ratio**: {worst.sharpe_ratio:.4f}\n"
            alert_msg += f"- **Max Drawdown**: {worst.max_drawdown_pct:.2f}%\n\n"
            
            if worst.mutation_reasoning:
                alert_msg += f"### 💡 Attempted Rationale\n{worst.mutation_reasoning}\n\n"
            if worst.agent_code:
                alert_msg += f"### 💻 Agent Code\n```python\n{worst.agent_code}\n```\n"
                
            SystemAlert.objects.create(
                level='WARNING',
                title=f'🧬 Variant #{worst.id} Failed: Terminated by Spawn Guard',
                message=alert_msg,
                related_model_reference=str(worst.id)
            )
        except Exception as alert_err:
            print(f"[NEURAL EVOLUTION] Failed to log spawn guard termination alert: {alert_err}")
            
        worst.delete()
    
    return True


def orchestrate_rewrite(crash_log=None, force=False):
    print("[NEURAL EVOLUTION] Booting Mutation Orchestrator...")
    daily = TradingReport.objects.filter(report_type='DAILY').order_by('-timestamp').first()
    weekly = TradingReport.objects.filter(report_type='WEEKLY').order_by('-timestamp').first()
    
    # Mutator Timing Constraint: Only mutate if a report was generated recently (last 3 hours)
    if not crash_log and not force:
        if not daily:
            print("[NEURAL EVOLUTION] No daily report found. Skipping mutation run.")
            return
        from django.utils import timezone
        import datetime
        age = timezone.now() - daily.timestamp
        if age > datetime.timedelta(hours=3):
            print(f"[NEURAL EVOLUTION] Latest daily report is too old ({age.total_seconds()/3600:.2f} hours ago). Skipping mutation run.")
            return
            
    # If no reports exist, generate a live performance snapshot from trade logs
    if not daily and not crash_log:
        try:
            from control_panel.models import PaperTrader
            running = PaperTrader.objects.filter(status__in=['RUNNING', 'SLEEPING']).first()
            if running:
                trades = running.trades.all()
                total_bought = sum(float(t.notional_value) for t in trades if t.action == 'BUY')
                total_sold = sum(float(t.notional_value) for t in trades if t.action == 'SELL')
                trade_count = trades.count()
                pnl = total_sold - total_bought
                initial = float(running.initial_cash or 0)
                
                synthetic_report = (
                    f"Live Performance Snapshot (No EOD report available)\n"
                    f"Trader #{running.id} | Model: {running.model_file}\n"
                    f"Initial Capital: ${initial:.2f}\n"
                    f"Total Trades: {trade_count}\n"
                    f"Capital Spent (Buys): ${total_bought:.2f}\n"
                    f"Capital Recovered (Sells): ${total_sold:.2f}\n"
                    f"Realized PnL: ${pnl:+.2f}\n"
                    f"Status: The model needs optimization — it is not generating consistent profits.\n"
                )
                print(f"[NEURAL EVOLUTION] No daily report found. Generated synthetic snapshot from {trade_count} trades.")
                # Create a temporary fake daily for the rest of the flow
                class FakeReport:
                    report_type = 'SYNTHETIC'
                    markdown_path = None
                    _content = synthetic_report
                daily = FakeReport()
            else:
                print("[NEURAL EVOLUTION] No running traders and no reports. Skipping.")
                return
        except Exception as e:
            print(f"[NEURAL EVOLUTION] Could not generate synthetic report: {e}")
            return

    try:
        client = get_gemini_client()
    except Exception as e:
        print(f"[NEURAL EVOLUTION] API Client failed: {e}")
        return

    target_path = Path(__file__).resolve().parent.parent / 'models' / 'ppo_agent.py'
    backup_path = Path(__file__).resolve().parent.parent / 'models' / 'ppo_agent.py.bak'
    
    current_code = load_ppo_agent_code()
    
    # Secure Backup (still kept as a filesystem safety net)
    backup_path.write_text(current_code, encoding='utf-8')
    print("[NEURAL EVOLUTION] Pre-flight rollback state captured.")

    def read_report(report_obj):
        if not report_obj:
            return "No report data."
        # Handle synthetic reports from live trade data
        if hasattr(report_obj, '_content'):
            return report_obj._content
        try:
            path = Path(report_obj.markdown_path)
            if path.exists():
                return path.read_text(encoding='utf-8')
            return f"Report file not found: {path}"
        except Exception as e:
            return f"Error reading report: {e}"

    daily_content = read_report(daily)
    weekly_content = read_report(weekly)
    
    # Extract AI suggestions from the latest EOD report to auto-implement
    eod_suggestions = None
    if daily_content and 'Generative AI Insights' in daily_content:
        try:
            ai_section = daily_content.split('## Generative AI Insights')[1]
            if '##' in ai_section[1:]:
                ai_section = ai_section[:ai_section.index('##', 1)]
            eod_suggestions = ai_section.strip()
            print(f"[NEURAL EVOLUTION] Injecting {len(eod_suggestions.split())} words of EOD AI directives into mutation prompt.")
        except Exception:
            pass

    # Collect recently failed or manually rejected variants to avoid repeating errors/bad logic
    from control_panel.models import SystemAlert
    from django.db.models import Q
    import datetime
    negative_cutoff = timezone.now() - datetime.timedelta(days=14)
    negative_alerts = SystemAlert.objects.filter(
        Q(level='WARNING') & Q(title__contains='Variant') &
        (Q(title__contains='Rejected') | Q(title__contains='Failed')),
        created_at__gte=negative_cutoff,
        related_model_reference__isnull=False
    ).order_by('-id')[:5]
    failed_context = ""
    
    if negative_alerts:
        failed_context = "\nCRITICAL NEGATIVE CONTEXT: The following recently generated model variant(s) failed or were manually rejected. You MUST review their code, reasoning rationale, and failure/rejection reasons (including performance metrics and user feedback comments), and ensure your new code does NOT repeat the same mistakes, patterns, or decisions:\n"
        for alert in negative_alerts:
            if "Rejected" in alert.title:
                failed_context += f"--- Manually Rejected Variant Audit Log ({alert.title}) ---\n"
            else:
                failed_context += f"--- Failed/Crashed Variant Audit Log ({alert.title}) ---\n"
            failed_context += f"{alert.message}\n"
            failed_context += "----------------------------------------\n"

    try:
        new_code, raw_response = rewrite_agent_code(
            client, 
            daily_content, 
            weekly_content,
            current_code,
            crash_log,
            eod_suggestions,
            failed_context
        )
        
        print("\n" + "="*50)
        print("GENERATIVE AI REASONING & CHANGES")
        print("="*50)
        print(raw_response[:2000])  # Truncate for log readability
        print("="*50 + "\n")
        
        # Sandbox Protocol: Syntax Validate
        try:
            compile(new_code, 'ppo_agent.py', 'exec')
            print("[NEURAL EVOLUTION] Syntax verification passed.")
        except SyntaxError as syntax_e:
            raise RuntimeError(f"Syntax validation failed. The LLM wrote broken code: {syntax_e}")
        
        # Sandbox Protocol: Runtime Smoke Test
        # Catches NameError/TypeError hallucinations that compile() misses
        try:
            import types as _types
            import numpy as np
            _sandbox_module = _types.ModuleType("sandbox_smoke_test")
            exec(compile(new_code, 'ppo_agent_smoke_test.py', 'exec'), _sandbox_module.__dict__)
            _SandboxAgent = getattr(_sandbox_module, 'PPOAgent', None)
            if _SandboxAgent:
                # Initialize agent
                agent_inst = _SandboxAgent(state_dim=50, action_dim=1)
                
                # Verify predict() does not return constant zeros
                test_states = [np.random.randn(50).astype(np.float32) for _ in range(10)]
                outputs = []
                for state in test_states:
                    out = agent_inst.predict(state)
                    outputs.append(out)
                
                print(f"[NEURAL EVOLUTION] Smoke test predictions: {outputs}")
                if all(abs(out) < 1e-9 for out in outputs):
                    raise RuntimeError("Evolved model returned flat 0.0 outputs for all inputs. Zero signals detected.")
                
                del agent_inst
            del _sandbox_module
            print("[NEURAL EVOLUTION] Runtime smoke test passed.")
        except Exception as runtime_e:
            raise RuntimeError(f"Runtime validation failed — AI hallucinated broken logic: {runtime_e}")
        
        # === NEURAL EVOLUTION: Create variant instead of overwriting ===
        # DO NOT write to ppo_agent.py — the production file stays untouched!
        
        # Generate Unified Diff
        diff_lines = list(difflib.unified_diff(
            current_code.splitlines(),
            new_code.splitlines(),
            fromfile='ppo_agent.py (current)',
            tofile='ppo_agent.py (mutant)',
            lineterm=''
        ))
        diff_str = "\n".join(diff_lines)
        
        print("CODEBASE DELTA (UNIFIED DIFF):")
        print("-" * 50)
        print(diff_str[:3000])
        print("-" * 50 + "\n")
        
        # Generate Simple Summary
        simple_summary = "Strategy evolved: Thresholds and logic flow optimized based on recent session performance."
        try:
            simple_summary = client.generate(
                f"Summarize these code changes in 2-3 simple sentences for a non-technical trader:\n\n{diff_str}",
                temperature=0.3
            )
        except Exception:
            pass
        
        # Generate mutation PDF report
        pdf_path = generate_mutation_pdf("ppo_agent", raw_response, diff_str, simple_summary)
        
        # Send email notification
        try:
            from src.reporting.email_dispatcher import send_mutator_alert
            send_mutator_alert(
                f"Neural Evolution: New variant spawned for 14-day trial.",
                diff_text=None,
                pdf_path=pdf_path,
                simple_summary=simple_summary
            )
        except Exception as mail_err:
            print(f"[NEURAL EVOLUTION] Failed to dispatch mutation email: {mail_err}")
        
        # === Spawn Guard: ensure we don't exceed max concurrent variants ===
        _enforce_spawn_guard(max_active=3)
        
        # === Find the parent PaperTrader and snapshot its current cash ===
        from control_panel.models import PaperTrader, ModelVariant
        
        # Pick the first running paper trader as the parent (benchmark)
        parent_trader = PaperTrader.objects.filter(
            status__in=['RUNNING', 'SLEEPING']
        ).first()
        
        parent_cash = _snapshot_parent_cash(parent_trader)
        
        # === Create the ModelVariant record ===
        variant_name = f"Evolution {timezone.now().strftime('%b %d %H:%M')}"
        if crash_log:
            variant_name = f"Crash Recovery {timezone.now().strftime('%b %d %H:%M')}"
        
        variant = ModelVariant.objects.create(
            name=variant_name,
            status='TESTING',
            parent_trader=parent_trader,
            agent_code=new_code,
            starting_cash=parent_cash,
            virtual_balance=parent_cash,
            mutation_reasoning=raw_response[:5000],
            diff_summary=diff_str[:10000],
        )
        
        print(f"[NEURAL EVOLUTION] Created Variant #{variant.id}: '{variant_name}' "
              f"with starting cash ${parent_cash:.2f}")
        
        # === Create a SystemAlert for dashboard notification ===
        from control_panel.models import SystemAlert
        SystemAlert.objects.create(
            level='INFO',
            title=f'Neural Evolution: Variant #{variant.id} Spawned',
            message=f'{simple_summary}\n\nStarting 14-day virtual paper trading trial with ${parent_cash:.2f} starting capital.',
            related_model_reference=str(variant.id),
        )
        
        # === Auto-spawn the virtual paper trading engine ===
        try:
            from control_panel.views import _spawn_background_process
            log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"evolution_variant_{variant.id}.log"
            
            process = _spawn_background_process(
                [sys.executable, "-m", "src.core.virtual_paper_engine", "--variant_id", str(variant.id)],
                log_file,
            )
            variant.celery_task_id = str(process.pid)
            variant.save(update_fields=['celery_task_id'])
            
            print(f"[NEURAL EVOLUTION] Virtual engine spawned (PID: {process.pid}) "
                  f"for Variant #{variant.id}. 14-day trial begins now.")
        except Exception as spawn_err:
            print(f"[NEURAL EVOLUTION] Failed to spawn virtual engine: {spawn_err}")
            # Create failure alert
            try:
                from control_panel.models import SystemAlert
                alert_msg = f"**Variant Name**: {variant.name}\n"
                alert_msg += f"**Failure Type**: SPAWN ERROR\n"
                alert_msg += f"**Error Exception**: {str(spawn_err)}\n\n"
                if variant.mutation_reasoning:
                    alert_msg += f"### 💡 Attempted Rationale\n{variant.mutation_reasoning}\n\n"
                if variant.agent_code:
                    alert_msg += f"### 💻 Agent Code\n```python\n{variant.agent_code}\n```\n"
                
                SystemAlert.objects.create(
                    level='WARNING',
                    title=f'🧬 Variant #{variant.id} Failed: Spawn Error',
                    message=alert_msg,
                    related_model_reference=str(variant.id)
                )
            except Exception as alert_err:
                print(f"[NEURAL EVOLUTION] Failed to log spawn failure alert: {alert_err}")
            
            variant.delete()

    except Exception as e:
        print(f"[NEURAL EVOLUTION] FATAL ANOMALY: {e}")
        import traceback
        tb_str = traceback.format_exc()
        try:
            from control_panel.models import SystemAlert
            alert_msg = f"**Mutation Run Failed**\n"
            alert_msg += f"**Error**: {str(e)}\n\n"
            alert_msg += f"### 🔴 Traceback\n```\n{tb_str}\n```\n\n"
            if 'raw_response' in locals() and raw_response:
                alert_msg += f"### 💡 Attempted Rationale\n{raw_response[:5000]}\n\n"
            if 'new_code' in locals() and new_code:
                alert_msg += f"### 💻 Agent Code\n```python\n{new_code}\n```\n"
            
            SystemAlert.objects.create(
                level='WARNING',
                title=f'🧬 Mutation Attempt Failed: {str(e)[:60]}',
                message=alert_msg
            )
        except Exception as alert_err:
            print(f"[NEURAL EVOLUTION] Failed to log failure alert: {alert_err}")

if __name__ == "__main__":
    import sys
    force_flag = "--force" in sys.argv
    orchestrate_rewrite(force=force_flag)

