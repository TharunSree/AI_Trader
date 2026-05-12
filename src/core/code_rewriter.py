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
    from google.genai import types
    GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
except ImportError:
    genai = None
    GEMINI_KEY = None

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

def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY") 
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required to run the cognitive rewriter.")
    return genai.Client(api_key=api_key)

def load_ppo_agent_code():
    agent_path = Path(__file__).resolve().parent.parent / 'models' / 'ppo_agent.py'
    if not agent_path.exists():
        raise FileNotFoundError(f"Could not find {agent_path}")
    return agent_path.read_text(encoding='utf-8')

def rewrite_agent_code(client, d_report, w_report, current_code, crash_log=None, eod_suggestions=None):
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
    
    
    Only rewrite the hyperparameters or the get_action / predict functions. Do NOT change the neural architecture (input/output shape) or the system will crash.
    
    CRITICAL: If the trade reports show very high trade counts (e.g. > 50 trades/day) or a high concentration of day-trades, you MUST implement logic (like higher confidence thresholds or a time-based decay) to reduce churn and avoid Pattern Day Trading (PDT) rejections.
    
    
    Here is the current code:
    ```python
    {current_code}
    ```
    
    Output ONLY valid Python code containing the full updated ppo_agent.py module. Do not use markdown wrappers if possible, but if you do, wrap the whole output exactly once.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
        )
    )
    
    code = response.text
    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
        
    return code.strip() + "\n", response.text

def orchestrate_rewrite(crash_log=None):
    print("[COGNITIVE REWRITER] Booting Sandbox Orchestrator...")
    daily = TradingReport.objects.filter(report_type='DAILY').order_by('-timestamp').first()
    weekly = TradingReport.objects.filter(report_type='WEEKLY').order_by('-timestamp').first()
    
    if not daily and not crash_log:
        print("[COGNITIVE REWRITER] No Daily Report or crash log found. Skipping rewrite iteration.")
        return

    try:
        client = get_gemini_client()
    except Exception as e:
        print(f"[COGNITIVE REWRITER] API Client failed: {e}")
        return

    target_path = Path(__file__).resolve().parent.parent / 'models' / 'ppo_agent.py'
    backup_path = Path(__file__).resolve().parent.parent / 'models' / 'ppo_agent.py.bak'
    
    current_code = load_ppo_agent_code()
    
    # Secure Backup (Sandbox Protocol Phase 1)
    backup_path.write_text(current_code, encoding='utf-8')
    print("[COGNITIVE REWRITER] Pre-flight rollback state captured.")

    def read_report(report_obj):
        if not report_obj:
            return "No report data."
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
            # Take everything up to the next section header
            if '##' in ai_section[1:]:
                ai_section = ai_section[:ai_section.index('##', 1)]
            eod_suggestions = ai_section.strip()
            print(f"[COGNITIVE REWRITER] Injecting {len(eod_suggestions.split())} words of EOD AI directives into mutation prompt.")
        except Exception:
            pass

    try:
        new_code, raw_response = rewrite_agent_code(
            client, 
            daily_content, 
            weekly_content,
            current_code,
            crash_log,
            eod_suggestions
        )
        
        print("\n" + "="*50)
        print("GENERATIVE AI REASONING & CHANGES")
        print("="*50)
        print(raw_response)
        print("="*50 + "\n")
        
        # Sandbox Protocol Phase 2: Syntax Validate BEFORE deploying
        try:
            compile(new_code, 'ppo_agent.py', 'exec')
            print("[COGNITIVE REWRITER] Syntax verification passed.")
        except SyntaxError as syntax_e:
            raise RuntimeError(f"Syntax validation failed. The LLM wrote broken code: {syntax_e}")
        
        # Only write to disk AFTER syntax validation passes
        target_path.write_text(new_code, encoding='utf-8')
        print("[COGNITIVE REWRITER] Mutated logic synced to core.")
            
        # Generate Unified Diff securely
        diff_lines = list(difflib.unified_diff(
            current_code.splitlines(),
            new_code.splitlines(),
            fromfile='ppo_agent.py.bak',
            tofile='ppo_agent.py',
            lineterm=''
        ))
        diff_str = "\n".join(diff_lines)
        
        print("CODEBASE DELTA (UNIFIED DIFF):")
        print("-" * 50)
        print(diff_str)
        print("-" * 50 + "\n")
        
        # Generate Simple Summary for Email
        simple_summary = "Strategy evolved: Thresholds and logic flow optimized based on recent session performance."
        if genai and GEMINI_KEY:
            try:
                sum_res = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=f"Summarize these code changes in 2-3 simple sentences for a non-technical trader:\n\n{diff_str}"
                )
                simple_summary = sum_res.text
            except: pass
        
        pdf_path = generate_mutation_pdf("ppo_agent", raw_response, diff_str, simple_summary)
        
        # Send Notification Mail with PDF
        try:
            from src.reporting.email_dispatcher import send_mutator_alert
            send_mutator_alert(
                f"Mutation Complete for ppo_agent. Logic synced to core.",
                diff_text=None,
                pdf_path=pdf_path,
                simple_summary=simple_summary
            )
        except Exception as mail_err:
            print(f"[COGNITIVE REWRITER] Failed to dispatch mutation email: {mail_err}")
            
        # If this was an Epoch Refinement (not a mid-day crash), automatically trigger a robust training cycle
        if not crash_log:
            try:
                from control_panel.models import TrainingJob
                from control_panel.views import _spawn_background_process
                
                # Spawn diverse training configurations to explore the parameter space
                mutation_configs = [
                    ("Mutant Alpha Discovery", "alpha_discovery", "high_frequency_alpha", 10),
                    ("Mutant Macro Trend", "macro_trend", "long_horizon_growth", 20),
                ]
                
                print("[COGNITIVE REWRITER] Spawning iterative backtesting processes against mutated configurations...")
                for name, feat_key, param_key, window in mutation_configs:
                    job = TrainingJob.objects.create(
                        name=name,
                        feature_set_key=feat_key,
                        hyperparameter_key=param_key,
                        window_size=window, 
                        initial_cash=100000
                    )
                    log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
                    log_dir.mkdir(exist_ok=True)
                    log_file = log_dir / f"train_job_{job.id}.log"
                    process = _spawn_background_process(
                        [sys.executable, "run_training.py", "--job_id", str(job.id)],
                        log_file,
                    )
                    job.celery_task_id = str(process.pid)
                    job.save()
                print("[COGNITIVE REWRITER] Mutation Training Nodes seamlessly isolated and dispatched.")
            except Exception as eval_e:
                print(f"[COGNITIVE REWRITER] Failed to spawn iteration nodes: {eval_e}")

    except Exception as e:
        print(f"[COGNITIVE REWRITER] FATAL ANOMALY: {e}")
        print("[COGNITIVE REWRITER] Executing Rollback Protocol...")
        # Instantly revert
        if backup_path.exists():
            target_path.write_text(backup_path.read_text(encoding='utf-8'), encoding='utf-8')
            print("[COGNITIVE REWRITER] System restored to last known good configuration.")

if __name__ == "__main__":
    orchestrate_rewrite()
