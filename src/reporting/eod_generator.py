import io
import html
import re
import os
from pathlib import Path
from django.utils import timezone
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from django.conf import settings
from django.db.models import Sum, Value, DecimalField
from django.db.models.functions import Coalesce
from control_panel.models import TradeLog, PaperTrader, TradingReport, TrainingJob

try:
    from google import genai
    from google.genai import types
    # Try OS environ first, fallback to Django settings
    GEMINI_KEY = (
        os.environ.get("GEMINI_API_KEY") or 
        getattr(settings, 'GEMINI_API_KEY', None) or 
        getattr(settings, 'API_KEY', None) or 
        ''
    )
    if not GEMINI_KEY:
        import logging as _log
        _log.getLogger("rl_trading_backend").warning(
            "[EOD GENERATOR] GEMINI_API_KEY not found in environment or Django settings. "
            "AI insights will be unavailable. Set GEMINI_API_KEY env var to enable."
        )
except ImportError:
    genai = None
    GEMINI_KEY = None

REPORTS_DIR = Path(settings.BASE_DIR) / "reports"

def _ensure_reports_dir():
    REPORTS_DIR.mkdir(exist_ok=True)
    return REPORTS_DIR

def _render_inline_markdown(text):
    escaped = html.escape(str(text))
    escaped = re.sub(r'`([^`]+)`', r'<code>\1</code>', escaped)
    escaped = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', escaped)
    return escaped

def markdown_to_html(markdown_string):
    lines = markdown_string.splitlines()
    html_parts = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1; continue
        if line == '---':
            html_parts.append('<hr />'); i += 1; continue
        if line.startswith('# '):
            html_parts.append(f"<h1>{_render_inline_markdown(line[2:])}</h1>"); i += 1; continue
        if line.startswith('## '):
            html_parts.append(f"<h2>{_render_inline_markdown(line[3:])}</h2>"); i += 1; continue
        if line.startswith('### '):
            html_parts.append(f"<h3>{_render_inline_markdown(line[4:])}</h3>"); i += 1; continue
        if line.startswith('> '):
            html_parts.append(f"<blockquote>{_render_inline_markdown(line[2:])}</blockquote>"); i += 1; continue
        if line.startswith('|'):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip()); i += 1
            rows = []
            for idx, table_line in enumerate(table_lines):
                cells = [cell.strip() for cell in table_line.strip('|').split('|')]
                if idx == 1 and all(set(cell) <= {'-', ':'} for cell in cells): continue
                rows.append(cells)
            if rows:
                header, body = rows[0], rows[1:]
                table_html = ['<table><thead><tr>']
                table_html.extend(f"<th>{_render_inline_markdown(c)}</th>" for c in header)
                table_html.append('</tr></thead>')
                if body:
                    table_html.append('<tbody>')
                    for row in body:
                        table_html.append('<tr>')
                        table_html.extend(f"<td>{_render_inline_markdown(c)}</td>" for c in row)
                        table_html.append('</tr>')
                    table_html.append('</tbody>')
                table_html.append('</table>')
                html_parts.append(''.join(table_html))
            continue
        if line.startswith('- '):
            list_items = []
            while i < len(lines) and lines[i].strip().startswith('- '):
                list_items.append(lines[i].strip()[2:]); i += 1
            html_parts.append('<ul>' + ''.join(f"<li>{_render_inline_markdown(item)}</li>" for item in list_items) + '</ul>')
            continue
        paragraph_lines = [line]; i += 1
        while i < len(lines):
            candidate = lines[i].strip()
            if not candidate or candidate.startswith(('#', '>', '|', '- ')) or candidate == '---': break
            paragraph_lines.append(candidate); i += 1
        html_parts.append(f"<p>{_render_inline_markdown(' '.join(paragraph_lines))}</p>")
    return ''.join(html_parts)

def fetch_daily_metrics(report_date=None, target_bot=None):
    from datetime import timedelta
    today = timezone.now().date()
    end_date = timezone.now()
    start_date = end_date - timedelta(hours=24)
    
    model_name = "Isolated Fleet (No Activity)"
    try:
        if target_bot and target_bot.model_file:
            if str(target_bot.model_file).startswith('db:'):
                job_id = str(target_bot.model_file).split(':')[1]
                tj = TrainingJob.objects.get(id=job_id)
                model_name = tj.name
            else:
                model_name = Path(target_bot.model_file).stem
    except Exception:
        pass  # model_name stays as default

    # Get recent trades exclusively for this target bot over the past exact 24 hours
    trades = TradeLog.objects.filter(trader=target_bot, timestamp__gte=start_date).order_by('-timestamp')
    total_trades = trades.count()
    zero_money = Value(0, output_field=DecimalField(max_digits=20, decimal_places=2))
    buy_volume = float(trades.filter(action='BUY').aggregate(total=Coalesce(Sum('notional_value'), zero_money))['total'])
    sell_volume = float(trades.filter(action='SELL').aggregate(total=Coalesce(Sum('notional_value'), zero_money))['total'])
    net_flow = sell_volume - buy_volume
    symbols_traded = sorted({t.symbol for t in trades})
    
    yesterday_start = start_date - timedelta(hours=24)
    yesterday_trades = TradeLog.objects.filter(trader=target_bot, timestamp__gte=yesterday_start, timestamp__lt=start_date)
    y_buy = float(yesterday_trades.filter(action='BUY').aggregate(total=Coalesce(Sum('notional_value'), zero_money))['total'])
    y_sell = float(yesterday_trades.filter(action='SELL').aggregate(total=Coalesce(Sum('notional_value'), zero_money))['total'])
    yesterday_net_flow = y_sell - y_buy
    
    week_start = end_date - timedelta(days=7)
    week_trades = TradeLog.objects.filter(trader=target_bot, timestamp__gte=week_start)
    w_buy = float(week_trades.filter(action='BUY').aggregate(total=Coalesce(Sum('notional_value'), zero_money))['total'])
    w_sell = float(week_trades.filter(action='SELL').aggregate(total=Coalesce(Sum('notional_value'), zero_money))['total'])
    week_net_flow = w_sell - w_buy

    month_start = end_date - timedelta(days=30)
    month_trades = TradeLog.objects.filter(trader=target_bot, timestamp__gte=month_start)
    m_buy = float(month_trades.filter(action='BUY').aggregate(total=Coalesce(Sum('notional_value'), zero_money))['total'])
    m_sell = float(month_trades.filter(action='SELL').aggregate(total=Coalesce(Sum('notional_value'), zero_money))['total'])
    month_net_flow = m_sell - m_buy

    ai_analysis = "No AI Analysis available."
    session_summary = "Standard analytical data available in logs."
    
    if genai and GEMINI_KEY:
        import time
        try:
            client = genai.Client(api_key=GEMINI_KEY)
            
            symbols_str = ", ".join(symbols_traded[:10]) if symbols_traded else "None"
            
            # High-fidelity 300-400 word diagnostic
            diag_prompt = (
                f"You are an elite quantitative analyst and AI systems engineer reviewing a live autonomous trading system. "
                f"Provide a HIGH-FIDELITY, extremely comprehensive professional diagnostic report (300-400 words) for the trading model '{model_name}'.\n\n"
                f"SESSION METRICS:\n"
                f"- Today's Net PnL: ${net_flow:,.2f}\n"
                f"- Total Trades Executed: {total_trades}\n"
                f"- Buy Volume: ${buy_volume:,.2f} | Sell Volume: ${sell_volume:,.2f}\n"
                f"- Symbols Traded: {symbols_str}\n"
                f"- Yesterday's PnL: ${yesterday_net_flow:,.2f}\n"
                f"- 7-Day Cumulative PnL: ${week_net_flow:,.2f}\n\n"
                "Your report MUST include these sections:\n"
                "1. Deep Performance Diagnostic - Analyze win rates, capital efficiency, and risk exposure\n"
                "2. Strategic Roadmap - Concrete recommendations for the next trading session\n"
                "3. Production Readiness Assessment - Is the model fit for continued autonomous operation?\n"
                "Maintain a precise, technical tone. Do NOT use markdown headers or formatting."
            )
            ai_analysis = client.models.generate_content(model='gemini-2.5-flash', contents=diag_prompt).text
            
            # Wait to respect rate limits
            time.sleep(5)
            
            # 300-400 word session narrative
            sum_prompt = (
                f"Write a professional executive mission summary (300-400 words) for an autonomous trading session.\n\n"
                f"Date: {report_date or today}\n"
                f"Model: {model_name}\n"
                f"Net Profit/Loss: ${net_flow:,.2f}\n"
                f"Trades: {total_trades} across {symbols_str}\n"
                f"Buy Volume: ${buy_volume:,.2f} | Sell Volume: ${sell_volume:,.2f}\n\n"
                "Cover: market conditions observed, strategy execution quality, notable patterns, "
                "and operational status. Write in a narrative style suitable for a CEO briefing. "
                "Do NOT use markdown headers or bullet points - write flowing paragraphs."
            )
            session_summary = client.models.generate_content(model='gemini-2.5-flash', contents=sum_prompt).text
        except Exception as e:
            import traceback
            err_str = traceback.format_exc()
            print(f"[EOD GENERATOR] Gemini AI generation failed:\n{err_str}")
            ai_analysis = f"**AI Analysis Generation Failed**\n\nThe AI system was unable to generate insights for this session due to an error. This is usually caused by an invalid/missing GEMINI_API_KEY, network issues, or API rate limits.\n\n`{e}`"
            session_summary = f"Session summary generation failed. See AI Analysis section for details."

    return {
        'date': report_date or today,
        'total_trades': total_trades,
        'buy_volume': buy_volume,
        'sell_volume': sell_volume,
        'net_flow': net_flow,
        'total_pnl': net_flow,
        'symbols_traded': symbols_traded,
        'trades': trades,
        'model_name': model_name,
        'hist_yesterday': yesterday_net_flow,
        'hist_7d': week_net_flow,
        'hist_30d': month_net_flow,
        'ai_analysis': ai_analysis,
        'session_summary': session_summary
    }

def generate_markdown_report_string(metrics):
    win_proxy = (metrics['sell_volume'] / max(metrics['buy_volume'], 1)) * 100
    
    # Balance descriptor
    if metrics['buy_volume'] > metrics['sell_volume'] * 1.5:
        balance_desc = "Buy-heavy"
    elif metrics['sell_volume'] > metrics['buy_volume'] * 1.5:
        balance_desc = "Sell-heavy"
    else:
        balance_desc = "Balanced"
    
    # Capital recycling
    if win_proxy > 80:
        recycling = "Strong capital rotation"
    elif win_proxy > 40:
        recycling = "Moderate recycling"
    else:
        recycling = "Potentially sticky inventory"
    
    # Inline progress bar HTML for exit efficiency
    bar_w = max(win_proxy, 1)
    bar_html = (
        f"<table style='width: 100px; margin: 0; padding: 0; border: none; background-color: #313244; "
        f"display: inline-table; vertical-align: middle;'><tr>"
        f"<td style='width: {bar_w}%; background-color: #a6e3a1; padding: 3px 0; border: none;'></td>"
        f"<td style='width: {100-bar_w}%; border: none;'></td>"
        f"</tr></table> <span style='color: #a6e3a1;'>{win_proxy:.1f}%</span>"
    )

    md = f"# {metrics['model_name']}\n"
    md += f"> Jarvis End-of-Day Intelligence Report\n\n"
    md += f"**Session Date:** {metrics['date']}  \n"
    md += f"**Model Reference:** `{metrics['model_name']}`  \n"
    md += f"**Primary Symbols:** {metrics['symbols_traded']}\n\n"
    md += f"---\n\n"
    
    md += f"## Mission Summary\n{metrics['session_summary']}\n\n"
    md += f"## Generative AI Insights & Diagnostics\n{metrics['ai_analysis']}"
    
    md += f"## Session Scorecard\n"
    md += f"| Metric | Value |\n| --- | ---: |\n"
    md += f"| Total Trades | **{metrics['total_trades']}** |\n"
    md += f"| Buy Volume | **${metrics['buy_volume']:,.2f}** |\n"
    md += f"| Sell Volume | **${metrics['sell_volume']:,.2f}** |\n"
    md += f"| Net Capital Flow | **${metrics['net_flow']:,.2f}** |\n"
    md += f"| Exit Efficiency Proxy | {bar_html} |\n"
    md += f"| Buy/Sell Balance | **{balance_desc}** |\n"
    md += f"| Capital Recycling | **{recycling}** |\n"
    md += f"| Yesterday's Edge | **${metrics['hist_yesterday']:,.2f}** |\n"
    md += f"| 7-Day Net Yield | **${metrics['hist_7d']:,.2f}** |\n"
    md += f"| 30-Day Net Yield | **${metrics['hist_30d']:,.2f}** |\n\n\n"
    
    md += "## Trade Execution Matrix\n"
    md += "| Time (UTC) | Action | Symbol | Quantity | Execution Price | Notional Value |\n"
    md += "| --- | --- | --- | --- | --- | --- |\n"
    for t in metrics['trades'][:30]:
        md += f"| {t.timestamp.strftime('%H:%M:%S')} | **{t.action}** | {t.symbol} | {t.quantity:,.4f} | ${t.price:,.2f} | ${t.notional_value:,.2f} |\n"
    
    md += "\n---\n## Operator Notes\n"
    md += "- Review whether fills are clustering around volatile intervals.\n"
    md += "- Compare `TradeLog.notional_value` against expected spread-adjusted fills from the broker.\n"
    md += "- If repeated micro-churn appears, tighten confidence thresholds before market-order routing.\n"
    return md

class JarvisReportPDF(FPDF):
    """Catppuccin Frappe themed PDF report with JetBrains Mono."""
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
    """Strip Unicode characters that Helvetica can't render (smart quotes, em-dashes, etc.)"""
    replacements = {
        '\u2018': "'", '\u2019': "'",  # Smart single quotes
        '\u201c': '"', '\u201d': '"',  # Smart double quotes
        '\u2013': '-', '\u2014': '--', # En/em dashes
        '\u2026': '...', '\u2022': '*', # Ellipsis, bullet
        '\u00b7': '*',  # Middle dot
        '\u2032': "'", '\u2033': '"',  # Prime marks
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Final pass: remove any remaining non-latin1 chars
    return text.encode('latin-1', errors='replace').decode('latin-1')

def generate_pdf_bytes_native(metrics):
    pdf = JarvisReportPDF()
    pdf.add_page()
    # Catppuccin Frappe palette
    mauve   = (202, 158, 230)  # #ca9ee6
    blue    = (140, 170, 238)  # #8caaee
    teal    = (129, 200, 190)  # #81c8be
    text_color = (198, 208, 245)  # #c6d0f5
    subtext = (165, 173, 206)  # #a5adce
    surface0 = (65, 69, 89)    # #414559
    mantle  = (41, 44, 60)     # #292c3c
    f = pdf.default_font
    pdf.set_font(f, "B", 20); pdf.set_text_color(*mauve)
    pdf.cell(0, 12, metrics['model_name'].title(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(f, "I", 10); pdf.set_text_color(*subtext)
    pdf.cell(0, 6, "Jarvis End-of-Day Intelligence Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    def draw_hdr(title, color):
        pdf.set_font(f, "B", 12); pdf.set_text_color(*color)
        pdf.cell(0, 8, title, border="B", new_x=XPos.LMARGIN, new_y=YPos.NEXT); pdf.ln(3)

    draw_hdr("Mission Summary", blue)
    pdf.set_font(f, "", 10)
    pdf.set_text_color(*text_color)
    sess_sum = _sanitize_for_pdf(metrics.get('session_summary', 'Detailed analytical data available in logs.'))
    pdf.multi_cell(0, 5, sess_sum)
    pdf.ln(5)

    # --- AI Analysis ---
    draw_hdr("Generative AI Insights & Diagnostics", mauve)
    pdf.set_font(f, "", 9)
    pdf.set_text_color(*text_color)
    ai_diag = _sanitize_for_pdf(metrics.get('ai_analysis', 'No AI Analysis available.').replace('#', '').strip())
    pdf.multi_cell(0, 5, ai_diag)
    pdf.ln(5)

    # --- Scorecard (as a proper table like Monday's format) ---
    draw_hdr("Session Scorecard", teal)
    win_proxy = (metrics['sell_volume'] / max(metrics['buy_volume'], 1)) * 100
    
    # Balance & recycling descriptors
    if metrics['buy_volume'] > metrics['sell_volume'] * 1.5:
        balance_desc = "Buy-heavy"
    elif metrics['sell_volume'] > metrics['buy_volume'] * 1.5:
        balance_desc = "Sell-heavy"
    else:
        balance_desc = "Balanced"
    
    if win_proxy > 80:
        recycling = "Strong capital rotation"
    elif win_proxy > 40:
        recycling = "Moderate recycling"
    else:
        recycling = "Potentially sticky inventory"
    
    scorecard_items = [
        ("Total Trades", str(metrics['total_trades'])),
        ("Buy Volume", f"${metrics['buy_volume']:,.2f}"),
        ("Sell Volume", f"${metrics['sell_volume']:,.2f}"),
        ("Net Capital Flow", f"${metrics['net_flow']:,.2f}"),
        ("Exit Efficiency Proxy", f"{win_proxy:.1f}%"),
        ("Buy/Sell Balance", balance_desc),
        ("Capital Recycling", recycling),
        ("Yesterday's Edge", f"${metrics['hist_yesterday']:,.2f}"),
        ("7-Day Net Yield", f"${metrics['hist_7d']:,.2f}"),
        ("30-Day Net Yield", f"${metrics['hist_30d']:,.2f}"),
    ]
    
    col_metric_w = 80
    col_value_w = 100
    row_h = 7
    
    # Table header row
    pdf.set_fill_color(*mantle)
    pdf.set_draw_color(*surface0)
    pdf.set_font(f, "B", 9); pdf.set_text_color(*teal)
    pdf.cell(col_metric_w, row_h, "  Metric", border=1, fill=True)
    pdf.cell(col_value_w, row_h, "  Value", border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    # Table data rows with alternating stripe
    for i, (label, val) in enumerate(scorecard_items):
        if i % 2 == 0:
            pdf.set_fill_color(*surface0)
        else:
            pdf.set_fill_color(48, 52, 70)  # Base (no fill contrast)
        
        pdf.set_font(f, "", 9); pdf.set_text_color(*subtext)
        pdf.cell(col_metric_w, row_h, "  " + label, border=1, fill=True)
        pdf.set_font(f, "B", 9); pdf.set_text_color(*text_color)
        pdf.cell(col_value_w, row_h, "  " + val, border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    # --- Trade Execution Matrix (proper table with header + alternating rows) ---
    draw_hdr("Trade Execution Matrix", teal)
    green  = (166, 209, 137)  # Frappe Green #a6d189
    red    = (231, 130, 132)  # Frappe Red #e78284
    
    # Column widths matching Monday's format
    cw = [24, 16, 22, 26, 35, 37]  # Time, Action, Symbol, Quantity, Exec Price, Notional
    headers = ["Time (UTC)", "Action", "Symbol", "Quantity", "Exec. Price", "Notional Value"]
    
    # Header row
    pdf.set_fill_color(*mantle)
    pdf.set_draw_color(*surface0)
    pdf.set_font(f, "B", 8); pdf.set_text_color(*teal)
    for j, hdr in enumerate(headers):
        last = (j == len(headers) - 1)
        pdf.cell(cw[j], row_h, " " + hdr, border=1, fill=True,
                 new_x=XPos.LMARGIN if last else XPos.RIGHT,
                 new_y=YPos.NEXT if last else YPos.TOP)
    
    # Data rows
    for i, t in enumerate(metrics['trades'][:40]):
        if pdf.get_y() > 268:
            pdf.add_page()
            # Repeat header on new page
            pdf.set_fill_color(*mantle)
            pdf.set_font(f, "B", 8); pdf.set_text_color(*teal)
            for j, hdr in enumerate(headers):
                last = (j == len(headers) - 1)
                pdf.cell(cw[j], row_h, " " + hdr, border=1, fill=True,
                         new_x=XPos.LMARGIN if last else XPos.RIGHT,
                         new_y=YPos.NEXT if last else YPos.TOP)
        
        # Alternating row background
        if i % 2 == 0:
            pdf.set_fill_color(*surface0)
        else:
            pdf.set_fill_color(48, 52, 70)  # Base
        fill = True
        
        # Time
        pdf.set_font(f, "", 8); pdf.set_text_color(*text_color)
        pdf.cell(cw[0], 6, " " + t.timestamp.strftime('%H:%M:%S'), border=1, fill=fill)
        
        # Action (color-coded)
        if t.action.upper() == 'BUY':
            pdf.set_font(f, "B", 8); pdf.set_text_color(*green)
        else:
            pdf.set_font(f, "B", 8); pdf.set_text_color(*red)
        pdf.cell(cw[1], 6, " " + t.action, border=1, fill=fill)
        
        # Symbol, Quantity, Price, Notional
        pdf.set_font(f, "", 8); pdf.set_text_color(*text_color)
        pdf.cell(cw[2], 6, " " + t.symbol, border=1, fill=fill)
        pdf.cell(cw[3], 6, " " + f"{t.quantity:.4f}", border=1, fill=fill)
        pdf.cell(cw[4], 6, " " + f"${t.price:.2f}", border=1, fill=fill)
        pdf.cell(cw[5], 6, " " + f"${t.notional_value:.2f}", border=1, fill=fill,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    return bytes(pdf.output())

def write_report_artifacts(report_date=None):
    from datetime import timedelta
    date_obj = report_date or timezone.now().date()
    _ensure_reports_dir()
    
    # Use a 24-hour rolling window to find active bots, avoiding timezone date-boundary issues
    end = timezone.now()
    start = end - timedelta(hours=24)
    bot_ids = set(TradeLog.objects.filter(timestamp__gte=start).values_list('trader_id', flat=True))
    
    # Fallback: also check for exact date match (handles both UTC and local TZ)
    if not bot_ids:
        bot_ids = set(TradeLog.objects.filter(timestamp__date=date_obj).values_list('trader_id', flat=True))
    
    latest_bots = PaperTrader.objects.filter(id__in=bot_ids)
    if not latest_bots.exists():
        latest_bots = PaperTrader.objects.filter(status='RUNNING')
    
    artifacts = []
    for bot in latest_bots:
        metrics = fetch_daily_metrics(report_date=date_obj, target_bot=bot)
        md_text = generate_markdown_report_string(metrics)
        pdf_bytes = generate_pdf_bytes_native(metrics)
        
        fname = f"report_{date_obj.strftime('%Y%m%d')}_bot{bot.id}"
        md_path = REPORTS_DIR / f"{fname}.md"
        pdf_path = REPORTS_DIR / f"{fname}.pdf"
        md_path.write_text(md_text, encoding='utf-8')
        pdf_path.write_bytes(pdf_bytes)
        
        TradingReport.objects.create(
            report_type='DAILY', markdown_path=str(md_path), pdf_path=str(pdf_path),
            total_revenue=metrics['net_flow'], total_trades=metrics['total_trades'],
            win_rate=(metrics['sell_volume']/max(metrics['buy_volume'], 1))*100
        )
        artifacts.append({"bot_id": bot.id, "md_path": md_path, "pdf_path": pdf_path, "pdf_bytes": pdf_bytes})
    return artifacts

