import os
from django.core.mail import EmailMultiAlternatives
from django.conf import settings

def send_sos_alert(model_name, traceback_str):
    subject = f"🛑 JARVIS EXCEPTION: {model_name}"
    text_content = f"Execution interrupted.\n\nTraceback:\n{traceback_str}"
    
    html_content = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #030712; padding: 40px 20px; color: #cbd5e1; line-height: 1.6;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #0f172a; border-radius: 16px; overflow: hidden; border: 1px solid #312e81; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);">
            <!-- Accent Header -->
            <div style="background: linear-gradient(135deg, #ef4444 0%, #991b1b 100%); padding: 25px 30px; text-align: left;">
                <span style="background-color: rgba(255, 255, 255, 0.15); color: #ffffff; padding: 4px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;">Node Alert</span>
                <h1 style="color: #ffffff; margin: 10px 0 0 0; font-size: 20px; font-weight: 800; letter-spacing: 0.5px;">SYSTEM EXCEPTION LOGGED</h1>
            </div>
            
            <div style="padding: 30px 25px;">
                <p style="color: #9ca3af; font-size: 14px; margin-top: 0; margin-bottom: 25px;">The execution thread on the neural core was abruptly interrupted by an unhandled state failure. Critical diagnostic parameters below:</p>
                
                <div style="background-color: #1e293b; border-radius: 10px; padding: 16px; margin-bottom: 25px; border: 1px solid #334155;">
                    <span style="color: #64748b; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; display: block; font-family: monospace;">Target Isolation Node</span>
                    <strong style="color: #f8fafc; font-size: 15px; font-family: monospace; display: block; margin-top: 4px;">{model_name}</strong>
                </div>

                <div style="background-color: #020617; padding: 20px; border-radius: 10px; overflow-x: auto; border: 1px solid #1e293b;">
                    <span style="color: #475569; font-size: 10px; text-transform: uppercase; letter-spacing: 1.5px; display: block; margin-bottom: 10px; font-family: monospace; font-weight: bold;">Exception Traceback</span>
                    <code style="color: #fca5a5; font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; white-space: pre-wrap; display: block; line-height: 1.5;">{traceback_str}</code>
                </div>
                
                <div style="margin-top: 30px; background-color: #ef444410; border: 1px solid #ef444425; border-radius: 8px; padding: 14px; text-align: center;">
                    <p style="font-size: 11px; color: #fca5a5; font-weight: 600; margin: 0; font-family: monospace;">Autonomic self-rewriter protocol is reviewing the stack trace for self-repair triggers.</p>
                </div>
            </div>
        </div>
        <div style="text-align: center; margin-top: 25px; font-size: 10px; color: #4b5563; font-family: monospace; text-transform: uppercase; letter-spacing: 1px;">
            Jarvis Quant System Orchestrator &bull; Encrypted Channel
        </div>
    </div>
    """
    
    try:
        recipient = os.environ.get('EMAIL_RECIPIENT', 'tarunsree@gmail.com')
        msg = EmailMultiAlternatives(
            subject, text_content, 
            os.environ.get('EMAIL_HOST_USER', 'noreply@jarvis-trading.ai'),
            [recipient]
        )
        msg.attach_alternative(html_content, "text/html")
        msg.send(fail_silently=False)
        return True
    except Exception as e:
        print(f"Failed to dispatch SOS Email: {e}")
        return False

def send_report_email(report_type, markdown_content, pdf_bytes=None, filename="report.pdf"):
    subject = f"Jarvis Insight // {report_type.upper()}"
    text_content = markdown_content
    
    html_content = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #030712; padding: 40px 20px; color: #cbd5e1; line-height: 1.6;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #0f172a; border-radius: 16px; overflow: hidden; border: 1px solid #1e3a8a; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);">
            <!-- Accent Header -->
            <div style="background: linear-gradient(135deg, #1d4ed8 0%, #1e3a8a 100%); padding: 25px 30px; text-align: left;">
                <span style="background-color: rgba(255, 255, 255, 0.15); color: #ffffff; padding: 4px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;">Intelligence Packet</span>
                <h1 style="color: #ffffff; margin: 10px 0 0 0; font-size: 20px; font-weight: 800; letter-spacing: 0.5px;">{report_type.upper()} REPORT GENERATED</h1>
            </div>
            
            <div style="padding: 30px 25px;">
                <p style="color: #9ca3af; font-size: 14px; margin-top: 0; margin-bottom: 25px;">Automated algorithmic market intelligence telemetry has been successfully compiled and archived.</p>
                
                <div style="background-color: #1b2735; border-radius: 12px; padding: 30px 20px; text-align: center; border: 1px solid #283e56; margin-bottom: 10px;">
                    <div style="font-size: 36px; margin-bottom: 12px;">📊</div>
                    <h3 style="color: #f8fafc; font-weight: 700; font-size: 15px; margin: 0 0 8px 0;">Telemetry Document Attached</h3>
                    <p style="color: #9ca3af; font-size: 12px; margin: 0; font-family: monospace;">A print-ready PDF performance ledger is attached securely to this transmission.</p>
                </div>
            </div>
        </div>
        <div style="text-align: center; margin-top: 25px; font-size: 10px; color: #4b5563; font-family: monospace; text-transform: uppercase; letter-spacing: 1px;">
            Jarvis Quant System Orchestrator &bull; Encrypted Channel
        </div>
    </div>
    """
    
    try:
        recipient = os.environ.get('EMAIL_RECIPIENT', 'tarunsree@gmail.com')
        msg = EmailMultiAlternatives(
            subject, text_content, 
            os.environ.get('EMAIL_HOST_USER', 'noreply@jarvis-trading.ai'),
            [recipient]
        )
        msg.attach_alternative(html_content, "text/html")
        if pdf_bytes:
            msg.attach(filename, pdf_bytes, 'application/pdf')
        msg.send(fail_silently=False)
        return True
    except Exception as e:
        print(f"Failed to dispatch Report Email: {e}")
        return False

def send_bundled_report_email(artifacts_list, date_str):
    subject = f"Jarvis Insight // MULTI-AGENT EOD SUMMARY ({date_str})"
    
    # 1. Build dynamic HTML metrics table
    table_rows = ""
    for a in artifacts_list:
        pnl = a.get('net_flow', 0.0)
        pnl_color = "#34d399" if pnl >= 0 else "#f87171"
        pnl_sign = "+" if pnl >= 0 else ""
        table_rows += f"""
        <tr style="border-bottom: 1px solid #1e293b;">
            <td style="padding: 12px 8px; font-family: monospace; font-size: 12px; color: #f3f4f6; text-align: left;">{a.get('bot_id', 'Unknown')}</td>
            <td style="padding: 12px 8px; font-family: monospace; font-size: 11px; color: #9ca3af; text-align: center;">{a.get('report_type', 'DAILY')}</td>
            <td style="padding: 12px 8px; font-family: monospace; font-size: 12px; color: {pnl_color}; text-align: right; font-weight: bold;">{pnl_sign}${pnl:,.2f}</td>
            <td style="padding: 12px 8px; font-family: monospace; font-size: 12px; color: #cbd5e1; text-align: right;">{a.get('total_trades', 0)}</td>
            <td style="padding: 12px 8px; font-family: monospace; font-size: 12px; color: #60a5fa; text-align: right; font-weight: bold;">{a.get('win_rate', 0.0):.1f}%</td>
        </tr>
        """
        
    combined_markdown = f"# Multi-Agent Execution Telemetry :: {date_str}\n\n"
    for a in artifacts_list:
        md_content = ""
        md_path = a.get('md_path')
        if md_path:
            from pathlib import Path
            p = Path(md_path)
            if p.exists():
                md_content = p.read_text(encoding='utf-8')
        combined_markdown += f"---\n\n{md_content}\n\n"
        
    text_content = combined_markdown
    
    html_content = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #030712; padding: 40px 20px; color: #cbd5e1; line-height: 1.6;">
        <div style="max-width: 650px; margin: 0 auto; background-color: #0f172a; border-radius: 16px; overflow: hidden; border: 1px solid #1e293b; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);">
            <!-- Accent Header -->
            <div style="background: linear-gradient(135deg, #1e3a8a 0%, #312e81 100%); padding: 25px 30px; text-align: left;">
                <span style="background-color: rgba(255, 255, 255, 0.15); color: #ffffff; padding: 4px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;">Fleet Telemetry</span>
                <h1 style="color: #ffffff; margin: 10px 0 0 0; font-size: 20px; font-weight: 800; letter-spacing: 0.5px;">MULTI-AGENT EOD SUMMARY</h1>
            </div>
            
            <div style="padding: 30px 25px;">
                <p style="color: #9ca3af; font-size: 14px; margin-top: 0; margin-bottom: 25px;">The algorithmic node fleet has finalized daily operations. Inline metrics ledger follows below:</p>
                
                <!-- Performance Table -->
                <div style="overflow-hidden: true; border-radius: 12px; border: 1px solid #1e293b; margin-bottom: 25px; background-color: #020617;">
                    <table style="width: 100%; border-collapse: collapse; text-align: left;">
                        <thead>
                            <tr style="background-color: #0f172a; border-bottom: 1px solid #1e293b;">
                                <th style="padding: 12px 8px; font-family: monospace; font-size: 10px; color: #94a3b8; text-transform: uppercase; font-weight: bold;">Agent</th>
                                <th style="padding: 12px 8px; font-family: monospace; font-size: 10px; color: #94a3b8; text-transform: uppercase; font-weight: bold; text-align: center;">Type</th>
                                <th style="padding: 12px 8px; font-family: monospace; font-size: 10px; color: #94a3b8; text-transform: uppercase; font-weight: bold; text-align: right;">PnL</th>
                                <th style="padding: 12px 8px; font-family: monospace; font-size: 10px; color: #94a3b8; text-transform: uppercase; font-weight: bold; text-align: right;">Trades</th>
                                <th style="padding: 12px 8px; font-family: monospace; font-size: 10px; color: #94a3b8; text-transform: uppercase; font-weight: bold; text-align: right;">Win Rate</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_rows}
                        </tbody>
                    </table>
                </div>

                <div style="background-color: #1b2735; border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #283e56;">
                    <div style="font-size: 24px; margin-bottom: 8px;">📚</div>
                    <h3 style="color: #f8fafc; font-weight: 700; font-size: 14px; margin: 0 0 4px 0;">PDF Performance Dossiers Enclosed</h3>
                    <p style="color: #9ca3af; font-size: 11px; margin: 0; font-family: monospace;">Individual PDF breakdowns for each agent are attached to this email packet.</p>
                </div>
            </div>
        </div>
        <div style="text-align: center; margin-top: 25px; font-size: 10px; color: #4b5563; font-family: monospace; text-transform: uppercase; letter-spacing: 1px;">
            Jarvis Quant System Orchestrator &bull; Encrypted Channel
        </div>
    </div>
    """
    
    try:
        recipient = os.environ.get('EMAIL_RECIPIENT', 'tarunsree@gmail.com')
        msg = EmailMultiAlternatives(
            subject, text_content, 
            os.environ.get('EMAIL_HOST_USER', 'noreply@jarvis-trading.ai'),
            [recipient]
        )
        msg.attach_alternative(html_content, "text/html")
        for a in artifacts_list:
            if a.get('pdf_bytes'):
                bot_label = a.get('bot_id', 'unknown')
                filename = f"{bot_label.lower().replace(' ', '_')}_eod_{date_str}.pdf"
                msg.attach(filename, a['pdf_bytes'], 'application/pdf')
                
        msg.send(fail_silently=False)
        return True
    except Exception as e:
        print(f"Failed to dispatch Bundled Report Email: {e}")
        return False

def send_mutator_alert(status_message, diff_text=None, pdf_path=None, simple_summary=None):
    subject = f"🧠 JARVIS COGNITIVE MUTATOR ALERT"
    text_content = f"Mutator Diagnostic: {status_message}\nSummary: {simple_summary}"
    
    html_content = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #030712; padding: 40px 20px; color: #cbd5e1; line-height: 1.6;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #0f172a; border-radius: 16px; overflow: hidden; border: 1px solid #4c1d95; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);">
            <!-- Accent Header -->
            <div style="background: linear-gradient(135deg, #7c3aed 0%, #4c1d95 100%); padding: 25px 30px; text-align: left;">
                <span style="background-color: rgba(255, 255, 255, 0.15); color: #ffffff; padding: 4px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;">Cognitive Optimization</span>
                <h1 style="color: #ffffff; margin: 10px 0 0 0; font-size: 20px; font-weight: 800; letter-spacing: 0.5px;">MUTATOR SEQUENCE COMPLETE</h1>
            </div>
            
            <div style="padding: 30px 25px;">
                <p style="color: #9ca3af; font-size: 14px; margin-top: 0; margin-bottom: 25px;">The neural rewriter engine has finalized an autonomous strategy evolution block.</p>
                
                <div style="background-color: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; margin-bottom: 20px;">
                    <span style="color: #8b5cf6; font-size: 10px; text-transform: uppercase; letter-spacing: 1.5px; display: block; font-family: monospace; font-weight: bold; margin-bottom: 6px;">Evolution Summary</span>
                    <p style="color: #f1f5f9; font-size: 13px; margin: 0; line-height: 1.6; font-family: monospace;">{simple_summary or status_message}</p>
                </div>

                <div style="background-color: #1b2735; border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #283e56;">
                    <div style="font-size: 24px; margin-bottom: 8px;">📄</div>
                    <h3 style="color: #f8fafc; font-weight: 700; font-size: 14px; margin: 0 0 4px 0;">Technical Delta Report Enclosed</h3>
                    <p style="color: #9ca3af; font-size: 11px; margin: 0; font-family: monospace;">A PDF containing code diff files and reasoning is attached.</p>
                </div>
            </div>
        </div>
        <div style="text-align: center; margin-top: 25px; font-size: 10px; color: #4b5563; font-family: monospace; text-transform: uppercase; letter-spacing: 1px;">
            Jarvis Quant System Orchestrator &bull; Encrypted Channel
        </div>
    </div>
    """
    
    try:
        recipient = os.environ.get('EMAIL_RECIPIENT', 'tarunsree@gmail.com')
        msg = EmailMultiAlternatives(subject, text_content, os.environ.get('EMAIL_HOST_USER', 'noreply@jarvis-trading.ai'), [recipient])
        msg.attach_alternative(html_content, "text/html")
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, 'rb') as f:
                msg.attach(os.path.basename(pdf_path), f.read(), 'application/pdf')
        msg.send(fail_silently=False)
        return True
    except Exception as e:
        print(f"Failed to dispatch Mutator Alert Email: {e}")
        return False

def send_node_status_email(node_type, identifier, status, message):
    subject = f"JARVIS ORCHESTRATION // {node_type.upper()} {status.upper()}"
    text_content = f"{node_type} [{identifier}] is now {status}.\nDetail: {message}"
    
    html_content = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #030712; padding: 40px 20px; color: #cbd5e1; line-height: 1.6;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #0f172a; border-radius: 16px; overflow: hidden; border: 1px solid #1e293b; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);">
            <!-- Accent Header -->
            <div style="background: linear-gradient(135deg, #0d9488 0%, #115e59 100%); padding: 25px 30px; text-align: left;">
                <span style="background-color: rgba(255, 255, 255, 0.15); color: #ffffff; padding: 4px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;">Orchestration Lifecycle</span>
                <h1 style="color: #ffffff; margin: 10px 0 0 0; font-size: 20px; font-weight: 800; letter-spacing: 0.5px;">NODE TRANSITION DETECTED</h1>
            </div>
            
            <div style="padding: 30px 25px;">
                <p style="color: #9ca3af; font-size: 14px; margin-top: 0; margin-bottom: 25px;">A state transition has occurred within the compute fleet cluster topology:</p>
                
                <div style="background-color: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155;">
                    <span style="color: #64748b; font-size: 10px; text-transform: uppercase; letter-spacing: 1.5px; display: block; font-family: monospace; font-weight: bold; margin-bottom: 8px;">Trigger Node</span>
                    <strong style="color: #f8fafc; font-size: 15px; font-family: monospace; display: block;">{node_type} - {identifier}</strong>
                    
                    <div style="margin-top: 15px; padding: 12px; background-color: #020617; border-radius: 8px; border: 1px solid #1e293b; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; color: #cbd5e1;">
                        <span style="color: #475569; font-size: 9px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 4px;">STATE CODE</span>
                        <strong style="color: #0d9488;">[{status.upper()}]</strong> {message}
                    </div>
                </div>
            </div>
        </div>
        <div style="text-align: center; margin-top: 25px; font-size: 10px; color: #4b5563; font-family: monospace; text-transform: uppercase; letter-spacing: 1px;">
            Jarvis Quant System Orchestrator &bull; Encrypted Channel
        </div>
    </div>
    """
    try:
        recipient = os.environ.get('EMAIL_RECIPIENT', 'tarunsree@gmail.com')
        msg = EmailMultiAlternatives(subject, text_content, os.environ.get('EMAIL_HOST_USER', 'noreply@jarvis-trading.ai'), [recipient])
        msg.attach_alternative(html_content, "text/html")
        msg.send(fail_silently=False)
        return True
    except Exception as e:
        print(f"Failed to dispatch System Orchestration Email: {e}")
        return False
