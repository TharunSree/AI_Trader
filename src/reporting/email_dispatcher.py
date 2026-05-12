import os
from django.core.mail import EmailMultiAlternatives
from django.conf import settings

def send_sos_alert(model_name, traceback_str):
    subject = f"🛑 JARVIS EXCEPTION: {model_name}"
    text_content = f"Execution interrupted.\n\nTraceback:\n{traceback_str}"
    
    html_content = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0f172a; padding: 40px 20px; color: #cbd5e1; line-height: 1.6;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #1e293b; border-radius: 16px; overflow: hidden; border: 1px solid #334155; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);">
            <div style="padding: 35px 30px;">
                
                <div style="display: flex; align-items: center; margin-bottom: 20px;">
                    <div style="background-color: #ef444420; color: #ef4444; padding: 6px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;">
                        System Anomaly
                    </div>
                </div>

                <p style="color: #94a3b8; font-size: 15px; margin-top: 0; margin-bottom: 30px; border-bottom: 1px solid #334155; padding-bottom: 25px;">The neural router encountered an unexpected state failure during execution blocks.</p>
                
                <div style="background-color: #0f172a; border-radius: 10px; padding: 20px; margin-bottom: 25px; border: 1px solid #334155;">
                    <div style="color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">Target Isolation Node</div>
                    <div style="color: #f8fafc; font-size: 16px; font-weight: 600; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;">{model_name}</div>
                </div>

                <div style="background-color: #020617; padding: 20px; border-radius: 10px; overflow-x: auto; border: 1px solid #1e293b;">
                    <div style="color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                        <span>Exception Traceback</span>
                    </div>
                    <code style="color: #fca5a5; font-size: 13px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; white-space: pre-wrap; display: block; line-height: 1.5;">{traceback_str}</code>
                </div>
                
                <div style="margin-top: 35px; background-color: #3b82f615; border: 1px solid #3b82f630; border-radius: 8px; padding: 16px; text-align: center;">
                    <p style="font-size: 12px; color: #93c5fd; font-weight: 500; margin: 0;">Cognitive Mutator protocol has intercepted the traceback for autonomic code repair.</p>
                </div>
            </div>
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
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0f172a; padding: 40px 20px; color: #cbd5e1; line-height: 1.6;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #1e293b; border-radius: 16px; overflow: hidden; border: 1px solid #334155; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);">
            <div style="padding: 35px 30px;">
                
                <div style="display: flex; align-items: center; margin-bottom: 20px;">
                    <div style="background-color: #3b82f620; color: #60a5fa; padding: 6px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;">
                        Insight Generation
                    </div>
                </div>

                <p style="color: #94a3b8; font-size: 15px; margin-top: 0; margin-bottom: 30px; border-bottom: 1px solid #334155; padding-bottom: 25px;">Your automated algorithmic market intelligence telemetry has been successfully compiled.</p>
                
                <div style="background-color: #0f172a; border-radius: 10px; padding: 30px 20px; text-align: center; border: 1px solid #334155;">
                    <div style="font-size: 32px; margin-bottom: 16px;">📊</div>
                    <div style="color: #f8fafc; font-weight: 600; font-size: 16px;">Telemetry Document Enclosed</div>
                    <div style="color: #94a3b8; font-size: 13px; margin-top: 8px;">The {report_type.lower()} execution analytics dataset is attached globally to this payload securely.</div>
                </div>
                
                <div style="margin-top: 35px; text-align: center;">
                    <p style="font-size: 11px; font-weight: 500; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; color: #475569; margin: 0; text-transform: uppercase; letter-spacing: 1px;">Jarvis Quantitative Operations</p>
                </div>
            </div>
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
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0f172a; padding: 40px 20px; color: #cbd5e1; line-height: 1.6;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #1e293b; border-radius: 16px; overflow: hidden; border: 1px solid #334155; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);">
            <div style="padding: 35px 30px;">
                
                <div style="display: flex; align-items: center; margin-bottom: 20px;">
                    <div style="background-color: #3b82f620; color: #60a5fa; padding: 6px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;">
                        Fleet Intelligence Hub
                    </div>
                </div>

                <p style="color: #94a3b8; font-size: 15px; margin-top: 0; margin-bottom: 30px; border-bottom: 1px solid #334155; padding-bottom: 25px;">Your automated algorithms have completed execution. Their telemetry has been explicitly bundled below.</p>
                
                <div style="background-color: #0f172a; border-radius: 10px; padding: 30px 20px; text-align: center; border: 1px solid #334155;">
                    <div style="font-size: 32px; margin-bottom: 16px;">📚</div>
                    <div style="color: #f8fafc; font-weight: 600; font-size: 16px;">{len(artifacts_list)} Telemetry Documents Enclosed</div>
                    <div style="color: #94a3b8; font-size: 13px; margin-top: 8px;">Physical PDF performance breakdowns for multiple concurrent agents have been securely attached to this thread.</div>
                </div>
                
                <div style="margin-top: 35px; text-align: center;">
                    <p style="font-size: 11px; font-weight: 500; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; color: #475569; margin: 0; text-transform: uppercase; letter-spacing: 1px;">Jarvis Quantitative Operations</p>
                </div>
            </div>
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
                msg.attach(f"bot_{bot_label}_eod_{date_str}.pdf", a['pdf_bytes'], 'application/pdf')
                
        msg.send(fail_silently=False)
        return True
    except Exception as e:
        print(f"Failed to dispatch Bundled Report Email: {e}")
        return False

import html

def send_mutator_alert(status_message, diff_text=None, pdf_path=None, simple_summary=None):
    subject = f"🧠 JARVIS COGNITIVE MUTATOR ALERT"
    text_content = f"Mutator Diagnostic: {status_message}\nSummary: {simple_summary}"
    
    html_content = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0f172a; padding: 40px 20px; color: #cbd5e1; line-height: 1.6;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #1e293b; border-radius: 16px; overflow: hidden; border: 1px solid #8b5cf6; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);">
            <div style="padding: 35px 30px;">
                <div style="display: flex; align-items: center; margin-bottom: 20px;">
                    <div style="background-color: #8b5cf620; color: #c084fc; padding: 6px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;">
                        Cognitive Optimization
                    </div>
                </div>
                <p style="color: #94a3b8; font-size: 15px; margin-top: 0; margin-bottom: 30px; border-bottom: 1px solid #334155; padding-bottom: 25px;">The Gemini Mutator Engine has concluded an autonomous strategy evolution block.</p>
                
                <div style="background-color: #0f172a; border-radius: 10px; padding: 25px; border: 1px solid #334155; margin-bottom: 20px;">
                    <div style="color: #f8fafc; font-weight: 600; font-size: 15px; margin-bottom: 10px;">Evolution Summary</div>
                    <div style="color: #e2e8f0; font-size: 14px; line-height: 1.6;">{simple_summary or status_message}</div>
                </div>

                <div style="background-color: #8b5cf610; border-radius: 10px; padding: 20px; text-align: center; border: 1px solid #8b5cf630;">
                    <div style="font-size: 24px; margin-bottom: 10px;">📄</div>
                    <div style="color: #f8fafc; font-weight: 600; font-size: 14px;">Technical Delta Report Attached</div>
                    <div style="color: #94a3b8; font-size: 12px; margin-top: 5px;">A detailed PDF containing the neural reasoning and unified code diff is enclosed.</div>
                </div>
            </div>
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
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0f172a; padding: 40px 20px; color: #cbd5e1; line-height: 1.6;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #1e293b; border-radius: 16px; overflow: hidden; border: 1px solid #334155; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);">
            <div style="padding: 35px 30px;">
                <div style="display: flex; align-items: center; margin-bottom: 20px;">
                    <div style="background-color: #f59e0b20; color: #fbbf24; padding: 6px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;">
                        System Orchestration
                    </div>
                </div>
                <p style="color: #94a3b8; font-size: 15px; margin-top: 0; margin-bottom: 30px; border-bottom: 1px solid #334155; padding-bottom: 25px;">A state transition has occurred within the compute fleet.</p>
                <div style="background-color: #0f172a; border-radius: 10px; padding: 20px; border: 1px solid #334155;">
                    <div style="color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">Lifecycle Trigger</div>
                    <div style="color: #f8fafc; font-size: 16px; font-weight: 600;">{node_type} - {identifier}</div>
                    <div style="margin-top: 15px; padding: 12px; background-color: #020617; border-radius: 8px; border: 1px solid #1e293b; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; color: #cbd5e1;">
                        <span style="color: #64748b; font-size: 10px; uppercase; letter-spacing: 1px; display: block; margin-bottom: 5px;">STATUS</span>
                        <strong style="color: #60a5fa;">[{status.upper()}]</strong> {message}
                    </div>
                </div>
            </div>
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
