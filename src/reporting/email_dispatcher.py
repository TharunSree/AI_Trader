import os
from django.core.mail import EmailMultiAlternatives
from django.conf import settings

def _build_premium_email_html(title, subtitle, category_tag, badge_color, main_content_html, accent_gradient=None):
    """
    Renders a premium institutional light-themed HTML notification with corporate branding.
    """
    accent_bar = accent_gradient or "linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%)"
    
    return f"""
    <div style="background-color: #f8fafc; padding: 40px 15px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.5; color: #334155;">
        <div style="max-width: 580px; margin: 0 auto; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);">
            <!-- Elegant Accent Bar -->
            <div style="height: 4px; background: {accent_bar}; width: 100%;"></div>
            
            <div style="padding: 32px 28px;">
                <!-- Header Logo & Branding -->
                <table cellpadding="0" cellspacing="0" border="0" style="width: 100%; margin-bottom: 24px; border-bottom: 1px solid #f1f5f9; padding-bottom: 16px;">
                    <tr>
                        <td style="font-size: 16px; font-weight: 800; color: #0f172a; font-family: sans-serif; letter-spacing: 0.5px;">
                            QUANT<span style="color: #3b82f6;">TRADER</span>
                            <span style="font-size: 9px; font-weight: bold; background-color: #eff6ff; color: #1d4ed8; border: 1px solid #dbeafe; padding: 2px 6px; border-radius: 4px; font-family: monospace; margin-left: 8px; vertical-align: middle; text-transform: uppercase; letter-spacing: 0.5px;">AI ENGINE</span>
                        </td>
                        <td style="text-align: right;">
                            <span style="background-color: {badge_color}15; color: {badge_color}; border: 1px solid {badge_color}30; padding: 3px 8px; border-radius: 6px; font-size: 9px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; font-family: monospace;">
                                {category_tag}
                            </span>
                        </td>
                    </tr>
                </table>
                
                <!-- Title & Subtitle -->
                <h1 style="color: #0f172a; font-size: 20px; font-weight: 800; margin: 0 0 6px 0; letter-spacing: -0.02em;">
                    {title}
                </h1>
                <p style="color: #64748b; font-size: 10.5px; margin: 0 0 24px 0; font-family: monospace; text-transform: uppercase; letter-spacing: 0.5px;">
                    {subtitle}
                </p>
                
                <!-- Main Body Content -->
                <div style="font-size: 13.5px; color: #334155;">
                    {main_content_html}
                </div>
                
                <!-- Corporate Footer -->
                <div style="margin-top: 36px; padding-top: 20px; border-top: 1px solid #f1f5f9; text-align: center;">
                    <p style="font-size: 9px; color: #94a3b8; font-family: monospace; margin: 0; text-transform: uppercase; letter-spacing: 1px;">
                        SYSTEM STATUS: ACTIVE &bull; SECURE QUANT LINK
                    </p>
                </div>
            </div>
        </div>
        <div style="text-align: center; margin-top: 24px; font-size: 9px; color: #94a3b8; font-family: monospace; text-transform: uppercase; letter-spacing: 0.5px;">
            QUANTTRADER TECHNOLOGIES &bull; CONFIDENTIAL TRANSMISSION
        </div>
    </div>
    """

def send_sos_alert(model_name, traceback_str):
    subject = f"🛑 JARVIS EXCEPTION: {model_name}"
    text_content = f"Execution interrupted.\n\nTraceback:\n{traceback_str}"
    
    main_content = f"""
    <p style="margin-top: 0; margin-bottom: 20px; color: #475569; font-size: 13.5px; line-height: 1.6;">
        An execution node encountered an unhandled critical exception. The self-healing watchdog daemon has logged the traceback sequence below:
    </p>
    
    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; margin-bottom: 20px; font-family: monospace;">
        <span style="color: #64748b; font-size: 9px; text-transform: uppercase; letter-spacing: 1px; display: block; font-weight: bold;">Affected Neural Core</span>
        <strong style="color: #0f172a; font-size: 14px; display: block; margin-top: 2px;">{model_name}</strong>
    </div>

    <div style="background-color: #fef2f2; padding: 16px; border: 1px solid #fee2e2; border-radius: 8px; overflow-x: auto; margin-bottom: 20px; font-family: monospace;">
        <span style="color: #b91c1c; font-size: 10px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 8px;">Stack Exception Traceback</span>
        <code style="color: #991b1b; font-size: 11.5px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap; display: block; line-height: 1.6; word-break: break-all;">{traceback_str}</code>
    </div>

    <div style="background-color: #fef2f2; border: 1px solid #fca5a5; border-radius: 8px; padding: 12px; text-align: center;">
        <p style="font-size: 11px; color: #b91c1c; font-weight: 600; margin: 0; font-family: monospace; text-transform: uppercase; letter-spacing: 0.5px;">
            Watchdog Action: Initiating rollback / self-repair protocols.
        </p>
    </div>
    """
    
    html_content = _build_premium_email_html(
        "CRITICAL SYSTEM EXCEPTION",
        "Isolation Core Alert",
        "CRITICAL ANOMALY",
        "#ef4444",
        main_content,
        "linear-gradient(90deg, #ef4444 0%, #b91c1c 100%)"
    )
    
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
    
    main_content = f"""
    <p style="margin-top: 0; margin-bottom: 20px; color: #475569; font-size: 13.5px; line-height: 1.6;">
        The system has compiled the rolling performance reports and created a print-ready intelligence record. Key metrics have been committed to the secure database ledger.
    </p>
    
    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 24px; text-align: center; margin-bottom: 15px;">
        <div style="font-size: 28px; margin-bottom: 12px;">📊</div>
        <h3 style="color: #0f172a; font-weight: 700; font-size: 14px; margin: 0 0 6px 0; text-transform: uppercase; letter-spacing: 0.5px;">Performance Ledger Enclosed</h3>
        <p style="color: #64748b; font-size: 11px; margin: 0; font-family: monospace;">
            A detailed, print-ready PDF performance ledger has been attached to this email.
        </p>
    </div>
    """
    
    html_content = _build_premium_email_html(
        f"{report_type.upper()} DOSSIER COMPILED",
        "Report Archive Gateway",
        "INTELLIGENCE PACKET",
        "#3b82f6",
        main_content,
        "linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%)"
    )
    
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
    
    table_rows = ""
    for a in artifacts_list:
        pnl = a.get('net_flow', 0.0)
        pnl_color = "#16a34a" if pnl >= 0 else "#dc2626"
        pnl_sign = "+" if pnl >= 0 else ""
        table_rows += f"""
        <tr style="border-bottom: 1px solid #f1f5f9;">
            <td style="padding: 10px 8px; font-family: monospace; font-size: 11.5px; color: #0f172a; text-align: left;">{a.get('bot_id', 'Unknown')}</td>
            <td style="padding: 10px 8px; font-family: monospace; font-size: 10.5px; color: #475569; text-align: center;">{a.get('report_type', 'DAILY')}</td>
            <td style="padding: 10px 8px; font-family: monospace; font-size: 11.5px; color: {pnl_color}; text-align: right; font-weight: bold;">{pnl_sign}${pnl:,.2f}</td>
            <td style="padding: 10px 8px; font-family: monospace; font-size: 11.5px; color: #64748b; text-align: right;">{a.get('total_trades', 0)}</td>
            <td style="padding: 10px 8px; font-family: monospace; font-size: 11.5px; color: #2563eb; text-align: right; font-weight: bold;">{a.get('win_rate', 0.0):.1f}%</td>
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
    
    main_content = f"""
    <p style="margin-top: 0; margin-bottom: 20px; color: #475569; font-size: 13.5px; line-height: 1.6;">
        The multi-agent trading fleet has completed EOD operations. Integrated performance metrics across active sandbox and live nodes are detailed in the ledger below:
    </p>

    <!-- Performance Table Wrapper -->
    <div style="border-radius: 10px; border: 1px solid #e2e8f0; margin-bottom: 24px; overflow: hidden; background-color: #ffffff;">
        <table style="width: 100%; border-collapse: collapse; text-align: left;">
            <thead>
                <tr style="background-color: #f8fafc; border-bottom: 1.5px solid #e2e8f0;">
                    <th style="padding: 10px 8px; font-family: monospace; font-size: 9px; color: #64748b; text-transform: uppercase; font-weight: bold; letter-spacing: 0.5px;">Agent</th>
                    <th style="padding: 10px 8px; font-family: monospace; font-size: 9px; color: #64748b; text-transform: uppercase; font-weight: bold; text-align: center; letter-spacing: 0.5px;">Type</th>
                    <th style="padding: 10px 8px; font-family: monospace; font-size: 9px; color: #64748b; text-transform: uppercase; font-weight: bold; text-align: right; letter-spacing: 0.5px;">Net Yield</th>
                    <th style="padding: 10px 8px; font-family: monospace; font-size: 9px; color: #64748b; text-transform: uppercase; font-weight: bold; text-align: right; letter-spacing: 0.5px;">Trades</th>
                    <th style="padding: 10px 8px; font-family: monospace; font-size: 9px; color: #64748b; text-transform: uppercase; font-weight: bold; text-align: right; letter-spacing: 0.5px;">Win Rate</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>

    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px; text-align: center;">
        <div style="font-size: 22px; margin-bottom: 8px;">📚</div>
        <h3 style="color: #0f172a; font-weight: 700; font-size: 13px; margin: 0 0 4px 0; text-transform: uppercase; letter-spacing: 0.5px;">PDF Performance Dossiers Enclosed</h3>
        <p style="color: #64748b; font-size: 10.5px; margin: 0; font-family: monospace;">
            Individual, detailed PDF breakdowns for each agent are attached to this transmission packet.
        </p>
    </div>
    """
    
    html_content = _build_premium_email_html(
        "MULTI-AGENT EOD SUMMARY",
        f"Fleet Performance Ledger &bull; {date_str}",
        "FLEET TELEMETRY",
        "#3b82f6",
        main_content,
        "linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%)"
    )
    
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
    
    main_content = f"""
    <p style="margin-top: 0; margin-bottom: 20px; color: #475569; font-size: 13.5px; line-height: 1.6;">
        The neural rewriter mutator has finalized an autonomous strategy evolution block. The runtime environment has adjusted dynamic weights successfully.
    </p>
    
    <div style="background-color: #f5f3ff; border: 1px solid #ddd6fe; border-radius: 8px; padding: 16px; margin-bottom: 20px; font-family: monospace;">
        <span style="color: #7c3aed; font-size: 9px; text-transform: uppercase; letter-spacing: 1px; display: block; font-weight: bold; margin-bottom: 6px;">Evolution Summary</span>
        <p style="color: #0f172a; font-size: 12.5px; margin: 0; line-height: 1.5;">{simple_summary or status_message}</p>
    </div>

    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px; text-align: center;">
        <div style="font-size: 22px; margin-bottom: 8px;">📄</div>
        <h3 style="color: #0f172a; font-weight: 700; font-size: 13px; margin: 0 0 4px 0; text-transform: uppercase; letter-spacing: 0.5px;">Technical Delta Report Attached</h3>
        <p style="color: #64748b; font-size: 10.5px; margin: 0; font-family: monospace;">
            A PDF breakdown containing code diffs and diagnostic reasoning is attached.
        </p>
    </div>
    """
    
    html_content = _build_premium_email_html(
        "MUTATOR SEQUENCE COMPLETE",
        "Cognitive Strategy Evolution Hook",
        "COGNITIVE OPTIMIZATION",
        "#a855f7",
        main_content,
        "linear-gradient(90deg, #a855f7 0%, #7c3aed 100%)"
    )
    
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
    
    status_color = "#0d9488" if status.upper() == "STARTED" else "#dc2626"
    
    main_content = f"""
    <p style="margin-top: 0; margin-bottom: 20px; color: #475569; font-size: 13.5px; line-height: 1.6;">
        A cluster topology transition has occurred in the compute execution fleet:
    </p>
    
    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px;">
        <span style="color: #64748b; font-size: 9px; text-transform: uppercase; letter-spacing: 1px; display: block; font-family: monospace;">Trigger Isolation Node</span>
        <strong style="color: #0f172a; font-size: 14px; font-family: monospace; display: block; margin-top: 2px;">{node_type} &middot; {identifier}</strong>
        
        <div style="margin-top: 14px; padding: 12px; background-color: #f0fdfa; border-radius: 6px; border: 1px solid #ccfbf1; font-family: monospace; font-size: 11.5px; color: #0f172a; line-height: 1.5;">
            <span style="color: #64748b; font-size: 8.5px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px; display: block; margin-bottom: 4px;">STATE CODE</span>
            <strong style="color: {status_color};">[{status.upper()}]</strong> {message}
        </div>
    </div>
    """
    
    html_content = _build_premium_email_html(
        "NODE TRANSITION DETECTED",
        "Orchestration Lifecycle Alert",
        "FLEET ORCHESTRATION",
        "#0d9488",
        main_content,
        "linear-gradient(90deg, #0d9488 0%, #115e59 100%)"
    )
    
    try:
        recipient = os.environ.get('EMAIL_RECIPIENT', 'tarunsree@gmail.com')
        msg = EmailMultiAlternatives(subject, text_content, os.environ.get('EMAIL_HOST_USER', 'noreply@jarvis-trading.ai'), [recipient])
        msg.attach_alternative(html_content, "text/html")
        msg.send(fail_silently=False)
        return True
    except Exception as e:
        print(f"Failed to dispatch System Orchestration Email: {e}")
        return False
