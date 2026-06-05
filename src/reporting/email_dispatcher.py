import os
from django.core.mail import EmailMultiAlternatives
from django.conf import settings

def _build_premium_email_html(title, subtitle, category_tag, badge_color, main_content_html, accent_gradient=None):
    """
    Renders a premium institutional HTML notification from scratch.
    """
    accent_bar = accent_gradient or "linear-gradient(90deg, #6366f1 0%, #a855f7 100%)"
    
    return f"""
    <div style="background-color: #08090d; padding: 45px 15px; font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.5; color: #b0b5c6;">
        <div style="max-width: 580px; margin: 0 auto; background-color: #0f111a; border: 1px solid #1c1e2b; border-radius: 12px; overflow: hidden; box-shadow: 0 20px 40px rgba(0, 0, 0, 0.45);">
            <!-- Neon Accent Bar -->
            <div style="height: 3.5px; background: {accent_bar}; width: 100%;"></div>
            
            <div style="padding: 32px 28px;">
                <!-- Header Category Tag -->
                <div style="margin-bottom: 16px;">
                    <span style="background-color: {badge_color}1a; color: {badge_color}; border: 1px solid {badge_color}2b; padding: 3px 8px; border-radius: 6px; font-size: 9px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; font-family: ui-monospace, monospace;">
                        {category_tag}
                    </span>
                </div>
                
                <!-- Title & Subtitle -->
                <h1 style="color: #ffffff; font-size: 19px; font-weight: 800; margin: 0 0 6px 0; letter-spacing: -0.02em;">
                    {title}
                </h1>
                <p style="color: #6a7185; font-size: 11px; margin: 0 0 28px 0; font-family: ui-monospace, monospace; text-transform: uppercase; letter-spacing: 0.5px;">
                    {subtitle}
                </p>
                
                <!-- Main Body Content -->
                <div style="font-size: 13.5px; color: #cad3f5;">
                    {main_content_html}
                </div>
                
                <!-- Minimalist Footer -->
                <div style="margin-top: 36px; padding-top: 20px; border-top: 1px solid #1c1e2b; text-align: center;">
                    <p style="font-size: 9px; color: #474c5d; font-family: ui-monospace, monospace; margin: 0; text-transform: uppercase; letter-spacing: 1.5px;">
                        SYSTEM: OPERATIONAL &bull; SECURE RELAY GATEWAY
                    </p>
                </div>
            </div>
        </div>
        <div style="text-align: center; margin-top: 20px; font-size: 9px; color: #3d404d; font-family: ui-monospace, monospace; text-transform: uppercase; letter-spacing: 1px;">
            QUANTTRADER CENTRAL ARCHITECTURE PROTOCOL &bull; DO NOT REPLY
        </div>
    </div>
    """

def send_sos_alert(model_name, traceback_str):
    subject = f"🛑 JARVIS EXCEPTION: {model_name}"
    text_content = f"Execution interrupted.\n\nTraceback:\n{traceback_str}"
    
    main_content = f"""
    <p style="margin-top: 0; margin-bottom: 20px; color: #a5adcb; font-size: 13.5px; line-height: 1.6;">
        An execution node encountered an unhandled critical exception. The self-healing watchdog daemon has logged the traceback sequence below:
    </p>
    
    <div style="background-color: #161722; border: 1px solid #232536; border-radius: 8px; padding: 14px; margin-bottom: 20px; font-family: ui-monospace, monospace;">
        <span style="color: #7a7e8c; font-size: 9px; text-transform: uppercase; letter-spacing: 1px; display: block;">Affected Neural Core</span>
        <strong style="color: #ffffff; font-size: 14px; display: block; margin-top: 2px;">{model_name}</strong>
    </div>

    <div style="background-color: #08090d; padding: 16px; border: 1px solid #1c1e2b; border-radius: 8px; overflow-x: auto; margin-bottom: 20px;">
        <span style="color: #fca5a5; font-size: 10px; font-family: ui-monospace, monospace; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 8px;">Stack Exception Traceback</span>
        <code style="color: #f87171; font-size: 11.5px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap; display: block; line-height: 1.6; word-break: break-all;">{traceback_str}</code>
    </div>

    <div style="background-color: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 8px; padding: 12px; text-align: center;">
        <p style="font-size: 11px; color: #fca5a5; font-weight: 600; margin: 0; font-family: ui-monospace, monospace; text-transform: uppercase; letter-spacing: 0.5px;">
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
    <p style="margin-top: 0; margin-bottom: 20px; color: #a5adcb; font-size: 13.5px; line-height: 1.6;">
        The system has compiled the rolling performance reports and created a print-ready intelligence record. Key metrics have been committed to the secure database ledger.
    </p>
    
    <div style="background-color: #161722; border: 1px solid #232536; border-radius: 10px; padding: 24px; text-align: center; margin-bottom: 15px;">
        <div style="font-size: 28px; margin-bottom: 12px;">📊</div>
        <h3 style="color: #ffffff; font-weight: 700; font-size: 14px; margin: 0 0 6px 0; text-transform: uppercase; letter-spacing: 0.5px;">Performance Ledger Enclosed</h3>
        <p style="color: #8e95a5; font-size: 11px; margin: 0; font-family: ui-monospace, monospace;">
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
        pnl_color = "#34d399" if pnl >= 0 else "#f87171"
        pnl_sign = "+" if pnl >= 0 else ""
        table_rows += f"""
        <tr style="border-bottom: 1px solid #1c1e2b;">
            <td style="padding: 10px 8px; font-family: ui-monospace, monospace; font-size: 11.5px; color: #ffffff; text-align: left;">{a.get('bot_id', 'Unknown')}</td>
            <td style="padding: 10px 8px; font-family: ui-monospace, monospace; font-size: 10.5px; color: #8e95a5; text-align: center;">{a.get('report_type', 'DAILY')}</td>
            <td style="padding: 10px 8px; font-family: ui-monospace, monospace; font-size: 11.5px; color: {pnl_color}; text-align: right; font-weight: bold;">{pnl_sign}${pnl:,.2f}</td>
            <td style="padding: 10px 8px; font-family: ui-monospace, monospace; font-size: 11.5px; color: #b0b5c6; text-align: right;">{a.get('total_trades', 0)}</td>
            <td style="padding: 10px 8px; font-family: ui-monospace, monospace; font-size: 11.5px; color: #60a5fa; text-align: right; font-weight: bold;">{a.get('win_rate', 0.0):.1f}%</td>
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
    <p style="margin-top: 0; margin-bottom: 20px; color: #a5adcb; font-size: 13.5px; line-height: 1.6;">
        The multi-agent trading fleet has completed EOD operations. Integrated performance metrics across active sandbox and live nodes are detailed in the ledger below:
    </p>

    <!-- Performance Table Wrapper -->
    <div style="border-radius: 10px; border: 1px solid #1c1e2b; margin-bottom: 24px; overflow: hidden; background-color: #0c0d12;">
        <table style="width: 100%; border-collapse: collapse; text-align: left;">
            <thead>
                <tr style="background-color: #12131a; border-bottom: 1.5px solid #1c1e2b;">
                    <th style="padding: 10px 8px; font-family: ui-monospace, monospace; font-size: 9px; color: #8e95a5; text-transform: uppercase; font-weight: bold; letter-spacing: 0.5px;">Agent</th>
                    <th style="padding: 10px 8px; font-family: ui-monospace, monospace; font-size: 9px; color: #8e95a5; text-transform: uppercase; font-weight: bold; text-align: center; letter-spacing: 0.5px;">Type</th>
                    <th style="padding: 10px 8px; font-family: ui-monospace, monospace; font-size: 9px; color: #8e95a5; text-transform: uppercase; font-weight: bold; text-align: right; letter-spacing: 0.5px;">Net Yield</th>
                    <th style="padding: 10px 8px; font-family: ui-monospace, monospace; font-size: 9px; color: #8e95a5; text-transform: uppercase; font-weight: bold; text-align: right; letter-spacing: 0.5px;">Trades</th>
                    <th style="padding: 10px 8px; font-family: ui-monospace, monospace; font-size: 9px; color: #8e95a5; text-transform: uppercase; font-weight: bold; text-align: right; letter-spacing: 0.5px;">Win Rate</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>

    <div style="background-color: #161722; border: 1px solid #232536; border-radius: 10px; padding: 18px; text-align: center;">
        <div style="font-size: 22px; margin-bottom: 8px;">📚</div>
        <h3 style="color: #ffffff; font-weight: 700; font-size: 13px; margin: 0 0 4px 0; text-transform: uppercase; letter-spacing: 0.5px;">PDF Performance Dossiers Enclosed</h3>
        <p style="color: #8e95a5; font-size: 10.5px; margin: 0; font-family: ui-monospace, monospace;">
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
    <p style="margin-top: 0; margin-bottom: 20px; color: #a5adcb; font-size: 13.5px; line-height: 1.6;">
        The neural rewriter mutator has finalized an autonomous strategy evolution block. The runtime environment has adjusted dynamic weights successfully.
    </p>
    
    <div style="background-color: #161722; border: 1px solid #232536; border-radius: 8px; padding: 16px; margin-bottom: 20px; font-family: ui-monospace, monospace;">
        <span style="color: #8b5cf6; font-size: 9px; text-transform: uppercase; letter-spacing: 1px; display: block; font-weight: bold; margin-bottom: 6px;">Evolution Summary</span>
        <p style="color: #ffffff; font-size: 12.5px; margin: 0; line-height: 1.5;">{simple_summary or status_message}</p>
    </div>

    <div style="background-color: #08090d; border: 1px solid #1c1e2b; border-radius: 10px; padding: 18px; text-align: center;">
        <div style="font-size: 22px; margin-bottom: 8px;">📄</div>
        <h3 style="color: #ffffff; font-weight: 700; font-size: 13px; margin: 0 0 4px 0; text-transform: uppercase; letter-spacing: 0.5px;">Technical Delta Report Attached</h3>
        <p style="color: #8e95a5; font-size: 10.5px; margin: 0; font-family: ui-monospace, monospace;">
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
    
    status_color = "#34d399" if status.upper() == "STARTED" else "#f87171"
    
    main_content = f"""
    <p style="margin-top: 0; margin-bottom: 20px; color: #a5adcb; font-size: 13.5px; line-height: 1.6;">
        A cluster topology transition has occurred in the compute execution fleet:
    </p>
    
    <div style="background-color: #161722; border: 1px solid #232536; border-radius: 10px; padding: 20px;">
        <span style="color: #7a7e8c; font-size: 9px; text-transform: uppercase; letter-spacing: 1px; display: block; font-family: ui-monospace, monospace;">Trigger Isolation Node</span>
        <strong style="color: #ffffff; font-size: 14px; font-family: ui-monospace, monospace; display: block; margin-top: 2px;">{node_type} &middot; {identifier}</strong>
        
        <div style="margin-top: 14px; padding: 12px; background-color: #08090d; border-radius: 6px; border: 1px solid #1c1e2b; font-family: ui-monospace, monospace; font-size: 11.5px; color: #cbd5e1; line-height: 1.5;">
            <span style="color: #474c5d; font-size: 8.5px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px; display: block; margin-bottom: 4px;">STATE CODE</span>
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
