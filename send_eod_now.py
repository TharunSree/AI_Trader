"""
Jarvis EOD Pipeline - Generates reports and triggers daily mutation.
Run this script after market close to:
  1. Generate EOD reports for all active bots
  2. Email the bundled reports
  3. Auto-trigger the cognitive mutation engine (implements EOD AI suggestions)
"""
from datetime import date, timedelta
import os
import sys
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trader_project.settings')
django.setup()

from src.reporting.eod_generator import write_report_artifacts
from src.reporting.email_dispatcher import send_bundled_report_email

def trigger_dispatch():
    today = date.today()
    print(f"[EOD PIPELINE] Generating artifacts for {today}...")
    artifacts_list = write_report_artifacts(today)
    
    if not artifacts_list:
        print("[EOD PIPELINE] No bots found for today. Trying yesterday...")
        yesterday = today - timedelta(days=1)
        artifacts_list = write_report_artifacts(yesterday)
        if artifacts_list:
            today = yesterday  # Use yesterday's date for email
    
    if artifacts_list:
        print(f"[EOD PIPELINE] Dispatching bundled email for {len(artifacts_list)} isolated bots...")
        res = send_bundled_report_email(artifacts_list, today.isoformat())
        print(f"[EOD PIPELINE] Email dispatch: {'Success' if res else 'FAILED'}")
    else:
        print("[EOD PIPELINE] No active trading bots found. Skipping report dispatch.")
    
    # Phase 2: Auto-trigger daily cognitive mutation
    # This reads the EOD AI suggestions and implements them into the trading strategy
    if '--no-mutate' not in sys.argv:
        if not artifacts_list:
            print("[EOD PIPELINE] Mutation skipped: no reports were generated today.")
        else:
            print("\n[EOD PIPELINE] Phase 2: Triggering Daily Cognitive Mutation...")
            try:
                from src.core.code_rewriter import orchestrate_rewrite
                orchestrate_rewrite()
                print("[EOD PIPELINE] Mutation complete. Strategy evolved for next session.")
            except Exception as e:
                print(f"[EOD PIPELINE] Mutation failed (non-fatal): {e}")
    else:
        print("[EOD PIPELINE] Mutation skipped (--no-mutate flag).")

if __name__ == '__main__':
    trigger_dispatch()
