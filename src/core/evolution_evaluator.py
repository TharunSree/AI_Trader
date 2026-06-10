"""
Neural Evolution Engine — A/B Evaluation Daemon

Checks all testing ModelVariants whose test window has expired,
compares their performance against the parent model, and marks
them as PENDING (winner) or FAILED (loser).

Can be run as:
    python -m src.core.evolution_evaluator

Or called from a scheduled task / Celery beat.
"""
import os
import sys
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trader_project.settings')
django.setup()

from control_panel.models import ModelVariant, PaperTrader, SystemAlert
from django.utils import timezone


def _delete_failed_variant(variant, reason_msg):
    from pathlib import Path
    import signal
    from django.conf import settings as django_settings
    
    # Try to kill process
    if variant.celery_task_id:
        try:
            os.kill(int(variant.celery_task_id), signal.SIGTERM)
        except Exception:
            pass
            
    # Try to delete log file
    log_file = Path(django_settings.BASE_DIR) / "logs" / f"evolution_variant_{variant.id}.log"
    if log_file.exists():
        try:
            log_file.unlink()
        except Exception as e:
            print(f"  [EVOLUTION EVAL] Failed to delete log file: {e}")
            
    # Compile detailed failure alert for negative feedback context
    alert_msg = f"**Variant Name**: {variant.name}\n"
    alert_msg += f"**Failure Type**: EVALUATION LOSS / INSUFFICIENT DATA\n"
    alert_msg += f"**Failure Reason**: {reason_msg}\n\n"
    
    # Compile performance metrics
    alert_msg += "### 📊 Trade Performance Summary\n"
    alert_msg += f"- **Total Trades**: {variant.virtual_trades_count}\n"
    alert_msg += f"- **Win Rate**: {variant.win_rate:.1f}%\n"
    alert_msg += f"- **Starting Balance**: ${float(variant.starting_cash):,.2f}\n"
    alert_msg += f"- **Final Balance**: ${float(variant.virtual_balance):,.2f}\n"
    alert_msg += f"- **Net P/L**: {variant.virtual_pnl_pct:+.2f}%\n"
    alert_msg += f"- **Sharpe Ratio**: {variant.sharpe_ratio:.4f}\n"
    alert_msg += f"- **Max Drawdown**: {variant.max_drawdown_pct:.2f}%\n\n"
    
    if variant.mutation_reasoning:
        alert_msg += f"### 💡 Attempted Rationale\n{variant.mutation_reasoning}\n\n"
    if variant.agent_code:
        alert_msg += f"### 💻 Agent Code\n```python\n{variant.agent_code}\n```\n"
        
    try:
        SystemAlert.objects.create(
            level='WARNING',
            title=f'🧬 Variant #{variant.id} Failed & Deleted: {reason_msg[:60]}',
            message=alert_msg,
            related_model_reference=str(variant.id)
        )
    except Exception as alert_err:
        print(f"  [EVOLUTION EVAL] Failed to write SystemAlert: {alert_err}")

    # Delete variant record
    print(f"  [EVOLUTION EVAL] Deleting variant #{variant.id} record. Reason: {reason_msg}")
    variant.delete()


def evaluate_expired_variants():
    """
    Check all TESTING variants whose test window has expired.
    Compare performance and promote/fail accordingly.
    """
    expired = [v for v in ModelVariant.objects.filter(status='TESTING') if v.is_test_expired]
    
    if not expired:
        print("[EVOLUTION EVAL] No expired test variants found.")
        return
    
    print(f"[EVOLUTION EVAL] Found {len(expired)} expired variant(s) to evaluate.")
    
    for variant in expired:
        print(f"\n[EVOLUTION EVAL] Evaluating Variant #{variant.id}: '{variant.name}'")
        print(f"  Started:    {variant.test_start}")
        print(f"  Duration:   {variant.test_duration_days} days")
        print(f"  Starting $: ${variant.starting_cash}")
        print(f"  Final $:    ${variant.virtual_balance}")
        print(f"  PnL:        {variant.virtual_pnl_pct:+.2f}%")
        print(f"  Sharpe:     {variant.sharpe_ratio:.4f}")
        print(f"  Max DD:     {variant.max_drawdown_pct:.2f}%")
        print(f"  Win Rate:   {variant.win_rate:.1f}%")
        print(f"  Trades:     {variant.virtual_trades_count}")
        
        # Get parent trader's performance over the same period
        parent = variant.parent_trader
        parent_pnl_pct = 0.0
        if parent:
            trades = parent.trades.filter(
                timestamp__gte=variant.test_start,
                timestamp__lte=variant.test_start + __import__('datetime').timedelta(days=variant.test_duration_days)
            )
            bought = sum(float(t.notional_value) for t in trades if t.action == 'BUY')
            sold = sum(float(t.notional_value) for t in trades if t.action == 'SELL')
            parent_realized = sold - bought
            parent_initial = float(variant.starting_cash)  # Same starting point
            if parent_initial > 0:
                parent_pnl_pct = (parent_realized / parent_initial) * 100
        
        print(f"  Parent PnL: {parent_pnl_pct:+.2f}% (same period)")
        
        # Decision logic
        variant_better = variant.virtual_pnl_pct > parent_pnl_pct
        variant_safer = variant.max_drawdown_pct <= max(10.0, abs(parent_pnl_pct) * 2)  # Drawdown cap
        margin = abs(variant.virtual_pnl_pct - parent_pnl_pct)
        
        if variant.virtual_trades_count < 5:
            # Too few trades to judge — delete
            msg = f'Inconclusive: only {variant.virtual_trades_count} trades in {variant.test_duration_days} days'
            print(f"  VERDICT: FAILED (only {variant.virtual_trades_count} trades — insufficient data)")
            _delete_failed_variant(variant, msg)
            continue
        
        if margin < 0.5:
            # Inconclusive — less than 0.5% difference
            # Extend by 7 days if under 28 day cap, otherwise fail
            if variant.test_duration_days < 28:
                variant.test_duration_days += 7
                variant.save(update_fields=['test_duration_days'])
                print(f"  VERDICT: INCONCLUSIVE (margin {margin:.2f}%). Extended to {variant.test_duration_days} days.")
                continue
            else:
                msg = f'Inconclusive after {variant.test_duration_days} days (margin: {margin:.2f}%)'
                print(f"  VERDICT: FAILED (inconclusive after max 28 days)")
                _delete_failed_variant(variant, msg)
                continue
        
        if variant_better and variant_safer:
            print(f"  VERDICT: *** WINNER *** — Variant outperformed by {margin:.2f}%")
            variant.status = 'PENDING'
            variant.save(update_fields=['status'])
            
            # Create promotion alert
            SystemAlert.objects.create(
                level='EVOLUTION',
                title=f'🧬 Variant #{variant.id} Ready for Promotion',
                message=(
                    f"Variant '{variant.name}' outperformed the active model after "
                    f"{variant.test_duration_days} days of virtual paper trading.\n\n"
                    f"Variant PnL: {variant.virtual_pnl_pct:+.2f}% | "
                    f"Parent PnL: {parent_pnl_pct:+.2f}%\n"
                    f"Sharpe: {variant.sharpe_ratio:.2f} | "
                    f"Max Drawdown: {variant.max_drawdown_pct:.2f}% | "
                    f"Win Rate: {variant.win_rate:.1f}%\n\n"
                    f"Approve promotion from the dashboard to deploy this variant as the new production model."
                ),
                related_model_reference=str(variant.id),
            )
            
            # Send notification email
            try:
                from src.reporting.email_dispatcher import send_mutator_alert
                send_mutator_alert(
                    f"Neural Evolution: Variant #{variant.id} is ready for promotion! "
                    f"({variant.virtual_pnl_pct:+.2f}% vs {parent_pnl_pct:+.2f}%)",
                    diff_text=None,
                    pdf_path=None,
                    simple_summary=(
                        f"After {variant.test_duration_days} days of virtual paper trading, "
                        f"variant '{variant.name}' beat the active model. "
                        f"Approve from the dashboard to go live."
                    )
                )
            except Exception as e:
                print(f"  Email notification failed: {e}")
        else:
            reason = []
            if not variant_better:
                reason.append(f"underperformed ({variant.virtual_pnl_pct:+.2f}% vs {parent_pnl_pct:+.2f}%)")
            if not variant_safer:
                reason.append(f"excessive drawdown ({variant.max_drawdown_pct:.2f}%)")
            
            msg = f'Lost evaluation: {", ".join(reason)}'
            print(f"  VERDICT: FAILED — {', '.join(reason)}")
            _delete_failed_variant(variant, msg)
    
    print(f"\n[EVOLUTION EVAL] Evaluation complete.")


if __name__ == "__main__":
    evaluate_expired_variants()
