import re

with open('src/core/code_rewriter.py', 'r', encoding='utf-8') as f:
    text = f.read()

old_block = '''        # Sandbox Protocol Phase 2: Syntax Validate
        try:
            compile(new_code, 'ppo_agent.py', 'exec')
            print("[COGNITIVE REWRITER] Syntax verification passed. Mutated logic synced to core.")
            
            # Send Notification Mail
            try:
                from src.reporting.email_dispatcher import send_mutator_alert
                reason = "Emergency Traceback Recovery" if crash_log else "Epoch Parameter Refinement"
                send_mutator_alert(f"Syntax Passed. Core recompiled successfully via {reason}.")
            except Exception as mail_err:
                print(f"[COGNITIVE REWRITER] Failed to dispatch mutation email: {mail_err}")
                
        except SyntaxError as syntax_e:
            raise RuntimeError(f"Syntax validation failed. The LLM wrote broken code: {syntax_e}")'''

new_block = '''        # Sandbox Protocol Phase 2: Syntax Validate
        try:
            compile(new_code, 'ppo_agent.py', 'exec')
            print("[COGNITIVE REWRITER] Syntax verification passed. Mutated logic synced to core.")
            
            # Generate Unified Diff securely
            import difflib
            diff_lines = list(difflib.unified_diff(
                current_code.splitlines(),
                new_code.splitlines(),
                fromfile='ppo_agent.py.bak',
                tofile='ppo_agent.py',
                lineterm=''
            ))
            diff_str = "\\n".join(diff_lines)
            
            # Send Notification Mail with Codebase Diff
            try:
                from src.reporting.email_dispatcher import send_mutator_alert
                reason = "Emergency Traceback Recovery" if crash_log else "Epoch Parameter Refinement"
                send_mutator_alert(f"Syntax Passed. Core recompiled successfully via {reason}.", diff_text=diff_str)
            except Exception as mail_err:
                print(f"[COGNITIVE REWRITER] Failed to dispatch mutation email: {mail_err}")
                
            # If this was an Epoch Refinement (not a mid-day crash), automatically trigger a robust training cycle
            # so the model can securely ingest the new PPO agent configurations over the weekend/night!
            if not crash_log:
                try:
                    from control_panel.models import TrainingJob
                    from control_panel.views import _spawn_background_process
                    import sys
                    
                    print("[COGNITIVE REWRITER] Spawning iterative backtesting processes against mutated configurations...")
                    for i in range(2):
                        job = TrainingJob.objects.create(
                            name=f"Mutant Configuration Node {i+1}",
                            feature_set_key="standard",
                            hyperparameter_key="standard",
                            window_size=10, 
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
                
        except SyntaxError as syntax_e:
            raise RuntimeError(f"Syntax validation failed. The LLM wrote broken code: {syntax_e}")'''

text = text.replace(old_block, new_block)

with open('src/core/code_rewriter.py', 'w', encoding='utf-8') as f:
    f.write(text)
