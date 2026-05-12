with open('control_panel/views.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''    if not should_throttle:
        return snapshot'''

replacement = '''    if not should_throttle:
        # Auto-Resume logic if resources recover massively (10% cooldown buffer)
        if (snapshot['system_used_percent'] is not None and snapshot['system_used_percent'] < snapshot['threshold_percent'] - 10) and total_runner_mb < (snapshot['trader_limit_mb'] * 0.8):
            auto_paused = PaperTrader.objects.filter(status='PAUSED', error_message='Paused automatically due to memory pressure.')
            for t in auto_paused:
                _resume_trader_instance(t)
                total_runner_mb += (_get_process_memory_mb(t.celery_task_id) or 0)
                if total_runner_mb >= snapshot['trader_limit_mb'] * 0.9:
                    break
        snapshot['running_trader_memory_mb'] = round(total_runner_mb, 1)
        return snapshot'''

if target in text:
    text = text.replace(target, replacement)
    with open('control_panel/views.py', 'w', encoding='utf-8') as f:
        f.write(text)
