import re

with open('src/core/async_engine.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace(
    '''            send_sos_alert(args.model_path, tb_str)
            
            # Auto-healing Framework Injection
            from src.core.code_rewriter import orchestrate_rewrite
            logger.info("Initializing Synced Cognitive Rollback Rewrite using Traceback...")
            orchestrate_rewrite(crash_log=tb_str)''',
    '''            await asyncio.to_thread(send_sos_alert, args.model_path, tb_str)
            
            # Auto-healing Framework Injection
            from src.core.code_rewriter import orchestrate_rewrite
            logger.info("Initializing Synced Cognitive Rollback Rewrite using Traceback...")
            await asyncio.to_thread(orchestrate_rewrite, crash_log=tb_str)'''
)

with open('src/core/async_engine.py', 'w', encoding='utf-8') as f:
    f.write(text)
