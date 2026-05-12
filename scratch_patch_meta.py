import os

path = r"d:\AI_Trader\run_meta_trainer.py"
with open(path, "r", encoding="utf-8") as f:
    c = f.read()

target = "TARGET_EQUITY = float(meta_job.target_equity) if meta_job.target_equity else PRINCIPAL * 2"
replacement = target + "\n            TICKER_SYMBOL = meta_job.ticker if hasattr(meta_job, 'ticker') else 'SPY'\n        else:\n            TICKER_SYMBOL = 'SPY'"
c = c.replace(target, replacement)

c = c.replace('TICKER = ["SPY"]', 'TICKER = [TICKER_SYMBOL]')

with open(path, "w", encoding="utf-8") as f:
    f.write(c)
