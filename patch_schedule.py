import re

with open('trader_project/settings.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace Mon-Fri with Daily
text = text.replace(
    "'schedule': crontab(hour=16, minute=15, day_of_week='mon-fri'),",
    "'schedule': crontab(hour=16, minute=15), # Daily Execution"
)
text = text.replace(
    "'schedule': crontab(hour=16, minute=30, day_of_week='mon-fri'),",
    "'schedule': crontab(hour=16, minute=30), # Daily Execution"
)

with open('trader_project/settings.py', 'w', encoding='utf-8') as f:
    f.write(text)
