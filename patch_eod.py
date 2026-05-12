import re

with open('src/reporting/eod_generator.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update the first gemini instance
old_prompt_1 = "Write a concise (2-3 sentences), highly analytical paragraph analyzing the performance trend between yesterday, today, and the rolling week, and confidently suggest one architectural or strategic improvement for the model."
new_prompt_1 = "Provide a comprehensive, multi-paragraph diagnostic breakdown analyzing the performance trend between yesterday, today, and the rolling week. Give explicit actionable advice on strategy improvements, and explicitly state your official recommendation on whether this exact model file is mathematically safe to deploy into LIVE PRODUCTION TRADING."

if old_prompt_1 in text:
    text = text.replace(old_prompt_1, new_prompt_1)

# 2. Update the second gemini instance
text = text.replace("gemini-1.5-flash-latest", "gemini-2.5-flash")

old_prompt_2 = "Provide 3 quick bullet points assessing today's performance and risk, contrasting it against the multi-day trend (yesterday, 7D, 30D). Also provide 1 explicit, actionable piece of advice on how the trader could improve profit margins tomorrow."
new_prompt_2 = "Provide an exhaustive, multi-paragraph diagnostic breakdown assessing today's performance and risk vs the multi-day trend. Provide deep actionable advice on improving profit margins, and explicitly state if this physical model is recommended for deployment into real Live Trading."

if old_prompt_2 in text:
    text = text.replace(old_prompt_2, new_prompt_2)

with open('src/reporting/eod_generator.py', 'w', encoding='utf-8') as f:
    f.write(text)

