import json
import os

log_path = r"C:\Users\tarun\.gemini\antigravity\brain\88f0e2d2-507a-4507-8f1b-28751b4148a4\.system_generated\logs\transcript.jsonl"
if not os.path.exists(log_path):
    print("Log path does not exist!")
    exit()

print("User messages found in history:")
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get('type') == 'USER_INPUT' or data.get('source') == 'USER_EXPLICIT':
                print(f"--- Step {data.get('step_index')} ---")
                print(data.get('content'))
        except Exception as e:
            pass
