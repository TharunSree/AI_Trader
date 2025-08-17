# control_panel/env_manager.py
from dotenv import get_key, set_key
from pathlib import Path

ENV_FILE_PATH = Path('.') / '.env'

def read_env_value(key: str) -> str:
    ENV_FILE_PATH.touch(exist_ok=True)
    return get_key(ENV_FILE_PATH, key) or ""

def write_env_value(key: str, value: str):
    set_key(ENV_FILE_PATH, key, value)