# control_panel/env_manager.py
from dotenv import dotenv_values, set_key
from pathlib import Path

ENV_FILE_PATH = Path(__file__).resolve().parent.parent / '.env'

def read_env_value(key: str) -> str:
    ENV_FILE_PATH.touch(exist_ok=True)
    return str(dotenv_values(ENV_FILE_PATH).get(key) or "")

def write_env_value(key: str, value: str):
    set_key(ENV_FILE_PATH, key, value)
