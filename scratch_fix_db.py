import os
import django
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "trader_project.settings")
django.setup()

from django.db import connection

queries = [
    "ALTER TABLE control_panel_systemsettings ADD COLUMN lockscreen_password varchar(128);",
    "ALTER TABLE control_panel_systemsettings ADD COLUMN idle_lock_minutes integer DEFAULT 5;",
    "ALTER TABLE control_panel_systemsettings ADD COLUMN idle_logout_minutes integer DEFAULT 30;"
]

with connection.cursor() as cursor:
    for q in queries:
        try:
            cursor.execute(q)
            print(f"Executed: {q}")
        except Exception as e:
            print(f"Error executing {q}: {e}")
