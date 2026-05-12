import sqlite3
import os

db_path = os.path.join("d:\\AI_Trader", "db.sqlite3")
print(f"Connecting to {db_path}")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    queries = [
        "ALTER TABLE control_panel_systemsettings ADD COLUMN lockscreen_password varchar(128);",
        "ALTER TABLE control_panel_systemsettings ADD COLUMN idle_lock_minutes integer DEFAULT 5;",
        "ALTER TABLE control_panel_systemsettings ADD COLUMN idle_logout_minutes integer DEFAULT 30;"
    ]
    
    for q in queries:
        try:
            cursor.execute(q)
            print(f"Executed: {q}")
        except Exception as e:
            print(f"Error: {e}")
            
    conn.commit()
    conn.close()
    print("Done fixing SQLite.")
except Exception as e:
    print(f"Failed: {e}")
