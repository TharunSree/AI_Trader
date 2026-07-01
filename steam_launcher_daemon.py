import os
import sys
import json
import ctypes
import subprocess
import threading
import time
import urllib.request
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

# Server Configuration
DJANGO_SERVER_URL = "http://localhost:8000" # Update as per your System 2 address
DAEMON_PORT = 5555

# Helper to check active foreground window on Windows
def get_active_window_title():
    try:
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value
    except Exception:
        return ""

# Helper to scan active process executable names using lightweight tasklist command
def get_running_executables():
    executables = set()
    try:
        # NH = No Headers, FO CSV = CSV format
        output = subprocess.check_output("tasklist /NH /FO CSV", shell=True).decode('utf-8', errors='ignore')
        for line in output.strip().splitlines():
            if line.startswith('"'):
                parts = line.split('","')
                if parts:
                    executables.add(parts[0].replace('"', '').lower())
    except Exception:
        pass
    return executables

monitored_games = []
last_fetched_time = 0

def fetch_monitored_list():
    global monitored_games, last_fetched_time
    if time.time() - last_fetched_time < 30:
        return
    try:
        url = f"{DJANGO_SERVER_URL}/relax/api/process-heartbeat/"
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=3) as res:
            data = json.loads(res.read().decode('utf-8'))
            monitored_games = data.get('monitored_games', [])
            last_fetched_time = time.time()
    except Exception:
        pass

# Thread to periodically report active game heartbeats to Django
def heartbeat_worker():
    print("Arcade Lounge heartbeat monitor thread started...")
    
    last_tasklist_run = 0
    running_executables = set()
    
    while True:
        try:
            # 1. Pull latest game list from Django
            fetch_monitored_list()
            
            # If no games exist, sleep and skip scanning to save RAM/CPU!
            if not monitored_games:
                time.sleep(3)
                continue
                
            active_title = get_active_window_title().lower()
            active_title_clean = " ".join(active_title.split())
            
            active_process = ""
            active_path = ""
            is_running = False
            
            # Step 1: High-efficiency foreground window title matching (0% RAM/CPU)
            matched_game = None
            for game in monitored_games:
                name_clean = " ".join(game['name'].lower().split())
                # Match title fragments (e.g. "wuthering waves", "genshin impact")
                if name_clean in active_title_clean:
                    matched_game = game
                    break
                    
            if matched_game:
                # Foreground window title matched the game directly!
                active_process = matched_game['executables'][0] if matched_game['executables'] else (matched_game['name'].lower() + '.exe')
                is_running = True
            else:
                # Step 2: Fallback to scanning running processes, but only once every 12 seconds to save RAM!
                if time.time() - last_tasklist_run >= 12:
                    running_executables = get_running_executables()
                    last_tasklist_run = time.time()
                
                # Check if any monitored executable is running in the background/foreground
                for game in monitored_games:
                    for exe in game['executables']:
                        if exe in running_executables:
                            active_process = exe
                            is_running = True
                            break
                    if is_running:
                        break
            
            # If we found an active game, report it
            if is_running:
                payload = {
                    "active_process": active_process,
                    "path": active_path,
                    "window_title": active_title,
                    "is_running": True
                }
            else:
                # Report inactive
                payload = {
                    "active_process": "",
                    "path": "",
                    "window_title": "",
                    "is_running": False
                }

            # Send heartbeat POST to Django
            url = f"{DJANGO_SERVER_URL}/relax/api/process-heartbeat/"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=3) as res:
                res.read()
                
        except Exception as e:
            # Silence network disconnect exceptions
            pass
            
        time.sleep(3)

# HTTP Request Handler for Launch commands
class LaunchRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        
        if parsed_url.path == '/launch':
            appid = query_params.get('appid', [None])[0]
            game_path = query_params.get('path', [None])[0]
            
            success = False
            message = ""
            
            try:
                if appid:
                    print(f"Launching Steam game AppID: {appid}")
                    subprocess.Popen(f"start steam://rungameid/{appid}", shell=True)
                    success = True
                    message = f"Launched Steam AppID {appid}"
                elif game_path:
                    print(f"Launching local game: {game_path}")
                    if os.path.exists(game_path):
                        subprocess.Popen(f'start "" "{game_path}"', shell=True)
                        success = True
                        message = f"Launched local game {game_path}"
                    else:
                        message = f"Path not found: {game_path}"
                else:
                    message = "No launch targets specified."
            except Exception as ex:
                message = f"Launch failed: {ex}"
                
            self.send_response(200 if success else 400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                "status": "success" if success else "error",
                "message": message
            }
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run_http_server():
    server_address = ('', DAEMON_PORT)
    httpd = HTTPServer(server_address, LaunchRequestHandler)
    print(f"Arcade Launch daemon running on port {DAEMON_PORT}...")
    httpd.serve_forever()

if __name__ == '__main__':
    # Start heartbeat thread
    t = threading.Thread(target=heartbeat_worker, daemon=True)
    t.start()
    
    # Start HTTP server
    run_http_server()
