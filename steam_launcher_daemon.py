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

# Thread to periodically report active game heartbeats to Django
def heartbeat_worker():
    print("Arcade Lounge heartbeat monitor thread started...")
    last_reported_active = False
    
    while True:
        try:
            active_title = get_active_window_title()
            executables = get_running_executables()
            
            # Common game launcher and engine process list to auto-detect
            game_indicators = [".exe", "game", "unity", "unreal", "steam", "genshin", "wuwa", "wuthering", "wuwa"]
            
            active_process = ""
            active_path = ""
            is_running = False
            
            # 1. Simple heuristic: If active window is a game and process is in running tasks
            for exe in executables:
                # Basic check for game processes
                if any(ind in exe for ind in ["wuwa", "genshin", "elden", "cyberpunk", "starrail", "honkai"]):
                    active_process = exe
                    is_running = True
                    break
            
            # 2. Check if active window title matches typical game names
            if not is_running and active_title:
                title_lower = active_title.lower()
                if any(kw in title_lower for kw in ["wuthering waves", "genshin impact", "elden ring", "cyberpunk"]):
                    # Find matching process
                    for exe in executables:
                        if any(kw in exe for kw in ["wuwa", "genshin", "elden", "cyberpunk"]):
                            active_process = exe
                            is_running = True
                            break
            
            # If we found an active game, report it
            if is_running:
                payload = {
                    "active_process": active_process,
                    "path": active_path,
                    "window_title": active_title,
                    "is_running": True
                }
                last_reported_active = True
            else:
                # Report inactive
                payload = {
                    "active_process": "",
                    "path": "",
                    "window_title": "",
                    "is_running": False
                }
                last_reported_active = False

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
            # Silence exceptions during network disconnects
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
