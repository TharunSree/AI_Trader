import os
import sys
import json

# Redirect stdout and stderr if running under pythonw.exe (where they are None or write-blocked)
if sys.stdout is None or sys.stderr is None:
    class DummyWriter:
        def write(self, data):
            pass
        def flush(self):
            pass
    sys.stdout = DummyWriter()
    sys.stderr = DummyWriter()
import ctypes
import subprocess
import threading
import time
import urllib.request
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

# Server Configuration
DJANGO_SERVER_URL = "http://localhost:8000"
DAEMON_PORT = 5555

# Parse Server URL from command line arguments if provided
if len(sys.argv) > 1:
    arg_url = sys.argv[1].strip()
    if not arg_url.startswith("http"):
        DJANGO_SERVER_URL = f"http://{arg_url}"
    else:
        DJANGO_SERVER_URL = arg_url

# Bypass system proxies to ensure loopback traffic goes directly to localhost
try:
    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)
    urllib.request.install_opener(opener)
except Exception:
    pass

# Helper to check active foreground window on Windows
def get_active_window_title():
    if sys.platform == 'win32':
        try:
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            return buf.value
        except Exception:
            return ""
    else:
        # Linux active window check using xdotool
        try:
            out = subprocess.check_output("xdotool getwindowfocus getwindowname", shell=True, stderr=subprocess.DEVNULL)
            return out.decode('utf-8', errors='ignore').strip()
        except Exception:
            try:
                # Fallback using xprop
                out = subprocess.check_output("xprop -id $(xprop -root 32x _NET_ACTIVE_WINDOW | awk '{print $NF}') WM_NAME", shell=True, stderr=subprocess.DEVNULL)
                title = out.decode('utf-8', errors='ignore').strip()
                if '=' in title:
                    return title.split('=', 1)[1].strip().strip('"')
                return title
            except Exception:
                return ""

# Helper to scan active process executable names using lightweight tasklist command
def get_running_executables():
    executables = set()
    if sys.platform == 'win32':
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
    else:
        # Linux process scan
        try:
            output = subprocess.check_output("ps -eo comm", shell=True, stderr=subprocess.DEVNULL).decode('utf-8', errors='ignore')
            for line in output.strip().splitlines():
                name = line.strip().lower()
                executables.add(name)
                # Map to .exe naming compatibility for cross-platform matches
                executables.add(name + '.exe')
        except Exception:
            pass
    return executables

monitored_games = []
last_fetched_time = 0
has_warned = False

def fetch_monitored_list():
    global monitored_games, last_fetched_time, has_warned
    if time.time() - last_fetched_time < 20:
        return
    try:
        url = f"{DJANGO_SERVER_URL}/relax/api/process-heartbeat/"
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=3) as res:
            data = json.loads(res.read().decode('utf-8'))
            monitored_games = data.get('monitored_games', [])
            
            # Print success log upon successful connection
            if len(monitored_games) > 0:
                print(f"[INFO] Successfully connected to Django server at {DJANGO_SERVER_URL}. Monitoring {len(monitored_games)} active library titles.")
                has_warned = False
            last_fetched_time = time.time()
    except Exception as e:
        if not has_warned:
            print(f"[WARNING] Could not connect to Django server at {DJANGO_SERVER_URL} ({e}).")
            print(f" -> Please run the daemon pointing to your System 2 Server IP e.g.:")
            print(f"    python steam_launcher_daemon.py http://<SERVER_IP>:8000")
            has_warned = True
        last_fetched_time = time.time()

# Thread to periodically report active game heartbeats to Django
def heartbeat_worker():
    global monitored_games
    print("Arcade Lounge heartbeat monitor thread started...")
    
    last_tasklist_run = 0
    running_executables = set()
    last_state = None
    last_process = None
    
    while True:
        try:
            # 1. Pull latest game list from Django
            fetch_monitored_list()
            
            active_title = get_active_window_title().lower()
            active_title_clean = " ".join(active_title.split())
            
            # Avoid false positives from music players, browsers, and chat apps
            is_noise_window = any(x in active_title_clean for x in ['spotify', 'discord', 'chrome', 'firefox', 'msedge', 'opera', 'browser', 'youtube'])
            
            active_process = ""
            active_path = ""
            is_running = False
            steam_appid_running = 0
            
            # Check if a Steam game is running via Windows Registry
            if sys.platform == 'win32':
                try:
                    import winreg
                    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam\ActiveProcess")
                    val, _ = winreg.QueryValueEx(key, "RunningAppId")
                    steam_appid_running = int(val)
                except Exception:
                    pass
            
            # Step 1: High-efficiency foreground window title matching (0% RAM/CPU)
            matched_game = None
            if not is_noise_window and monitored_games:
                for game in monitored_games:
                    name_clean = " ".join(game['name'].lower().split())
                    # Match title fragments (e.g. "wuthering waves", "genshin impact")
                    if name_clean in active_title_clean:
                        matched_game = game
                        break
                    
                    # Try base name without subtitles (e.g. "The Witcher 3: Wild Hunt" -> "the witcher 3")
                    name_base = game['name'].split(':')[0].split(' - ')[0].lower().strip()
                    name_base_clean = " ".join(name_base.split())
                    if len(name_base_clean) >= 4 and (name_base_clean in active_title_clean or active_title_clean in name_base_clean):
                        matched_game = game
                        break
                        
                    # Secondary: check if any executable stem appears in the window title
                    # This catches games like NTE whose window title is "HTGame" not "Neverness to Everness"
                    for exe in game.get('executables', []):
                        exe_stem = exe.replace('.exe', '').lower()
                        if len(exe_stem) >= 4 and exe_stem in active_title_clean:
                            matched_game = game
                            break
                    if matched_game:
                        break
                    
            if matched_game:
                # Foreground window title matched the game directly!
                active_process = matched_game['executables'][0] if matched_game['executables'] else (matched_game['name'].lower() + '.exe')
                is_running = True
            elif steam_appid_running > 0:
                # Fallback to Steam Registry detection! This detects ANY running Steam game!
                active_process = f"steam_{steam_appid_running}"
                is_running = True
            elif monitored_games:
                # Step 2: Fallback to scanning running processes, but only once every 3 seconds to save RAM!
                if time.time() - last_tasklist_run >= 3:
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
            
            # Print transition logs to the terminal for debugging
            if is_running != last_state or active_process != last_process:
                status_str = f"ACTIVE ({active_process})" if is_running else "INACTIVE"
                print(f"[STATUS] Focus state changed to: {status_str} | Title: '{active_title}'")
                last_state = is_running
                last_process = active_process
 
            # Resolve our local LAN IP to auto-register with Django
            import socket
            local_ip = ""
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
            except Exception:
                pass
 
            # If we found an active game, report it
            if is_running:
                payload = {
                    "active_process": active_process,
                    "path": active_path,
                    "window_title": active_title,
                    "is_running": True,
                    "steam_app_id": steam_appid_running if steam_appid_running > 0 else None,
                    "local_ip": local_ip
                }
            else:
                # Report inactive
                payload = {
                    "active_process": "",
                    "path": "",
                    "window_title": "",
                    "is_running": False,
                    "steam_app_id": None,
                    "local_ip": local_ip
                }

            # Send heartbeat POST to Django
            url = f"{DJANGO_SERVER_URL}/relax/api/process-heartbeat/"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=3) as res:
                res_data = json.loads(res.read().decode('utf-8'))
                
                # Update monitored games list in real-time from heartbeat response
                if 'monitored_games' in res_data:
                    monitored_games = res_data['monitored_games']
                
                # Check for pull-based pending launch triggers returned by Django (firewall-proof fallback!)
                pending = res_data.get('pending_launches', [])
                for pl in pending:
                    appid = pl.get('appid')
                    game_path = pl.get('path')
                    
                    try:
                        if appid:
                            print(f"[DAEMON] Launching Steam game AppID: {appid}")
                            if sys.platform == 'win32':
                                subprocess.Popen(f"start steam://rungameid/{appid}", shell=True)
                            else:
                                subprocess.Popen(f"xdg-open steam://rungameid/{appid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        elif game_path:
                            print(f"[DAEMON] Launching local game: {game_path}")
                            if os.path.exists(game_path):
                                if sys.platform == 'win32':
                                    subprocess.Popen(f'start "" "{game_path}"', shell=True)
                                else:
                                    subprocess.Popen(f'xdg-open "{game_path}"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            else:
                                print(f"[DAEMON] Path not found: {game_path}")
                    except Exception as launch_err:
                        print(f"[DAEMON] Launch failed: {launch_err}")
                
        except Exception as e:
            # Silence network disconnect exceptions
            pass
            
        time.sleep(1)

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
                    if sys.platform == 'win32':
                        subprocess.Popen(f"start steam://rungameid/{appid}", shell=True)
                    else:
                        subprocess.Popen(f"xdg-open steam://rungameid/{appid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    success = True
                    message = f"Launched Steam AppID {appid}"
                elif game_path:
                    print(f"Launching local game: {game_path}")
                    if os.path.exists(game_path):
                        if sys.platform == 'win32':
                            subprocess.Popen(f'start "" "{game_path}"', shell=True)
                        else:
                            subprocess.Popen(f'xdg-open "{game_path}"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
