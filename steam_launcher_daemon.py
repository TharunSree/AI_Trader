import time
import urllib.request
import json
import subprocess
import sys

SERVER_URL = "http://localhost:8000"  # Update this with your server host IP

def get_running_processes():
    try:
        # Run standard Windows tasklist without showing cmd console window
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        output = subprocess.check_output(
            ["tasklist", "/FO", "CSV", "/NH"],
            startupinfo=startupinfo,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        ).decode('utf-8', errors='ignore')
        
        processes = set()
        for line in output.strip().split('\n'):
            if line:
                parts = line.split(',')
                if parts:
                    proc_name = parts[0].strip('"').lower()
                    processes.add(proc_name)
        return processes
    except Exception as e:
        print(f"Error checking processes: {e}")
        return set()

def main():
    server_host = sys.argv[1] if len(sys.argv) > 1 else SERVER_URL
    print(f"Starting lightweight playtime tracking daemon. Server: {server_host}")
    print("Consumes ~10-12MB RAM and 0% CPU. Checking every 60 seconds...")
    
    while True:
        try:
            # 1. Fetch games with tracked process names
            req = urllib.request.Request(f"{server_host}/relax/api/process-heartbeat/")
            with urllib.request.urlopen(req, timeout=5) as res:
                response = json.loads(res.read().decode('utf-8'))
                
            if response.get('status') == 'success':
                tracked_games = response.get('games', [])
                if tracked_games:
                    # 2. Check running processes on this gaming PC
                    running = get_running_processes()
                    
                    # 3. Check if any matches are running
                    active_ids = []
                    for game in tracked_games:
                        p_name = game.get('process_name', '').lower().strip()
                        if p_name in running:
                            active_ids.append(game.get('id'))
                            
                    # 4. Post heartbeat to increment playtime
                    if active_ids:
                        post_data = json.dumps({'game_ids': active_ids}).encode('utf-8')
                        post_req = urllib.request.Request(
                            f"{server_host}/relax/api/process-heartbeat/",
                            data=post_data,
                            headers={'Content-Type': 'application/json'}
                        )
                        with urllib.request.urlopen(post_req, timeout=5) as post_res:
                            post_response = json.loads(post_res.read().decode('utf-8'))
                            print(f"Logged active games heartbeat: {active_ids} -> {post_response}")
                            
        except Exception as e:
            print(f"Heartbeat query failed: {e}")
            
        time.sleep(60)

if __name__ == "__main__":
    main()
