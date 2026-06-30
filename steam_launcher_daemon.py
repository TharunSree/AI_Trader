from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import os
import json

class LauncherHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed_url.query)
        
        if parsed_url.path == '/launch':
            appid = params.get('appid', [None])[0]
            path = params.get('path', [None])[0]
            
            success = False
            message = ""
            
            if appid:
                try:
                    os.startfile(f"steam://rungameid/{appid}")
                    success = True
                    message = f"Successfully launched Steam app ID {appid}"
                except Exception as e:
                    message = f"Failed to start steam app: {str(e)}"
            elif path:
                try:
                    os.startfile(path)
                    success = True
                    message = f"Successfully launched local path {path}"
                except Exception as e:
                    message = f"Failed to start local file: {str(e)}"
            else:
                message = "Missing parameter: appid or path"
                
            self.send_response(200 if success else 400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            res = {
                "status": "success" if success else "error",
                "message": message
            }
            self.wfile.write(json.dumps(res).encode('utf-8'))
        elif parsed_url.path == '/detect':
            name = params.get('name', [None])[0]
            paths = []
            if name:
                paths = find_game_paths(name)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            res = {
                "status": "success",
                "paths": paths
            }
            self.wfile.write(json.dumps(res).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def find_game_paths(name):
    import glob
    
    # Common gaming roots
    drives = ['C', 'D', 'E', 'F']
    roots = []
    for d in drives:
        if os.path.exists(f"{d}:\\"):
            roots.extend([
                f"{d}:\\Games",
                f"{d}:\\Program Files",
                f"{d}:\\Program Files (x86)",
            ])
            # Scan parent drive directly for custom libraries (e.g. D:\SteamLibrary, E:\EpicGames)
            try:
                for entry in os.listdir(f"{d}:\\"):
                    full_entry = os.path.join(f"{d}:\\", entry)
                    if os.path.isdir(full_entry) and any(kw in entry.lower() for kw in ['steam', 'epic', 'gog', 'xbox', 'game', 'play', 'hoyo']):
                        roots.append(full_entry)
            except Exception:
                pass
            
    # Add user specific folders
    user_profile = os.environ.get('USERPROFILE')
    if user_profile:
        roots.extend([
            os.path.join(user_profile, 'Desktop'),
            os.path.join(user_profile, 'AppData', 'Local', 'Programs'),
        ])
        
    found_paths = []
    
    # Standardize search queries
    name_clean = name.lower().replace(" ", "").replace("'", "").replace("-", "")
    search_terms = [name_clean, name.lower()]
    
    for r in roots:
        if not os.path.isdir(r):
            continue
        try:
            # Walk up to 3 levels deep to remain fast
            for root_dir, dirs, files in os.walk(r):
                depth = root_dir[len(r):].count(os.sep)
                if depth > 3:
                    dirs.clear() # Stop walking this branch
                    continue
                    
                for f in files:
                    if f.endswith('.exe') or f.endswith('.lnk'):
                        f_lower = f.lower()
                        # Match if any term is in the file name or containing directory
                        if any(term in f_lower or term in root_dir.lower() for term in search_terms):
                            if all(skip not in f_lower for skip in ['uninstall', 'crash', 'reporter', 'setup', 'update', 'patcher', 'redist', 'python']):
                                full_path = os.path.join(root_dir, f)
                                if full_path not in found_paths:
                                    found_paths.append(full_path)
                                    if len(found_paths) >= 6:
                                        return found_paths
        except Exception:
            pass
    return found_paths

if __name__ == '__main__':
    port = 5555
    print(f"==================================================")
    print(f"      STEAM REMOTE LAUNCHER DAEMON ACTIVE         ")
    print(f"==================================================")
    print(f"Listening on http://0.0.0.0:{port}")
    print(f"Run this script in the background of your gaming rig.")
    print(f"Keep this window open to receive launch triggers.")
    print(f"==================================================")
    
    server = HTTPServer(('0.0.0.0', port), LauncherHandler)
    server.serve_forever()
