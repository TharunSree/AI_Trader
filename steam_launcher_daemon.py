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
        else:
            self.send_response(404)
            self.end_headers()

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
