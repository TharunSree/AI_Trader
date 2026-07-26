import os
import sys

def setup_windows_autostart(server_url="http://192.168.29.165:8000"):
    appdata = os.environ.get('APPDATA')
    if not appdata:
        print("[ERROR] APPDATA environment variable not found.")
        return False

    startup_dir = os.path.join(appdata, r'Microsoft\Windows\Start Menu\Programs\Startup')
    if not os.path.exists(startup_dir):
        print(f"[ERROR] Startup directory not found at: {startup_dir}")
        return False

    # Ensure URL formatting
    if not server_url.startswith('http://') and not server_url.startswith('https://'):
        server_url = f"http://{server_url}"

    # Target VBS script path
    vbs_path = os.path.join(startup_dir, 'arcade_lounge_daemon.vbs')

    # Paths
    base_dir = os.path.abspath(os.path.dirname(__file__))
    pythonw_path = os.path.join(base_dir, r'.venv\Scripts\pythonw.exe')
    daemon_path = os.path.join(base_dir, 'steam_launcher_daemon.py')

    vbs_content = f'''Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "{base_dir}"

' Start Steam Game Detection Daemon silently pointing to Linux Server IP & Port
WshShell.Run """{pythonw_path}"" ""{daemon_path}"" ""{server_url}""", 0, False
'''

    try:
        # Clean up legacy/old shortcuts
        for old_file in ['pythonw.exe.lnk', 'ai_trader_autostart.vbs']:
            old_path = os.path.join(startup_dir, old_file)
            if os.path.exists(old_path):
                try:
                    os.remove(old_path)
                    print(f"[CLEANUP] Removed legacy file: {old_path}")
                except Exception:
                    pass

        with open(vbs_path, 'w', encoding='utf-8') as f:
            f.write(vbs_content)

        print(f"[SUCCESS] Auto-start script created successfully at:")
        print(f" -> {vbs_path}")
        print(f" -> Target Server IP & Port: {server_url}")
        print("\nGame Daemon will now launch silently in the background on Windows boot pointing to your server.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to write autostart script: {e}")
        return False

if __name__ == '__main__':
    server_ip = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.29.165:8000"
    setup_windows_autostart(server_ip)
