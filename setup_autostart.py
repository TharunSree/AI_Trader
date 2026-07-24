import os
import sys

def setup_windows_autostart():
    appdata = os.environ.get('APPDATA')
    if not appdata:
        print("[ERROR] APPDATA environment variable not found.")
        return False

    startup_dir = os.path.join(appdata, r'Microsoft\Windows\Start Menu\Programs\Startup')
    if not os.path.exists(startup_dir):
        print(f"[ERROR] Startup directory not found at: {startup_dir}")
        return False

    # Target VBS script path
    vbs_path = os.path.join(startup_dir, 'ai_trader_autostart.vbs')
    
    # Paths
    base_dir = os.path.abspath(os.path.dirname(__file__))
    pythonw_path = os.path.join(base_dir, r'.venv\Scripts\pythonw.exe')
    manage_path = os.path.join(base_dir, 'manage.py')
    daemon_path = os.path.join(base_dir, 'steam_launcher_daemon.py')

    vbs_content = f'''Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "{base_dir}"

' 1. Start Django Web Server & Trading Engine silently (0 = Hidden, False = Async)
WshShell.Run """{pythonw_path}"" ""{manage_path}"" runserver 0.0.0.0:8000", 0, False

' 2. Wait 3 seconds for server socket to open
WScript.Sleep 3000

' 3. Start Steam Game Detection Daemon silently
WshShell.Run """{pythonw_path}"" ""{daemon_path}"" http://127.0.0.1:8000", 0, False
'''

    try:
        # Remove old broken shortcut if present
        old_lnk = os.path.join(startup_dir, 'pythonw.exe.lnk')
        if os.path.exists(old_lnk):
            try:
                os.remove(old_lnk)
                print(f"[CLEANUP] Removed old shortcut: {old_lnk}")
            except Exception:
                pass

        with open(vbs_path, 'w', encoding='utf-8') as f:
            f.write(vbs_content)

        print(f"[SUCCESS] Auto-start script created successfully at:")
        print(f" -> {vbs_path}")
        print("\nDjango Server (Port 8000) and Game Daemon will now launch silently in the background on Windows boot.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to write autostart script: {e}")
        return False

if __name__ == '__main__':
    setup_windows_autostart()
