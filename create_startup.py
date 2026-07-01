import os
import sys

def main():
    # 1. Resolve Windows Startup folder (shell:startup)
    appdata = os.environ.get('APPDATA')
    if not appdata:
        print("[ERROR] APPDATA environment variable not found. Are you running on Windows?")
        return

    startup_dir = os.path.join(appdata, r'Microsoft\Windows\Start Menu\Programs\Startup')
    vbs_path = os.path.join(startup_dir, 'arcade_lounge_daemon.vbs')

    # 2. Get server address (default to the active Linux Server IP)
    server_address = "192.168.29.165:8000"
    if len(sys.argv) > 1:
        server_address = sys.argv[1].strip()

    # 3. Resolve paths
    base_dir = os.path.abspath(os.path.dirname(__file__))
    venv_pythonw = os.path.join(base_dir, r'.venv\Scripts\pythonw.exe')
    script_path = os.path.join(base_dir, 'steam_launcher_daemon.py')

    if not os.path.exists(venv_pythonw):
        print(f"[ERROR] Could not find virtual environment pythonw.exe at: {venv_pythonw}")
        return
    if not os.path.exists(script_path):
        print(f"[ERROR] Could not find daemon script at: {script_path}")
        return

    # 4. Generate VBScript to launch pythonw silently in background (0 = Hidden, False = Do not wait)
    vbs_content = f'''Set WshShell = CreateObject("WScript.Shell")
WshShell.Run """{venv_pythonw}"" ""{script_path}"" ""{server_address}""", 0, False
'''

    try:
        with open(vbs_path, 'w', encoding='utf-8') as f:
            f.write(vbs_content)
        print(f"\n[SUCCESS] Silent Windows startup script created successfully!")
        print(f" -> Startup Link: {vbs_path}")
        print(f" -> Pointing to Server: {server_address}")
        print(f"\nThe daemon will now automatically start in the background when your PC boots.")
    except Exception as e:
        print(f"[ERROR] Failed to write startup file: {e}")

if __name__ == '__main__':
    main()
