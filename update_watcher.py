import subprocess
import time
import sys
import os
import logging
from pathlib import Path

# Create logs directory if not exists
os.makedirs("logs", exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/update_watcher.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("update_watcher")

def run_cmd(cmd):
    """Runs a shell command and returns (success, stdout, stderr)"""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout.strip(), result.stderr.strip()

def check_for_updates():
    """Checks remote origin/master for new updates."""
    # Fetch origin
    success, _, err = run_cmd(['git', 'fetch', 'origin'])
    if not success:
        return False, None, None
        
    _, local_sha, _ = run_cmd(['git', 'rev-parse', 'HEAD'])
    _, remote_sha, _ = run_cmd(['git', 'rev-parse', 'origin/master'])
    
    if local_sha != remote_sha:
        return True, local_sha, remote_sha
    return False, local_sha, remote_sha

def execute_update(stable_sha):
    """Performs the upgrade process (git pull, pip install, migrate, reload)"""
    logger.info("Update detected! Starting auto-update protocol...")
    
    # 1. Git pull
    logger.info("Syncing remote master repository...")
    success, out, err = run_cmd(['git', 'reset', '--hard', 'origin/master'])
    if not success:
        logger.error(f"Git reset failed: {err}")
        rollback(stable_sha)
        return False
        
    # 2. Pip install
    logger.info("Installing updated dependencies...")
    pip_path = get_executable_path("pip")
    success, out, err = run_cmd([pip_path, 'install', '-r', 'requirements.txt'])
    if not success:
        logger.error(f"Pip install failed: {err}")
        rollback(stable_sha)
        return False
        
    # 3. Database migrations
    logger.info("Executing database migrations...")
    python_path = get_executable_path("python")
    success, out, err = run_cmd([python_path, 'manage.py', 'migrate'])
    if not success:
        logger.error(f"Database migrations failed: {err}")
        rollback(stable_sha)
        return False
        
    # 4. Restart Python Server
    logger.info("Deploy succeeded. Restarting trading bot service...")
    restart_server()
    return True

def rollback(stable_sha):
    """Rollback to the last known working git commit if the update process crashed."""
    logger.warning(f"AUTO-ROLLBACK: Restoring previous stable release ({stable_sha[:7]})...")
    run_cmd(['git', 'reset', '--hard', stable_sha])
    pip_path = get_executable_path("pip")
    run_cmd([pip_path, 'install', '-r', 'requirements.txt'])
    python_path = get_executable_path("python")
    run_cmd([python_path, 'manage.py', 'migrate'])
    logger.info("Rollback complete. Restarting server to restore online state.")
    restart_server()

def get_executable_path(name):
    """Finds binary inside virtual env or returns fallback."""
    for folder in ['.venv', 'venv']:
        binary = Path(folder) / "bin" / name if os.name != 'nt' else Path(folder) / "Scripts" / f"{name}.exe"
        if binary.exists():
            return str(binary.absolute())
    # Fallback to current runtime
    return sys.executable if name == "python" else str(Path(sys.executable).parent / name)

def restart_server():
    """Restarts Django server by terminating the running manage.py runserver process or calling systemctl."""
    try:
        import psutil
        killed = False
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            cmd = proc.info['cmdline']
            if cmd and 'manage.py' in cmd and 'runserver' in cmd:
                logger.info(f"Terminating server PID {proc.info['pid']}...")
                proc.kill()
                killed = True
        if killed:
            logger.info("Server process terminated. Wrapper daemon will restart it.")
            return
    except Exception as e:
        logger.warning(f"Could not kill process via psutil: {e}")
        
    # Systemd fallback restart command
    success, out, err = run_cmd(['sudo', 'systemctl', 'restart', 'jarvis_brain'])
    if success:
        logger.info("Server restarted via systemctl.")
    else:
        logger.error(f"Systemctl restart failed: {err}")

def main():
    logger.info("Jarvis Core Update Watcher initialized.")
    logger.info("Polling remote repository every 30 seconds...")
    while True:
        try:
            has_update, local_sha, remote_sha = check_for_updates()
            if has_update:
                execute_update(local_sha)
        except Exception as e:
            logger.error(f"Watcher exception in poll loop: {e}")
        time.sleep(30)

if __name__ == "__main__":
    main()
