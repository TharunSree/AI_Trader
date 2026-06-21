#!/bin/bash
# ============================================================
# AI_Trader — Full System Boot Script
#
# Starts ALL services needed for the trading system:
#   1. Redis (event bus for market ticks + channels)
#   2. Daphne (Django ASGI server — dashboard + API)
#   3. Alpaca Stream (live crypto market data → Redis pub/sub)
#   4. Update Watcher (auto git-pull + deploy on code push)
#
# Once Daphne starts, Django's auto-restart daemon handles:
#   - Respawning all RUNNING paper traders
#   - Respawning all TESTING evolution variants
# ============================================================

PROJECT_DIR="/home/tharun/AI_Trader"
VENV_DIR="$PROJECT_DIR/.venv"
LOG_DIR="$PROJECT_DIR/logs"
PID_DIR="$PROJECT_DIR/logs/pids"

# Create directories
mkdir -p "$LOG_DIR" "$PID_DIR"

# Activate virtualenv
source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

# Load .env for API keys
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# ── 1. Redis ──
if ! pgrep -x "redis-server" > /dev/null 2>&1; then
    echo "[BOOT] Starting Redis server..."
    redis-server --daemonize yes --logfile "$LOG_DIR/redis.log" 2>&1 \
        || echo "[BOOT] Redis start failed (may be running as systemd service)"
else
    echo "[BOOT] Redis already running."
fi

# Wait for Redis to be ready
for i in {1..10}; do
    if redis-cli ping > /dev/null 2>&1; then
        echo "[BOOT] Redis is ready."
        break
    fi
    sleep 1
done

# ── 2. Alpaca Stream (live market data → Redis pub/sub) ──
# Kill any stale instances first
if [ -f "$PID_DIR/alpaca_stream.pid" ]; then
    OLD_PID=$(cat "$PID_DIR/alpaca_stream.pid")
    kill "$OLD_PID" 2>/dev/null
    sleep 1
fi
echo "[BOOT] Starting Alpaca Crypto Stream..."
nohup python "$PROJECT_DIR/alpaca_stream.py" \
    >> "$LOG_DIR/alpaca_stream.log" 2>&1 &
echo $! > "$PID_DIR/alpaca_stream.pid"
echo "[BOOT] Alpaca Stream PID: $(cat $PID_DIR/alpaca_stream.pid)"

# ── 3. Update Watcher (auto git-pull + deploy) ──
if [ -f "$PID_DIR/update_watcher.pid" ]; then
    OLD_PID=$(cat "$PID_DIR/update_watcher.pid")
    kill "$OLD_PID" 2>/dev/null
    sleep 1
fi
echo "[BOOT] Starting Update Watcher (auto-deploy)..."
nohup python "$PROJECT_DIR/update_watcher.py" \
    >> "$LOG_DIR/update_watcher.log" 2>&1 &
echo $! > "$PID_DIR/update_watcher.pid"
echo "[BOOT] Update Watcher PID: $(cat $PID_DIR/update_watcher.pid)"

# ── 4. Daphne (Django ASGI server — runs in foreground for systemd) ──
echo "[BOOT] Starting Daphne ASGI server on 0.0.0.0:8000..."
exec daphne \
    -b 0.0.0.0 \
    -p 8000 \
    --access-log "$LOG_DIR/daphne_access.log" \
    trader_project.asgi:application \
    >> "$LOG_DIR/daphne_server.log" 2>&1
