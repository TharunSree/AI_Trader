#!/bin/bash
# ============================================================
# AI_Trader — One-Time Setup Script for Auto-Boot
# 
# Run this ONCE on your Linux server to enable auto-start.
# After this, the ENTIRE system survives reboots automatically:
#   - Redis, Daphne, Alpaca Stream, Update Watcher
#   - All running bots and testing variants
# ============================================================

set -e

PROJECT_DIR="/home/tharun/AI_Trader"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║   AI_Trader — Full Auto-Boot Setup                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# 1. Make scripts executable
echo "[1/5] Making scripts executable..."
chmod +x "$PROJECT_DIR/scripts/start_server.sh"
echo "  ✓ start_server.sh"

# 2. Create required directories
echo "[2/5] Creating log directories..."
mkdir -p "$PROJECT_DIR/logs/pids"
echo "  ✓ logs/ and logs/pids/"

# 3. Copy service file to systemd
echo "[3/5] Installing systemd service..."
sudo cp "$PROJECT_DIR/scripts/ai_trader.service" /etc/systemd/system/ai_trader.service
echo "  ✓ Copied to /etc/systemd/system/"

# 4. Reload systemd and enable the service
echo "[4/5] Enabling auto-start on boot..."
sudo systemctl daemon-reload
sudo systemctl enable ai_trader
echo "  ✓ ai_trader.service enabled (starts on every boot)"

# 5. Start the service now
echo "[5/5] Starting the service..."
sudo systemctl start ai_trader
sleep 5

# Check status
if sudo systemctl is-active --quiet ai_trader; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║   ✅ SUCCESS — Full trading system is running!           ║"
    echo "║                                                          ║"
    echo "║   On every reboot, the system will automatically:        ║"
    echo "║     1. Start Redis (event bus)                           ║"
    echo "║     2. Start Alpaca Stream (live market data)            ║"
    echo "║     3. Start Update Watcher (auto git-pull deploy)       ║"
    echo "║     4. Start Daphne (Django dashboard + API)             ║"
    echo "║     5. Respawn all RUNNING bots (30s after Django boot)  ║"
    echo "║     6. Respawn all TESTING variants                      ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    echo "Useful commands:"
    echo "  sudo systemctl status ai_trader     # Check status"
    echo "  sudo journalctl -u ai_trader -f     # Live logs"  
    echo "  sudo systemctl restart ai_trader    # Manual restart"
    echo "  sudo systemctl stop ai_trader       # Stop everything"
    echo ""
    echo "Log files:"
    echo "  $PROJECT_DIR/logs/daphne_server.log   # Django server"
    echo "  $PROJECT_DIR/logs/alpaca_stream.log    # Market data stream"
    echo "  $PROJECT_DIR/logs/update_watcher.log   # Auto-updater"
    echo ""
else
    echo ""
    echo "⚠️  Service may need a moment to fully start."
    echo "Check with: sudo systemctl status ai_trader"
    echo "View logs:  sudo journalctl -u ai_trader -n 50"
fi
