#!/usr/bin/env bash
# ── Render Start Script ──
# Runs Celery worker in background + Daphne ASGI server in foreground.
# Combined into one service since Render free tier doesn't support workers.

# Start Celery in background (low concurrency to save memory)
celery -A trader_project worker --loglevel=info --concurrency=1 &

# Start Daphne (ASGI server) in foreground
daphne -b 0.0.0.0 -p $PORT trader_project.asgi:application
