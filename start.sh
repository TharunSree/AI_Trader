#!/usr/bin/env bash
# Generic app launcher
# Runs a Celery worker in the background and Daphne in the foreground.

# Start Celery in background (low concurrency to save memory)
celery -A trader_project worker --loglevel=info --concurrency=1 &

# Start Daphne (ASGI server) in foreground
daphne -b 0.0.0.0 -p $PORT trader_project.asgi:application
