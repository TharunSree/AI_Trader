#!/usr/bin/env bash
# ── Render Build Script ──
# Runs on every deploy. Installs deps, collects static, migrates DB.

set -o errexit  # Exit on error

# Install CPU-only PyTorch first (saves ~1.5GB vs full CUDA build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt

# Collect static files (whitenoise serves them)
python manage.py collectstatic --no-input

# Run database migrations
python manage.py migrate

# Create superuser from environment variables (fails silently if already exists)
if [ "$DJANGO_SUPERUSER_USERNAME" ]; then
  python manage.py createsuperuser --no-input || true
fi
