#!/usr/bin/env bash
# Generic deployment bootstrap
# Installs dependencies, collects static, migrates DB, and creates the admin user if configured.

set -o errexit  # Exit on error

# Install CPU-only PyTorch first (saves ~1.5GB vs full CUDA build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt

# Collect static files (whitenoise serves them)
python manage.py collectstatic --no-input

# Run database migrations
python manage.py migrate

# Create superuser from environment variables only if it does not already exist
if [ "$DJANGO_SUPERUSER_USERNAME" ]; then
  python manage.py shell -c "from django.contrib.auth import get_user_model; User = get_user_model(); username = '${DJANGO_SUPERUSER_USERNAME}'; email = '${DJANGO_SUPERUSER_EMAIL:-admin@example.com}'; password = '${DJANGO_SUPERUSER_PASSWORD:-}'; exists = User.objects.filter(username=username).exists(); print('Superuser already exists, skipping creation.' if exists else 'Creating superuser...'); (None if exists else User.objects.create_superuser(username=username, email=email, password=password))"
fi
