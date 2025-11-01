#!/usr/bin/env bash
set -e

# load .env if present (only in dev)
if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Default gunicorn command
exec gunicorn -c /app/gunicorn_conf.py app:app
