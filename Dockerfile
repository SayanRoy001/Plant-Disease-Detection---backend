# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Avoid interactive prompts
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (if pillow/torch need some libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libjpeg62-turbo libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement file first for layer caching
COPY backend/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY backend/ .

# Expose (Cloud Run/Render will set PORT)
ENV PORT=8080
EXPOSE 8080

# Health check (Docker specific - optional)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import requests, os; import sys; import time; import urllib.request;\nimport json;\nurl=f'http://127.0.0.1:{os.environ.get('PORT','8080')}/health';\nprint('Checking', url);\nurllib.request.urlopen(url)" || exit 1

# Start with gunicorn (more robust than flask dev server)
# Using 1 worker (model in memory) + 2 threads; tweak if more CPU
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "2", "--timeout", "120"]
