# Use slim Python image
FROM python:3.11-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Install system deps for requests (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements first for faster rebuilds
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app
COPY . /app
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8080
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/app/entrypoint.sh"]
