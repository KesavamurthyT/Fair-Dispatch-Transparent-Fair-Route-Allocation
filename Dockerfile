# Fair Dispatch System — Multi-purpose Docker Image
# Used for both 'web' (FastAPI/Uvicorn) and 'worker' (Celery) services.
# The CMD is overridden per service in docker-compose.yml.

FROM python:3.12-slim AS base

# System dependencies for psycopg2, scipy, ortools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create model storage directory
RUN mkdir -p models/

# Expose FastAPI port (only used by web service)
EXPOSE 8000

# Default command (overridden in docker-compose.yml)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
