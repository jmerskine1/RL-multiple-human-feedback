# Dockerfile for main_gcp.py — Cloud Run deployment
# Build:  docker build -t pacman-feedback .
# Run:    docker run -p 8080:8080 -e GCS_BUCKET=... -e FLASK_SECRET=... pacman-feedback

FROM python:3.11-slim

# Non-interactive apt + headless matplotlib backend
ENV DEBIAN_FRONTEND=noninteractive \
    MPLBACKEND=agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python deps first (layer-caches if requirements unchanged)
COPY requirements_gcp.txt .
RUN pip install --no-cache-dir -r requirements_gcp.txt

# Copy application source
COPY agent.py envPacMan.py feedback.py ghost.py gcs_utils.py \
     main_gcp.py mylib.py trainer.py ./

# Copy templates and static assets
COPY templates/ templates/
COPY static/   static/
COPY sprites/  sprites/

# Cloud Run injects PORT; gunicorn binds to it.
# --workers=1 keeps _bundle_cache coherent within the instance.
# --threads=4 allows concurrent requests without multiple processes.
ENV PORT=8080
CMD exec gunicorn \
        --bind "0.0.0.0:${PORT}" \
        --workers 1 \
        --threads 4 \
        --timeout 120 \
        main_gcp:app
