# Inference Container Dockerfile
# Base image: Python 3.10 slim for smaller image size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for PyTorch, Pillow, and FastAPI
# - build-essential: C/C++ compilers for building Python packages
# - libjpeg-dev: JPEG image support for Pillow
# - zlib1g-dev: Compression library for image processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: Don't cache pip packages (reduces image size)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory
COPY app/ ./app/

# Set default environment variable for model directory
# This will be mounted as a PersistentVolumeClaim in Kubernetes
ENV MODEL_DIR=/mnt/model

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Health check (optional but recommended for production)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/healthz')" || exit 1

# Default command: run uvicorn server
# --host 0.0.0.0: Listen on all network interfaces
# --port 8000: Use port 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
