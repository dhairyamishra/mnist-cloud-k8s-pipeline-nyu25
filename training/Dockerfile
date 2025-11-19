# Training Container Dockerfile
# Base image: Python 3.10 slim for smaller image size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for PyTorch and torchvision
# - build-essential: C/C++ compilers for building Python packages
# - libjpeg-dev: JPEG image support for torchvision
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

# Copy training code and model definition
COPY train.py .
COPY model_def.py .

# Set default environment variable for model directory
# This will be mounted as a PersistentVolumeClaim in Kubernetes
ENV MODEL_DIR=/mnt/model

# Default command: run the training script
CMD ["python", "train.py"]
