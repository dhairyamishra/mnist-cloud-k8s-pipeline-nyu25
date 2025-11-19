"""
Configuration module for MNIST inference application.

Reads environment variables and defines application settings for:
- Model storage paths
- Application server configuration
"""
import os
from pathlib import Path

# Model storage configuration
# In Kubernetes, this will be the PVC mount point
MODEL_DIR = os.environ.get("MODEL_DIR", "/mnt/model")
MODEL_PATH = Path(MODEL_DIR) / "mnist_cnn.pt"
METADATA_PATH = Path(MODEL_DIR) / "metadata.json"

# Application server configuration
APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
APP_PORT = int(os.environ.get("APP_PORT", "8000"))

# Image preprocessing configuration (must match training)
# MNIST normalization values
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# Image dimensions
IMAGE_SIZE = 28  # MNIST images are 28x28 pixels

# Application metadata
APP_NAME = "MNIST Inference Service"
APP_VERSION = "1.0.0"
