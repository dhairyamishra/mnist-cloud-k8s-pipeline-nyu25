"""
Configuration module for MNIST K8s pipeline.
Reads environment variables and computes project paths.
"""
import os
from pathlib import Path

# Compute repository root (parent of tools directory)
ROOT = Path(__file__).resolve().parents[1]

# Docker registry configuration
DOCKER_REGISTRY = os.environ.get("DOCKER_REGISTRY", "gcr.io/YOUR_GCP_PROJECT")
TRAIN_IMAGE_NAME = os.environ.get("TRAIN_IMAGE_NAME", "mnist-train")
INFER_IMAGE_NAME = os.environ.get("INFER_IMAGE_NAME", "mnist-infer")

# Full image names
TRAIN_IMAGE = f"{DOCKER_REGISTRY}/{TRAIN_IMAGE_NAME}"
INFER_IMAGE = f"{DOCKER_REGISTRY}/{INFER_IMAGE_NAME}"

# Project directories
K8S_DIR = ROOT / "k8s"
TRAINING_DIR = ROOT / "training"
INFERENCE_DIR = ROOT / "inference"
LOCAL_MODEL_DIR = ROOT / "local_model"

# Ensure local model directory exists for local testing
LOCAL_MODEL_DIR.mkdir(exist_ok=True)
