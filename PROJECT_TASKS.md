# MNIST Cloud K8s Pipeline - Project Tasks

**Project**: End-to-End MNIST Deep Learning Service on Google Kubernetes Engine (GKE)  
**Started**: 2025-11-18  
**Status**: Completed

---

## Project Overview

This project implements a complete ML pipeline on Kubernetes with two main components:

1. **Training Pipeline**: A Kubernetes Job that trains an MNIST model and saves it to persistent storage (PVC)
2. **Inference Pipeline**: A FastAPI web service deployed as a Kubernetes Deployment that loads the trained model and serves predictions via a web interface

**Key Requirements:**
- CPU-only PyTorch (no GPU)
- Platform-agnostic Python tooling (no bash scripts)
- Persistent storage sharing between training and inference via PVC
- Self-healing capabilities (replicas, health probes)
- Complete workflow: Data → Training → Model → Inference → Web UI

---

## Task Tracking

### **Phase 1: Project Foundation** ✅ COMPLETED

- [x] **Task 1.1**: Create `.gitignore` with Python, venv, data, and local model exclusions
- [x] **Task 1.2**: Create `.env.example` with Docker registry and image name configuration
- [x] **Task 1.3**: Create `tools/__init__.py` (package marker)
- [x] **Task 1.4**: Create `tools/config.py` with environment variable reading and path computation

**Phase Status**: ✅ All tasks completed

---

### **Phase 2: Training Component** ✅ COMPLETED

- [x] **Task 2.1**: Create `training/model_def.py` with MnistCNN class (2 conv + 2 FC layers) and `create_model()` helper
- [x] **Task 2.2**: Create `training/train.py` with complete training loop, MNIST download, model saving, and metadata export
- [x] **Task 2.3**: Create `training/requirements.txt` with torch, torchvision, tqdm
- [x] **Task 2.4**: Create `training/Dockerfile` with Python 3.10-slim base, system dependencies, and optimized image size

**Phase Status**: ✅ All tasks completed  
**Testing**: ✅ Validated locally with Docker Desktop - training works, model saved successfully

---

### **Phase 3: Kubernetes Storage & Training Job** ✅ COMPLETED

- [x] **Task 3.1**: Create `k8s/storage/pvc-model.yaml` with 2Gi ReadWriteOnce PVC named `mnist-model-pvc`
- [x] **Task 3.2**: Create `k8s/training/job-train.yaml` with Job spec, PVC mount at `/mnt/model`, and environment variables

**Phase Status**: ✅ All tasks completed  
**Testing**: ✅ Deployed to Docker Desktop Kubernetes - Job completed successfully, model saved to PVC (99.14% accuracy)  
**Structure**: ✅ Dual manifests created (`k8s/local/` and `k8s/gke/`) for local and cloud deployments

---

### **Phase 4: Inference Component - Model & Config** ✅ COMPLETED

- [x] **Task 4.1**: Create `inference/app/__init__.py` (package marker)
- [x] **Task 4.2**: Create `inference/app/model_def.py` with MnistCNN class and `load_model()` helper for CPU inference
- [x] **Task 4.3**: Create `inference/app/config.py` with MODEL_DIR, MODEL_PATH, APP_HOST, APP_PORT configuration
- [x] **Task 4.4**: Create `inference/app/schemas.py` with Pydantic models for PredictionResponse

**Phase Status**: ✅ All tasks completed

---

### **Phase 5: Inference Component - FastAPI Application** ✅ COMPLETED

- [x] **Task 5.1**: Create `inference/app/main.py` with FastAPI app, startup model loading, and `/healthz` endpoint
- [x] **Task 5.2**: Add `GET /` route to `main.py` that renders the HTML template
- [x] **Task 5.3**: Add `POST /predict` route to `main.py` for image upload, preprocessing, and prediction
- [x] **Task 5.4**: Create `inference/app/templates/index.html` with file upload form and result display
- [x] **Task 5.5**: Create empty `inference/app/static/` directory (for future CSS/JS if needed)

**Phase Status**: ✅ All tasks completed  
**Testing**: ✅ FastAPI app tested locally - web UI works, predictions accurate, beautiful interface

---

### **Phase 6: Inference Component - Docker** ✅ COMPLETED

- [x] **Task 6.1**: Create `inference/requirements.txt` with torch, torchvision, fastapi, uvicorn, pillow, numpy, jinja2
- [x] **Task 6.2**: Create `inference/Dockerfile` with Python 3.10-slim, app copy, port 8000 expose, and uvicorn CMD

**Phase Status**: ✅ All tasks completed  
**Note**: Updated requirements.txt to include Pydantic 2.x for Python 3.13 compatibility

---

### **Phase 7: Kubernetes Inference Deployment & Service** ✅ COMPLETED

- [x] **Task 7.1**: Create `k8s/inference/deployment-infer.yaml` with 2 replicas, PVC mount, liveness/readiness probes, and self-healing comments
- [x] **Task 7.2**: Create `k8s/inference/service-infer.yaml` with LoadBalancer type, port 80→8000 mapping

**Phase Status**: ✅ All tasks completed  
**Testing**: ✅ Deployed to Kubernetes, 2 replicas running, health checks passing, accessible at http://localhost/  
**Self-Healing**: ✅ Comprehensive documentation with 7 mechanisms (replicas, probes, restarts, rolling updates, resource limits)

---

### **Phase 8: Python Tooling Scripts** ✅ COMPLETED

- [x] **Task 8.1**: Create `tools/build_images.py` with argparse, docker build/push functions for train and infer images
- [x] **Task 8.2**: Create `tools/deploy_k8s.py` with kubectl apply commands in sequence and user input pause
- [x] **Task 8.3**: Create `tools/run_local.py` with local Docker run commands for train/infer with volume mounts

**Phase Status**: ✅ All tasks completed

---

### **Phase 9: Documentation** ✅ COMPLETED

- [x] **Task 9.1**: Create `README.md` with project overview, architecture diagram, and workflow explanation
- [x] **Task 9.2**: Add quickstart guide to `README.md` with step-by-step deployment instructions
- [x] **Task 9.3**: Add self-healing explanation to `README.md` covering replicas, probes, and automatic recovery
- [x] **Task 9.4**: Add troubleshooting section and usage examples to `README.md`

**Phase Status**: ✅ All tasks completed  
**Documentation**: ✅ Comprehensive README with architecture, quick start, self-healing (7 mechanisms), troubleshooting, and usage examples

---

## Progress Summary

| Phase | Tasks Complete | Total Tasks | Status |
|-------|----------------|-------------|--------|
| Phase 1: Project Foundation | 4 | 4 | ✅ Complete |
| Phase 2: Training Component | 4 | 4 | ✅ Complete |
| Phase 3: K8s Storage & Training | 2 | 2 | ✅ Complete |
| Phase 4: Inference - Model & Config | 4 | 4 | ✅ Complete |
| Phase 5: Inference - FastAPI App | 5 | 5 | ✅ Complete |
| Phase 6: Inference - Docker | 2 | 2 | ✅ Complete |
| Phase 7: K8s Inference Deploy | 2 | 2 | ✅ Complete |
| Phase 8: Python Tooling | 3 | 3 | ✅ Complete |
| Phase 9: Documentation | 4 | 4 | ✅ Complete |
| **TOTAL** | **30** | **30** | **100% Complete** |

---

## File Structure Checklist

```
mnist-cloud-k8s-pipeline/
├── [x] .gitignore
├── [x] .env.example
├── [x] README.md
├── [x] PROJECT_TASKS.md (this file)
│
├── training/
│   ├── [x] train.py
│   ├── [x] model_def.py
│   ├── [x] requirements.txt
│   └── [x] Dockerfile
│
├── inference/
│   ├── app/
│   │   ├── [x] __init__.py
│   │   ├── [x] main.py
│   │   ├── [x] model_def.py
│   │   ├── [x] schemas.py
│   │   ├── [x] config.py
│   │   ├── templates/
│   │   │   └── [x] index.html
│   │   └── static/
│   ├── [x] requirements.txt
│   └── [x] Dockerfile
│
├── k8s/
│   ├── storage/
│   │   └── [x] pvc-model.yaml
│   ├── training/
│   │   └── [x] job-train.yaml
│   └── inference/
│       ├── [x] deployment-infer.yaml
│       └── [x] service-infer.yaml
│
└── tools/
    ├── [x] __init__.py
    ├── [x] config.py
    ├── [x] build_images.py
    ├── [x] deploy_k8s.py
    └── [x] run_local.py
```

---

## Notes & Decisions

### Design Decisions
- Using Python 3.10+ for all components
- PyTorch CPU-only (no GPU dependencies)
- FastAPI for inference web service
- PVC for model persistence between training and inference
- LoadBalancer service type for external access on GKE

### Technical Constraints
- All tooling must be platform-agnostic (Windows/Mac/Linux)
- No bash scripts - Python only
- Optimized Docker images (slim base, cleaned caches)
- Self-healing via K8s Deployment with health probes

### Testing Notes
- **Phase 2 Testing (2025-11-18)**:
  - ✅ Python training script tested locally (1 epoch test run)
  - ✅ Docker image built successfully (~8.5 minutes build time)
  - ✅ Docker container run with volume mount validated
  - ✅ Model training completed (5 epochs, ~98% accuracy achieved)
  - ✅ Model files saved to `local_model/`: `mnist_cnn.pt` and `metadata.json`
  - ✅ Volume mounting works correctly (critical for K8s PVC)

---

## Next Steps

**Current Phase**: Project Complete  
**Next Task**: None

---

*Last Updated: 2025-11-18 14:37:43*
