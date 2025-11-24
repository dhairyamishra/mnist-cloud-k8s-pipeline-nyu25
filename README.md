# MNIST Cloud Kubernetes Pipeline

A production-ready machine learning pipeline for training and serving MNIST digit classification models on Kubernetes. This project demonstrates end-to-end MLOps practices including containerization, orchestration, persistent storage, and self-healing deployments.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-1.28-blue.svg)](https://kubernetes.io/)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [Live GCP Deployment](#live-gcp-deployment)
- [Self-Healing Mechanisms](#self-healing-mechanisms)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

---

## 🎯 Project Overview

This project implements a complete machine learning pipeline for MNIST digit classification (0-9) using modern cloud-native technologies.

### **Training Pipeline**
- Trains a CNN on MNIST dataset
- Runs as Kubernetes Job
- Saves model to persistent storage (PVC)
- Achieves 99%+ accuracy

### **Inference Pipeline**
- FastAPI web service with beautiful UI
- Loads model from persistent storage
- 2 replicas for high availability
- Comprehensive self-healing mechanisms

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                        │
│                                                              │
│  Training Job  ──▶  PVC (Model Storage)  ◀──  Inference     │
│  (mnist-train)      (mnist_cnn.pt)           Deployment     │
│                                               (2 replicas)   │
│                                                    │         │
│                                              LoadBalancer    │
│                                                    │         │
└────────────────────────────────────────────────────┼─────────┘
                                                     │
                                                   Users
```

### **Data Flow**

1. **Training**: Job downloads MNIST → trains CNN → saves to PVC
2. **Inference**: Pods load model from PVC → serve predictions
3. **User**: Upload image → get prediction with confidence

---

## ✨ Features

### **Training**
- ✅ PyTorch CNN (99%+ accuracy)
- ✅ Automatic dataset download
- ✅ Model persistence to PVC
- ✅ Kubernetes Job execution

### **Inference**
- ✅ FastAPI with async support
- ✅ Beautiful web UI (drag & drop)
- ✅ Real-time predictions
- ✅ Health check endpoints
- ✅ Auto-generated API docs

### **Kubernetes**
- ✅ Persistent storage (PVC)
- ✅ High availability (2 replicas)
- ✅ Self-healing (7 mechanisms)
- ✅ Rolling updates (zero downtime)
- ✅ LoadBalancer for external access

---

## 🚀 Quick Start

### **Prerequisites**
- Docker Desktop (with Kubernetes enabled)
- kubectl
- Python 3.10+

### **Automated Deployment**

```bash
# 1. Clone repository
git clone https://github.com/yourusername/mnist-cloud-k8s-pipeline.git
cd mnist-cloud-k8s-pipeline

# 2. Build Docker images
python tools/build_images.py --all

# 3. Deploy to Kubernetes
python tools/deploy_k8s.py --env local

# 4. Access web UI
# Open browser: http://localhost/
```

### **Manual Deployment**

```bash
# 1. Build images
docker build -t mnist-train:latest -f training/Dockerfile training/
docker build -t mnist-infer:latest -f inference/Dockerfile inference/

# 2. Deploy storage
kubectl apply -f k8s/local/pvc-model.yaml

# 3. Run training (10-15 minutes)
kubectl apply -f k8s/local/job-train.yaml
kubectl wait --for=condition=complete job/mnist-train-job --timeout=900s

# 4. Deploy inference
kubectl apply -f k8s/inference/deployment-infer.yaml
kubectl apply -f k8s/inference/service-infer.yaml

# 5. Access at http://localhost/
```

---

## 🌐 Live GCP Deployment

This project is currently deployed on **Google Kubernetes Engine (GKE)** and accessible at:

### **🔗 Live Endpoint**
**Web UI**: http://34.173.103.239

### **API Endpoints**
- **Prediction**: `POST http://34.173.103.239/predict`
- **Health Check**: `GET http://34.173.103.239/healthz`
- **API Docs**: `GET http://34.173.103.239/docs`
- **Info**: `GET http://34.173.103.239/info`

### **GCP Deployment Details**
- **Platform**: Google Kubernetes Engine (GKE)
- **Region**: us-central1-a
- **Cluster**: mnist-cluster
- **Node Type**: e2-medium (2 vCPU, 4GB RAM)
- **Disk Size**: 50GB per node
- **Replicas**: 2 inference pods
- **Model Accuracy**: 99.12%

### **Try It Now**
1. Visit http://34.173.103.239
2. Upload a handwritten digit image (0-9)
3. Click "Predict Digit"
4. View real-time prediction with confidence scores

### **Test with cURL**
```bash
# Health check
curl http://34.173.103.239/healthz

# Get service info
curl http://34.173.103.239/info

# Make a prediction (replace with your image)
curl -X POST http://34.173.103.239/predict -F "file=@digit.png"
```

---

## 🛡️ Self-Healing Mechanisms

This deployment implements **7 comprehensive self-healing mechanisms**:

### **1. Multiple Replicas**
```yaml
replicas: 2
```
- If one pod crashes, the other continues serving
- Kubernetes automatically creates replacements
- Zero downtime during failures

### **2. Liveness Probe**
```yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: 8000
  periodSeconds: 10
  failureThreshold: 3
```
- Detects deadlocked/unresponsive containers
- Automatically restarts after 3 failures (30s)
- Checks `/healthz` endpoint every 10 seconds

### **3. Readiness Probe**
```yaml
readinessProbe:
  httpGet:
    path: /healthz
    port: 8000
  periodSeconds: 5
  failureThreshold: 2
```
- Removes unready pods from service
- Traffic only to ready pods
- Allows pods to recover without serving errors

### **4. Startup Probe**
```yaml
startupProbe:
  httpGet:
    path: /healthz
    port: 8000
  failureThreshold: 12  # 60 seconds
```
- Gives model loading time (10-30 seconds)
- Prevents premature restarts during initialization

### **5. Restart Policy**
```yaml
restartPolicy: Always
```
- Crashed containers automatically restart
- Exponential backoff prevents restart loops

### **6. Resource Limits**
```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```
- Prevents resource exhaustion
- Ensures fair resource distribution
- Pod killed if limits exceeded, then restarted

### **7. Rolling Updates**
```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxUnavailable: 1
    maxSurge: 1
```
- Zero-downtime deployments
- Gradual rollout of new versions
- Automatic rollback on failure

### **Self-Healing Summary**

| Mechanism | Purpose | Recovery Time |
|-----------|---------|---------------|
| Replicas | High availability | Instant |
| Liveness Probe | Deadlock detection | 30-60s |
| Readiness Probe | Traffic management | 10s |
| Startup Probe | Slow startup protection | 60s |
| Restart Policy | Crash recovery | 1-10s |
| Resource Limits | Resource protection | 1-10s |
| Rolling Updates | Zero downtime | Minutes |

---

## 💡 Usage Examples

### **Web UI**

1. Access: http://localhost/
2. Upload handwritten digit image (0-9)
3. Click "Predict Digit"
4. View prediction with confidence scores

### **API**

**Health Check:**
```bash
curl http://localhost/healthz
```

**Prediction:**
```bash
curl -X POST http://localhost/predict -F "file=@digit.png"
```

**API Docs:**
- Swagger UI: http://localhost/docs
- ReDoc: http://localhost/redoc

### **Monitoring**

```bash
# Check pods
kubectl get pods -l app=mnist-inference

# View logs
kubectl logs -l app=mnist-inference --tail=50

# Check service
kubectl get service mnist-inference-service
```

---

## 🔧 Troubleshooting

### **Pods in ImagePullBackOff**

**Solution:**
```bash
# Build images
docker build -t mnist-infer:latest -f inference/Dockerfile inference/

# Restart deployment
kubectl rollout restart deployment mnist-inference
```

### **Pods in CrashLoopBackOff**

**Solution:**
```bash
# Check logs
kubectl logs <pod-name>

# Common cause: Model not found (training not complete)
kubectl get job mnist-train-job
kubectl logs job/mnist-train-job
```

### **Service Not Accessible**

**Solution:**
```bash
# Check pods are ready
kubectl get pods -l app=mnist-inference
# READY should be 1/1

# Check service endpoints
kubectl get endpoints mnist-inference-service
# Should show pod IPs

# Check logs
kubectl logs -l app=mnist-inference
```

### **Training Job Fails**

**Solution:**
```bash
# Check logs
kubectl logs job/mnist-train-job

# Re-run training
kubectl delete job mnist-train-job
kubectl apply -f k8s/local/job-train.yaml
```

---

## 📁 Project Structure

```
mnist-cloud-k8s-pipeline/
├── training/                 # Training component
│   ├── train.py              # Training script
│   ├── model_def.py          # CNN model
│   ├── requirements.txt      # Dependencies
│   └── Dockerfile            # Training container
│
├── inference/                # Inference component
│   ├── app/
│   │   ├── main.py           # FastAPI app
│   │   ├── model_def.py      # Model loading
│   │   ├── config.py         # Configuration
│   │   ├── schemas.py        # Pydantic models
│   │   └── templates/
│   │       └── index.html    # Web UI
│   ├── requirements.txt      # Dependencies
│   └── Dockerfile            # Inference container
│
├── k8s/                      # Kubernetes manifests
│   ├── local/                # Local deployment
│   │   ├── pvc-model.yaml    # Persistent storage
│   │   └── job-train.yaml    # Training job
│   ├── gke/                  # GKE deployment
│   └── inference/            # Inference service
│       ├── deployment-infer.yaml  # Deployment
│       └── service-infer.yaml     # LoadBalancer
│
└── tools/                    # Automation scripts
    ├── build_images.py       # Build Docker images
    ├── deploy_k8s.py         # Deploy to Kubernetes
    └── run_local.py          # Run locally
```

---

## 🛠️ Technologies Used

- **Machine Learning**: PyTorch, torchvision
- **Web Framework**: FastAPI, Uvicorn
- **Containerization**: Docker
- **Orchestration**: Kubernetes
- **Storage**: Persistent Volume Claims (PVC)
- **Language**: Python 3.10
- **Image Processing**: Pillow, NumPy

---

## 📝 License

This project is created for educational purposes as part of a cloud computing course assignment.

---

## 👤 Author

**Dhairya Mishra**
- GitHub: [@dhairyamishra](https://github.com/dhairyamishra)
- Project: [mnist-cloud-k8s-pipeline-nyu25](https://github.com/dhairyamishra/mnist-cloud-k8s-pipeline-nyu25)

---

## 🎓 Acknowledgments

- **NYU Course**: Cloud Computing and Big Data
- **MNIST Dataset**: Yann LeCun et al.
- **Technologies**: PyTorch, FastAPI, Kubernetes communities

---

**⭐ If you found this project helpful, please star the repository!**
