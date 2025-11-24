# Terraform Configuration for MNIST GKE Cluster

This directory contains Terraform configuration to deploy a Google Kubernetes Engine (GKE) cluster for the MNIST ML pipeline.

---

## 📋 Prerequisites

1. **Terraform installed** (v1.0+)
   ```powershell
   # Check if installed
   terraform version
   
   # If not installed, download from: https://www.terraform.io/downloads
   ```

2. **Google Cloud SDK authenticated**
   ```powershell
   gcloud auth application-default login
   ```

3. **GCP Project with billing enabled**
   - Project ID: `mnist-k8s-pipeline`
   - Billing account linked
   - Required APIs enabled (Terraform will enable them)

---

## 🚀 Quick Start

### **Step 1: Initialize Terraform**

```powershell
cd terraform
terraform init
```

This downloads the Google Cloud provider plugin.

---

### **Step 2: Review the Plan**

```powershell
terraform plan
```

This shows what resources will be created:
- GKE cluster (`mnist-cluster`)
- Node pool (2 x e2-medium nodes)
- Network configuration

---

### **Step 3: Create the Cluster**

```powershell
terraform apply
```

Type `yes` when prompted. This takes **5-10 minutes**.

---

### **Step 4: Configure kubectl**

After creation, run the output command:

```powershell
# Get the command from Terraform output
terraform output kubectl_config_command

# Run it (example)
gcloud container clusters get-credentials mnist-cluster --zone=us-central1-a --project=mnist-k8s-pipeline
```

---

### **Step 5: Verify Cluster**

```powershell
kubectl get nodes
kubectl cluster-info
```

---

## 🎛️ Configuration

Edit `variables.tf` or override via command line:

```powershell
# Use different machine type
terraform apply -var="machine_type=e2-small"

# Use different node count
terraform apply -var="node_count=3"

# Use different zone
terraform apply -var="zone=us-west1-a"
```

---

## 📊 Resources Created

| Resource | Type | Configuration |
|----------|------|---------------|
| **Cluster** | GKE Cluster | Regional, no default node pool |
| **Node Pool** | Managed Node Pool | 2 nodes, e2-medium, 20GB disk |
| **Autoscaling** | Enabled | Min: 1, Max: 3 nodes |
| **Auto-repair** | Enabled | Automatic node repair |
| **Auto-upgrade** | Enabled | Automatic K8s version upgrades |

---

## 💰 Cost Estimate

- **e2-medium**: ~$0.03/hour per node
- **2 nodes**: ~$0.06/hour
- **Storage**: ~$0.10/month for 40GB
- **Total**: ~$1.50/day if left running

---

## 🗑️ Cleanup

**Important**: Delete the cluster when done to avoid charges!

```powershell
terraform destroy
```

Type `yes` when prompted. This takes **2-3 minutes**.

---

## 📝 Files

- `main.tf` - Main Terraform configuration (cluster, node pool, outputs)
- `variables.tf` - Variable definitions with defaults
- `README.md` - This file

---

## 🔧 Troubleshooting

### **Error: APIs not enabled**

```powershell
gcloud services enable container.googleapis.com compute.googleapis.com
```

### **Error: Permission denied**

Ensure you have Owner or Editor role:
```powershell
gcloud projects get-iam-policy mnist-k8s-pipeline
```

### **Error: Quota exceeded**

Check quotas in GCP Console:
https://console.cloud.google.com/iam-admin/quotas

---

## 🎯 Next Steps

After cluster creation:

1. **Push images to GCR**
   ```powershell
   docker tag mnist-train:latest gcr.io/mnist-k8s-pipeline/mnist-train:latest
   docker tag mnist-infer:latest gcr.io/mnist-k8s-pipeline/mnist-infer:latest
   docker push gcr.io/mnist-k8s-pipeline/mnist-train:latest
   docker push gcr.io/mnist-k8s-pipeline/mnist-infer:latest
   ```

2. **Deploy to GKE**
   ```powershell
   cd ..
   python tools/deploy_k8s.py --env gke
   ```

3. **Access the service**
   ```powershell
   kubectl get service mnist-inference-service
   ```

---

## 📚 Learn More

- [Terraform GCP Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [GKE Documentation](https://cloud.google.com/kubernetes-engine/docs)
- [Terraform Best Practices](https://www.terraform.io/docs/cloud/guides/recommended-practices/index.html)
