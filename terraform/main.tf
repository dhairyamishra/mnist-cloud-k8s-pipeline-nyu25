# Terraform configuration for MNIST GKE Cluster
# This creates a GKE cluster with 2 nodes, no GPU, for ML inference

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# Configure the Google Cloud Provider
provider "google" {
  project = var.project_id
  region  = var.region
}

# GKE Cluster
resource "google_container_cluster" "mnist_cluster" {
  name     = var.cluster_name
  location = var.zone

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1

  # Disable logging and monitoring to reduce costs
  logging_service    = "none"
  monitoring_service = "none"

  # Network configuration
  network    = "default"
  subnetwork = "default"
}

# Separately Managed Node Pool
resource "google_container_node_pool" "mnist_nodes" {
  name       = "${var.cluster_name}-node-pool"
  location   = var.zone
  cluster    = google_container_cluster.mnist_cluster.name
  node_count = var.node_count

  node_config {
    machine_type = var.machine_type
    disk_size_gb = var.disk_size_gb

    # OAuth scopes for the nodes
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    # Metadata
    metadata = {
      disable-legacy-endpoints = "true"
    }

    # Labels
    labels = {
      app     = "mnist-pipeline"
      env     = "production"
      managed = "terraform"
    }

    # No GPU (CPU only)
    # guest_accelerator {
    #   type  = "nvidia-tesla-t4"
    #   count = 0
    # }
  }

  # Autoscaling configuration
  autoscaling {
    min_node_count = 1
    max_node_count = 3
  }

  # Management configuration
  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# Output the cluster endpoint
output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.mnist_cluster.endpoint
  sensitive   = true
}

# Output the cluster name
output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.mnist_cluster.name
}

# Output kubectl config command
output "kubectl_config_command" {
  description = "Command to configure kubectl"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.mnist_cluster.name} --zone=${var.zone} --project=${var.project_id}"
}
