#!/usr/bin/env python3
"""
GCP Deployment Automation Script for MNIST Pipeline

This script automates the entire deployment process to Google Cloud Platform:
1. Verifies GCP project and authentication
2. Builds Docker images
3. Pushes images to Google Container Registry (GCR)
4. Updates Kubernetes manifests with project ID
5. Deploys to GKE cluster

Usage:
    python tools/deploy_gcp.py --project-id YOUR_PROJECT_ID
    python tools/deploy_gcp.py --project-id YOUR_PROJECT_ID --skip-build
    python tools/deploy_gcp.py --project-id YOUR_PROJECT_ID --deploy-only
"""

import argparse
import subprocess
import sys
import time
import re
from pathlib import Path
from typing import List, Optional, Tuple


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class GCPDeployer:
    """Handles GCP deployment operations."""
    
    def __init__(self, project_id: str, skip_build: bool = False, deploy_only: bool = False):
        """
        Initialize the GCP deployer.
        
        Args:
            project_id: GCP project ID
            skip_build: Skip Docker image building
            deploy_only: Only deploy to Kubernetes (skip build and push)
        """
        self.project_id = project_id
        self.skip_build = skip_build
        self.deploy_only = deploy_only
        self.project_root = Path(__file__).parent.parent
        self.train_image = f"gcr.io/{project_id}/mnist-train:latest"
        self.infer_image = f"gcr.io/{project_id}/mnist-infer:latest"
        
    def print_header(self, message: str):
        """Print a formatted header."""
        print(f"\n{Colors.HEADER}{'='*70}")
        print(f"  {message}")
        print(f"{'='*70}{Colors.ENDC}\n")
    
    def print_success(self, message: str):
        """Print a success message."""
        print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")
    
    def print_error(self, message: str):
        """Print an error message."""
        print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")
    
    def print_warning(self, message: str):
        """Print a warning message."""
        print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")
    
    def print_info(self, message: str):
        """Print an info message."""
        print(f"{Colors.OKCYAN}ℹ {message}{Colors.ENDC}")
    
    def run_command(self, cmd: List[str], check: bool = True, capture: bool = True) -> Tuple[bool, str]:
        """
        Run a shell command.
        
        Args:
            cmd: Command to run as list
            check: Whether to raise exception on failure
            capture: Whether to capture output
            
        Returns:
            Tuple of (success, output)
        """
        try:
            if capture:
                result = subprocess.run(
                    cmd,
                    check=check,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
                return True, result.stdout.strip()
            else:
                result = subprocess.run(cmd, check=check)
                return True, ""
        except subprocess.CalledProcessError as e:
            if capture:
                return False, e.stderr
            return False, ""
        except Exception as e:
            return False, str(e)
    
    def verify_gcloud_installed(self) -> bool:
        """Verify gcloud CLI is installed."""
        self.print_info("Checking if gcloud is installed...")
        success, output = self.run_command(["gcloud", "--version"])
        if success:
            self.print_success("gcloud CLI is installed")
            return True
        else:
            self.print_error("gcloud CLI is not installed")
            self.print_info("Install from: https://cloud.google.com/sdk/docs/install")
            return False
    
    def verify_kubectl_installed(self) -> bool:
        """Verify kubectl is installed."""
        self.print_info("Checking if kubectl is installed...")
        success, output = self.run_command(["kubectl", "version", "--client"])
        if success:
            self.print_success("kubectl is installed")
            return True
        else:
            self.print_error("kubectl is not installed")
            self.print_info("Install with: gcloud components install kubectl")
            return False
    
    def verify_docker_installed(self) -> bool:
        """Verify Docker is installed."""
        self.print_info("Checking if Docker is installed...")
        success, output = self.run_command(["docker", "--version"])
        if success:
            self.print_success("Docker is installed")
            return True
        else:
            self.print_error("Docker is not installed")
            self.print_info("Install from: https://www.docker.com/products/docker-desktop")
            return False
    
    def verify_gcp_auth(self) -> bool:
        """Verify GCP authentication."""
        self.print_info("Checking GCP authentication...")
        success, output = self.run_command(["gcloud", "auth", "list"])
        if success and "ACTIVE" in output:
            self.print_success("GCP authentication is active")
            return True
        else:
            self.print_error("Not authenticated with GCP")
            self.print_info("Run: gcloud auth login")
            return False
    
    def verify_project_exists(self) -> bool:
        """Verify GCP project exists and is accessible."""
        self.print_info(f"Checking if project '{self.project_id}' exists...")
        success, output = self.run_command(["gcloud", "projects", "describe", self.project_id])
        if success:
            self.print_success(f"Project '{self.project_id}' is accessible")
            return True
        else:
            self.print_error(f"Project '{self.project_id}' not found or not accessible")
            self.print_info("Verify project ID and permissions")
            return False
    
    def set_gcp_project(self) -> bool:
        """Set the default GCP project."""
        self.print_info(f"Setting default project to '{self.project_id}'...")
        success, output = self.run_command(["gcloud", "config", "set", "project", self.project_id])
        if success:
            self.print_success(f"Default project set to '{self.project_id}'")
            return True
        else:
            self.print_error("Failed to set default project")
            return False
    
    def verify_cluster_exists(self) -> bool:
        """Verify GKE cluster exists."""
        self.print_info("Checking if GKE cluster exists...")
        success, output = self.run_command(["gcloud", "container", "clusters", "list", "--format=value(name)"])
        if success and "mnist-cluster" in output:
            self.print_success("GKE cluster 'mnist-cluster' found")
            return True
        else:
            self.print_warning("GKE cluster 'mnist-cluster' not found")
            self.print_info("Create cluster with: cd terraform && terraform apply")
            return False
    
    def configure_kubectl(self) -> bool:
        """Configure kubectl to use the GKE cluster."""
        self.print_info("Configuring kubectl for GKE cluster...")
        success, output = self.run_command([
            "gcloud", "container", "clusters", "get-credentials",
            "mnist-cluster", "--zone=us-central1-a", f"--project={self.project_id}"
        ])
        if success:
            self.print_success("kubectl configured for GKE cluster")
            return True
        else:
            self.print_error("Failed to configure kubectl")
            return False
    
    def configure_docker_gcr(self) -> bool:
        """Configure Docker to authenticate with GCR."""
        self.print_info("Configuring Docker for GCR...")
        success, output = self.run_command(["gcloud", "auth", "configure-docker", "--quiet"])
        if success:
            self.print_success("Docker configured for GCR")
            return True
        else:
            self.print_error("Failed to configure Docker for GCR")
            return False
    
    def build_docker_image(self, dockerfile_dir: str, image_name: str, image_tag: str) -> bool:
        """
        Build a Docker image.
        
        Args:
            dockerfile_dir: Directory containing Dockerfile
            image_name: Name of the image
            image_tag: Full image tag (e.g., gcr.io/project/image:latest)
            
        Returns:
            True if successful, False otherwise
        """
        self.print_info(f"Building {image_name} image...")
        print(f"  This may take 5-10 minutes...")
        
        dockerfile_path = self.project_root / dockerfile_dir
        
        success, output = self.run_command([
            "docker", "build",
            "-t", image_tag,
            "-f", str(dockerfile_path / "Dockerfile"),
            str(dockerfile_path)
        ], capture=False)
        
        if success:
            self.print_success(f"{image_name} image built successfully")
            return True
        else:
            self.print_error(f"Failed to build {image_name} image")
            return False
    
    def push_docker_image(self, image_tag: str, image_name: str) -> bool:
        """
        Push a Docker image to GCR.
        
        Args:
            image_tag: Full image tag
            image_name: Name of the image (for display)
            
        Returns:
            True if successful, False otherwise
        """
        self.print_info(f"Pushing {image_name} to GCR...")
        print(f"  This may take 2-5 minutes...")
        
        success, output = self.run_command(["docker", "push", image_tag], capture=False)
        
        if success:
            self.print_success(f"{image_name} pushed to GCR")
            return True
        else:
            self.print_error(f"Failed to push {image_name} to GCR")
            return False
    
    def verify_image_in_gcr(self, image_name: str) -> bool:
        """Verify image exists in GCR."""
        self.print_info(f"Verifying {image_name} in GCR...")
        success, output = self.run_command([
            "gcloud", "container", "images", "list",
            f"--repository=gcr.io/{self.project_id}",
            "--format=value(name)"
        ])
        if success and image_name in output:
            self.print_success(f"{image_name} found in GCR")
            return True
        else:
            self.print_warning(f"{image_name} not found in GCR")
            return False
    
    def update_manifest_image(self, manifest_path: Path, new_image: str) -> bool:
        """
        Update image reference in a Kubernetes manifest.
        
        Args:
            manifest_path: Path to manifest file
            new_image: New image reference
            
        Returns:
            True if successful, False otherwise
        """
        try:
            content = manifest_path.read_text()
            
            # Replace image reference (matches gcr.io/*/image:tag pattern)
            pattern = r'image:\s+gcr\.io/[^/]+/[^:\s]+:[^\s]+'
            replacement = f'image: {new_image}'
            
            updated_content = re.sub(pattern, replacement, content)
            
            if updated_content != content:
                manifest_path.write_text(updated_content)
                return True
            return False
        except Exception as e:
            self.print_error(f"Failed to update manifest: {e}")
            return False
    
    def update_manifests(self) -> bool:
        """Update all Kubernetes manifests with correct project ID."""
        self.print_info("Updating Kubernetes manifests with project ID...")
        
        # Update training job manifest
        train_manifest = self.project_root / "k8s" / "gke" / "job-train.yaml"
        if train_manifest.exists():
            if self.update_manifest_image(train_manifest, self.train_image):
                self.print_success(f"Updated {train_manifest.name}")
            else:
                self.print_warning(f"No changes needed for {train_manifest.name}")
        
        # Update inference deployment manifest
        infer_manifest = self.project_root / "k8s" / "inference" / "deployment-infer.yaml"
        if infer_manifest.exists():
            if self.update_manifest_image(infer_manifest, self.infer_image):
                self.print_success(f"Updated {infer_manifest.name}")
            else:
                self.print_warning(f"No changes needed for {infer_manifest.name}")
        
        return True
    
    def deploy_to_kubernetes(self) -> bool:
        """Deploy to Kubernetes using the deploy_k8s.py script."""
        self.print_info("Deploying to Kubernetes...")
        
        deploy_script = self.project_root / "tools" / "deploy_k8s.py"
        
        success, output = self.run_command([
            sys.executable, str(deploy_script), "--env", "gke"
        ], capture=False)
        
        return success
    
    def run_deployment(self) -> bool:
        """Run the complete deployment process."""
        
        # Phase 1: Verification
        self.print_header("PHASE 1: Verification")
        
        if not self.verify_gcloud_installed():
            return False
        
        if not self.verify_kubectl_installed():
            return False
        
        if not self.verify_docker_installed():
            return False
        
        if not self.verify_gcp_auth():
            return False
        
        if not self.verify_project_exists():
            return False
        
        if not self.set_gcp_project():
            return False
        
        if not self.verify_cluster_exists():
            self.print_warning("Please create the GKE cluster first:")
            self.print_info("  cd terraform")
            self.print_info("  terraform init")
            self.print_info("  terraform apply")
            return False
        
        if not self.configure_kubectl():
            return False
        
        self.print_success("All verifications passed!")
        
        # Phase 2: Docker Images (skip if deploy_only)
        if not self.deploy_only:
            self.print_header("PHASE 2: Docker Images")
            
            if not self.configure_docker_gcr():
                return False
            
            # Build images (skip if skip_build)
            if not self.skip_build:
                if not self.build_docker_image("training", "mnist-train", self.train_image):
                    return False
                
                if not self.build_docker_image("inference", "mnist-infer", self.infer_image):
                    return False
            else:
                self.print_warning("Skipping image build (--skip-build flag)")
            
            # Push images
            if not self.push_docker_image(self.train_image, "mnist-train"):
                return False
            
            if not self.push_docker_image(self.infer_image, "mnist-infer"):
                return False
            
            # Verify images
            self.verify_image_in_gcr("mnist-train")
            self.verify_image_in_gcr("mnist-infer")
            
            self.print_success("Docker images ready!")
        else:
            self.print_warning("Skipping Docker build and push (--deploy-only flag)")
        
        # Phase 3: Update Manifests
        self.print_header("PHASE 3: Update Manifests")
        
        if not self.update_manifests():
            return False
        
        self.print_success("Manifests updated!")
        
        # Phase 4: Kubernetes Deployment
        self.print_header("PHASE 4: Kubernetes Deployment")
        
        if not self.deploy_to_kubernetes():
            self.print_error("Kubernetes deployment failed")
            return False
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Automated GCP deployment for MNIST Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full deployment (build, push, deploy)
  python tools/deploy_gcp.py --project-id mnist-k8s-pipeline-12345
  
  # Skip building (use existing local images)
  python tools/deploy_gcp.py --project-id mnist-k8s-pipeline-12345 --skip-build
  
  # Deploy only (skip build and push)
  python tools/deploy_gcp.py --project-id mnist-k8s-pipeline-12345 --deploy-only

Prerequisites:
  1. GCP project created with billing enabled
  2. GKE cluster created (cd terraform && terraform apply)
  3. gcloud, kubectl, docker installed
  4. Authenticated with GCP (gcloud auth login)
        """
    )
    
    parser.add_argument(
        "--project-id",
        type=str,
        required=True,
        help="GCP Project ID (e.g., mnist-k8s-pipeline-12345)"
    )
    
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip Docker image building (use existing local images)"
    )
    
    parser.add_argument(
        "--deploy-only",
        action="store_true",
        help="Only deploy to Kubernetes (skip build and push)"
    )
    
    args = parser.parse_args()
    
    # Validate project ID format
    if not re.match(r'^[a-z][a-z0-9-]{4,28}[a-z0-9]$', args.project_id):
        print(f"{Colors.FAIL}✗ Invalid project ID format{Colors.ENDC}")
        print(f"{Colors.WARNING}Project ID must:")
        print("  - Start with a lowercase letter")
        print("  - Contain only lowercase letters, numbers, and hyphens")
        print("  - Be 6-30 characters long")
        print(f"  - End with a letter or number{Colors.ENDC}")
        sys.exit(1)
    
    # Initialize deployer
    deployer = GCPDeployer(
        project_id=args.project_id,
        skip_build=args.skip_build,
        deploy_only=args.deploy_only
    )
    
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║         MNIST GCP Deployment Automation                          ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")
    print(f"Project ID: {Colors.BOLD}{args.project_id}{Colors.ENDC}")
    print(f"Skip Build: {args.skip_build}")
    print(f"Deploy Only: {args.deploy_only}")
    
    # Run deployment
    success = deployer.run_deployment()
    
    # Summary
    print(f"\n{Colors.BOLD}{'='*70}")
    if success:
        print(f"{Colors.OKGREEN}✓ DEPLOYMENT COMPLETED SUCCESSFULLY!{Colors.ENDC}")
        print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
        print("  1. Get external IP: kubectl get service mnist-inference-service")
        print("  2. Access web UI: http://<EXTERNAL-IP>/")
        print("  3. Test predictions with digit images")
        print(f"\n{Colors.BOLD}Monitoring:{Colors.ENDC}")
        print("  kubectl get pods")
        print("  kubectl logs -f job/mnist-train-job")
        print("  kubectl logs -l app=mnist-inference")
        print(f"\n{Colors.WARNING}⚠ REMEMBER TO CLEANUP:{Colors.ENDC}")
        print("  cd terraform && terraform destroy")
    else:
        print(f"{Colors.FAIL}✗ DEPLOYMENT FAILED{Colors.ENDC}")
        print(f"\n{Colors.BOLD}Troubleshooting:{Colors.ENDC}")
        print("  1. Check error messages above")
        print("  2. Verify GCP authentication: gcloud auth list")
        print("  3. Verify project: gcloud config get-value project")
        print("  4. Check cluster: gcloud container clusters list")
        print("  5. Review GCP_DEPLOYMENT_GUIDE.md for detailed help")
    print(f"{'='*70}\n")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
