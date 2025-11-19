#!/usr/bin/env python3
"""
Kubernetes Deployment Automation Script

This script automates the deployment of the MNIST training and inference
services to Kubernetes in the correct order with health checks.

Usage:
    python tools/deploy_k8s.py --env local          # Deploy to local Kubernetes
    python tools/deploy_k8s.py --env gke            # Deploy to GKE
    python tools/deploy_k8s.py --training-only      # Deploy training job only
    python tools/deploy_k8s.py --inference-only     # Deploy inference service only
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


class KubernetesDeployer:
    """Handles Kubernetes deployment operations."""
    
    def __init__(self, env: str = "local"):
        """
        Initialize the deployer.
        
        Args:
            env: Environment to deploy to ('local' or 'gke')
        """
        self.env = env
        self.project_root = Path(__file__).parent.parent
        self.k8s_dir = self.project_root / "k8s"
        
    def run_kubectl(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """
        Run a kubectl command.
        
        Args:
            args: kubectl command arguments
            check: Whether to raise exception on failure
            
        Returns:
            CompletedProcess object
        """
        cmd = ["kubectl"] + args
        return subprocess.run(
            cmd,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
    
    def apply_manifest(self, manifest_path: str) -> bool:
        """
        Apply a Kubernetes manifest.
        
        Args:
            manifest_path: Path to manifest file (relative to project root)
            
        Returns:
            True if successful, False otherwise
        """
        full_path = self.project_root / manifest_path
        
        if not full_path.exists():
            print(f"✗ Manifest not found: {manifest_path}")
            return False
        
        print(f"\n{'='*70}")
        print(f"Applying manifest: {manifest_path}")
        print(f"{'='*70}")
        
        try:
            result = self.run_kubectl(["apply", "-f", str(full_path)])
            print(result.stdout)
            print(f"✓ Successfully applied {manifest_path}\n")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to apply {manifest_path}")
            print(f"Error: {e.stderr}")
            return False
    
    def wait_for_pvc(self, pvc_name: str, timeout: int = 60) -> bool:
        """
        Wait for a PVC to be bound.
        
        Args:
            pvc_name: Name of the PVC
            timeout: Timeout in seconds
            
        Returns:
            True if PVC is bound, False otherwise
        """
        print(f"\nWaiting for PVC '{pvc_name}' to be bound (timeout: {timeout}s)...")
        
        try:
            result = self.run_kubectl([
                "wait", "--for=condition=Bound",
                f"pvc/{pvc_name}",
                f"--timeout={timeout}s"
            ])
            print(f"✓ PVC '{pvc_name}' is bound\n")
            return True
        except subprocess.CalledProcessError:
            print(f"✗ PVC '{pvc_name}' failed to bind within {timeout}s\n")
            return False
    
    def wait_for_job(self, job_name: str, timeout: int = 900) -> bool:
        """
        Wait for a Job to complete.
        
        Args:
            job_name: Name of the Job
            timeout: Timeout in seconds (default 15 minutes for training)
            
        Returns:
            True if Job completed, False otherwise
        """
        print(f"\nWaiting for Job '{job_name}' to complete (timeout: {timeout}s)...")
        print("This may take several minutes for training to complete...")
        
        try:
            result = self.run_kubectl([
                "wait", "--for=condition=complete",
                f"job/{job_name}",
                f"--timeout={timeout}s"
            ])
            print(f"✓ Job '{job_name}' completed successfully\n")
            return True
        except subprocess.CalledProcessError:
            print(f"✗ Job '{job_name}' failed to complete within {timeout}s")
            print("Check job logs: kubectl logs job/{job_name}\n")
            return False
    
    def wait_for_deployment(self, deployment_name: str, timeout: int = 300) -> bool:
        """
        Wait for a Deployment to be available.
        
        Args:
            deployment_name: Name of the Deployment
            timeout: Timeout in seconds
            
        Returns:
            True if Deployment is available, False otherwise
        """
        print(f"\nWaiting for Deployment '{deployment_name}' to be available (timeout: {timeout}s)...")
        
        try:
            result = self.run_kubectl([
                "wait", "--for=condition=available",
                f"deployment/{deployment_name}",
                f"--timeout={timeout}s"
            ])
            print(f"✓ Deployment '{deployment_name}' is available\n")
            return True
        except subprocess.CalledProcessError:
            print(f"✗ Deployment '{deployment_name}' failed to become available within {timeout}s")
            print(f"Check pods: kubectl get pods -l app={deployment_name}\n")
            return False
    
    def get_service_info(self, service_name: str):
        """
        Get and display service information.
        
        Args:
            service_name: Name of the Service
        """
        print(f"\n{'='*70}")
        print(f"Service Information: {service_name}")
        print(f"{'='*70}\n")
        
        try:
            result = self.run_kubectl(["get", "service", service_name])
            print(result.stdout)
            
            # Get external IP
            result = self.run_kubectl([
                "get", "service", service_name,
                "-o", "jsonpath={.status.loadBalancer.ingress[0].ip}"
            ], check=False)
            
            external_ip = result.stdout.strip()
            if external_ip:
                print(f"\n✓ Service is accessible at: http://{external_ip}/")
            else:
                print("\n⚠ External IP not yet assigned (may take 1-2 minutes on GKE)")
                print("  On Docker Desktop: http://localhost/")
                print(f"  Check status: kubectl get service {service_name} --watch")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to get service info: {e.stderr}")
    
    def deploy_storage(self) -> bool:
        """Deploy persistent storage (PVC)."""
        print(f"\n{'#'*70}")
        print("# STEP 1: Deploying Persistent Storage")
        print(f"{'#'*70}\n")
        
        manifest = f"k8s/{self.env}/pvc-model.yaml"
        if not self.apply_manifest(manifest):
            return False
        
        return self.wait_for_pvc("mnist-model-pvc")
    
    def deploy_training(self) -> bool:
        """Deploy training job."""
        print(f"\n{'#'*70}")
        print("# STEP 2: Deploying Training Job")
        print(f"{'#'*70}\n")
        
        manifest = f"k8s/{self.env}/job-train.yaml"
        if not self.apply_manifest(manifest):
            return False
        
        # Ask user if they want to wait for training
        print("\nTraining job submitted. This will take 10-15 minutes to complete.")
        response = input("Wait for training to complete? (y/n): ").strip().lower()
        
        if response == 'y':
            return self.wait_for_job("mnist-train-job", timeout=900)
        else:
            print("\n⚠ Skipping wait. Check status with: kubectl get jobs")
            print("  Training must complete before deploying inference service!")
            return True
    
    def deploy_inference(self) -> bool:
        """Deploy inference service."""
        print(f"\n{'#'*70}")
        print("# STEP 3: Deploying Inference Service")
        print(f"{'#'*70}\n")
        
        # Deploy deployment
        manifest = f"k8s/inference/deployment-infer.yaml"
        if not self.apply_manifest(manifest):
            return False
        
        if not self.wait_for_deployment("mnist-inference"):
            return False
        
        # Deploy service
        manifest = f"k8s/inference/service-infer.yaml"
        if not self.apply_manifest(manifest):
            return False
        
        # Get service info
        self.get_service_info("mnist-inference-service")
        
        return True


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Deploy MNIST services to Kubernetes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full deployment to local Kubernetes
  python tools/deploy_k8s.py --env local
  
  # Full deployment to GKE
  python tools/deploy_k8s.py --env gke
  
  # Deploy only training job
  python tools/deploy_k8s.py --training-only
  
  # Deploy only inference service
  python tools/deploy_k8s.py --inference-only
        """
    )
    
    parser.add_argument(
        "--env",
        type=str,
        choices=["local", "gke"],
        default="local",
        help="Environment to deploy to (default: local)"
    )
    
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Deploy only the training job"
    )
    
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Deploy only the inference service"
    )
    
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = KubernetesDeployer(env=args.env)
    
    print(f"\n{'='*70}")
    print(f"MNIST Kubernetes Deployment - Environment: {args.env.upper()}")
    print(f"{'='*70}\n")
    
    success = True
    
    # Deploy based on flags
    if args.inference_only:
        # Inference only (assumes storage and training already done)
        success = deployer.deploy_inference()
    elif args.training_only:
        # Training only
        success = deployer.deploy_storage() and deployer.deploy_training()
    else:
        # Full deployment
        success = (
            deployer.deploy_storage() and
            deployer.deploy_training() and
            deployer.deploy_inference()
        )
    
    # Summary
    print(f"\n{'='*70}")
    if success:
        print("✓ Deployment completed successfully!")
        print("\nNext steps:")
        print("  1. Access the web UI (see service info above)")
        print("  2. Upload a digit image and test predictions")
        print("  3. Monitor with: kubectl get pods,svc")
    else:
        print("✗ Deployment failed. Check the output above for details.")
        print("\nTroubleshooting:")
        print("  kubectl get pods")
        print("  kubectl describe pod <pod-name>")
        print("  kubectl logs <pod-name>")
    print(f"{'='*70}\n")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
