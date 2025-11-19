#!/usr/bin/env python3
"""
Local Docker Run Script

This script runs the MNIST training and inference containers locally
without Kubernetes, useful for quick testing and development.

Usage:
    python tools/run_local.py --train          # Run training locally
    python tools/run_local.py --infer          # Run inference locally
    python tools/run_local.py --infer --port 8080  # Custom port
"""
import argparse
import subprocess
import sys
from pathlib import Path


class LocalRunner:
    """Handles local Docker container execution."""
    
    def __init__(self):
        """Initialize the local runner."""
        self.project_root = Path(__file__).parent.parent
        self.local_model_dir = self.project_root / "local_model"
        
        # Ensure local_model directory exists
        self.local_model_dir.mkdir(exist_ok=True)
    
    def run_docker(self, args: list) -> subprocess.CompletedProcess:
        """
        Run a docker command.
        
        Args:
            args: docker command arguments
            
        Returns:
            CompletedProcess object
        """
        cmd = ["docker"] + args
        return subprocess.run(cmd, check=True)
    
    def run_training(self):
        """Run the training container locally."""
        print(f"\n{'='*70}")
        print("Running MNIST Training Container Locally")
        print(f"{'='*70}\n")
        
        print("Configuration:")
        print(f"  Image: mnist-train:latest")
        print(f"  Model output: {self.local_model_dir}")
        print(f"  Epochs: 5 (default)")
        print()
        
        # Docker run command
        cmd = [
            "run",
            "--rm",  # Remove container after exit
            "-v", f"{self.local_model_dir.absolute()}:/mnt/model",  # Mount model directory
            "-e", "MODEL_DIR=/mnt/model",  # Set environment variable
            "-e", "EPOCHS=5",  # Training epochs
            "mnist-train:latest"
        ]
        
        print("Starting training...")
        print("This will take 10-15 minutes to complete.\n")
        
        try:
            self.run_docker(cmd)
            print(f"\n{'='*70}")
            print("✓ Training completed successfully!")
            print(f"✓ Model saved to: {self.local_model_dir}/mnist_cnn.pt")
            print(f"✓ Metadata saved to: {self.local_model_dir}/metadata.json")
            print(f"{'='*70}\n")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Training failed with exit code {e.returncode}")
            sys.exit(1)
    
    def run_inference(self, port: int = 8000):
        """
        Run the inference container locally.
        
        Args:
            port: Port to expose the service on
        """
        print(f"\n{'='*70}")
        print("Running MNIST Inference Container Locally")
        print(f"{'='*70}\n")
        
        # Check if model exists
        model_path = self.local_model_dir / "mnist_cnn.pt"
        if not model_path.exists():
            print(f"✗ Model not found at: {model_path}")
            print("\nPlease run training first:")
            print("  python tools/run_local.py --train")
            sys.exit(1)
        
        print("Configuration:")
        print(f"  Image: mnist-infer:latest")
        print(f"  Model path: {model_path}")
        print(f"  Port: {port}")
        print(f"  URL: http://localhost:{port}/")
        print()
        
        # Docker run command
        cmd = [
            "run",
            "--rm",  # Remove container after exit
            "-it",   # Interactive terminal
            "-v", f"{self.local_model_dir.absolute()}:/mnt/model:ro",  # Mount model (read-only)
            "-e", "MODEL_DIR=/mnt/model",  # Set environment variable
            "-p", f"{port}:8000",  # Port mapping
            "mnist-infer:latest"
        ]
        
        print("Starting inference service...")
        print(f"Access the web UI at: http://localhost:{port}/")
        print("Press Ctrl+C to stop the service.\n")
        
        try:
            self.run_docker(cmd)
        except KeyboardInterrupt:
            print(f"\n\n{'='*70}")
            print("✓ Inference service stopped")
            print(f"{'='*70}\n")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Inference service failed with exit code {e.returncode}")
            sys.exit(1)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run MNIST containers locally without Kubernetes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run training locally
  python tools/run_local.py --train
  
  # Run inference locally (default port 8000)
  python tools/run_local.py --infer
  
  # Run inference on custom port
  python tools/run_local.py --infer --port 8080
  
  # Full workflow
  python tools/run_local.py --train   # Train model
  python tools/run_local.py --infer   # Serve predictions

Notes:
  - Training saves model to local_model/ directory
  - Inference reads model from local_model/ directory
  - Both containers must be built first (use tools/build_images.py)
        """
    )
    
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training container"
    )
    
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Run inference container"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for inference service (default: 8000)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (args.train or args.infer):
        parser.error("Please specify --train or --infer")
    
    if args.train and args.infer:
        parser.error("Cannot run both --train and --infer simultaneously")
    
    # Initialize runner
    runner = LocalRunner()
    
    # Run requested service
    if args.train:
        runner.run_training()
    elif args.infer:
        runner.run_inference(port=args.port)


if __name__ == "__main__":
    main()
