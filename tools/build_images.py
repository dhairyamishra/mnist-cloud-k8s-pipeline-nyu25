#!/usr/bin/env python3
"""
Docker Image Build Automation Script

This script automates the building and pushing of Docker images for the
MNIST training and inference services.

Usage:
    python tools/build_images.py --all                    # Build both images
    python tools/build_images.py --train                  # Build training image only
    python tools/build_images.py --infer                  # Build inference image only
    python tools/build_images.py --all --push             # Build and push to registry
    python tools/build_images.py --all --registry gcr.io/my-project
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional


class ImageBuilder:
    """Handles Docker image building and pushing operations."""
    
    def __init__(self, registry: Optional[str] = None):
        """
        Initialize the image builder.
        
        Args:
            registry: Optional Docker registry URL (e.g., 'gcr.io/my-project')
        """
        self.registry = registry
        self.project_root = Path(__file__).parent.parent
        
    def build_image(self, name: str, dockerfile_path: str, context_path: str) -> bool:
        """
        Build a Docker image.
        
        Args:
            name: Image name (e.g., 'mnist-train')
            dockerfile_path: Path to Dockerfile
            context_path: Build context directory
            
        Returns:
            True if build succeeded, False otherwise
        """
        # Construct full image name with registry if provided
        if self.registry:
            full_name = f"{self.registry}/{name}:latest"
        else:
            full_name = f"{name}:latest"
        
        print(f"\n{'='*70}")
        print(f"Building image: {full_name}")
        print(f"Dockerfile: {dockerfile_path}")
        print(f"Context: {context_path}")
        print(f"{'='*70}\n")
        
        # Build command
        cmd = [
            "docker", "build",
            "-t", full_name,
            "-f", str(self.project_root / dockerfile_path),
            str(self.project_root / context_path)
        ]
        
        try:
            # Run docker build
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Print output
            print(result.stdout)
            print(f"\n✓ Successfully built {full_name}\n")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Failed to build {full_name}")
            print(f"Error: {e.stdout}")
            return False
    
    def push_image(self, name: str) -> bool:
        """
        Push a Docker image to the registry.
        
        Args:
            name: Image name (e.g., 'mnist-train')
            
        Returns:
            True if push succeeded, False otherwise
        """
        if not self.registry:
            print(f"\n⚠ No registry specified, skipping push for {name}")
            return True
        
        full_name = f"{self.registry}/{name}:latest"
        
        print(f"\n{'='*70}")
        print(f"Pushing image: {full_name}")
        print(f"{'='*70}\n")
        
        cmd = ["docker", "push", full_name]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            print(result.stdout)
            print(f"\n✓ Successfully pushed {full_name}\n")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Failed to push {full_name}")
            print(f"Error: {e.stdout}")
            return False
    
    def build_training_image(self) -> bool:
        """Build the training image."""
        return self.build_image(
            name="mnist-train",
            dockerfile_path="training/Dockerfile",
            context_path="training"
        )
    
    def build_inference_image(self) -> bool:
        """Build the inference image."""
        return self.build_image(
            name="mnist-infer",
            dockerfile_path="inference/Dockerfile",
            context_path="inference"
        )


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Build Docker images for MNIST training and inference services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build both images locally
  python tools/build_images.py --all
  
  # Build only training image
  python tools/build_images.py --train
  
  # Build and push to Docker Hub
  python tools/build_images.py --all --push --registry yourusername
  
  # Build and push to Google Container Registry
  python tools/build_images.py --all --push --registry gcr.io/your-project-id
        """
    )
    
    # Image selection arguments
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build both training and inference images"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Build training image only"
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Build inference image only"
    )
    
    # Registry arguments
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push images to registry after building"
    )
    parser.add_argument(
        "--registry",
        type=str,
        help="Docker registry URL (e.g., 'gcr.io/my-project' or 'yourusername' for Docker Hub)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (args.all or args.train or args.infer):
        parser.error("Please specify --all, --train, or --infer")
    
    if args.push and not args.registry:
        parser.error("--push requires --registry to be specified")
    
    # Initialize builder
    builder = ImageBuilder(registry=args.registry)
    
    # Track success
    success = True
    images_to_push = []
    
    # Build training image
    if args.all or args.train:
        if builder.build_training_image():
            images_to_push.append("mnist-train")
        else:
            success = False
    
    # Build inference image
    if args.all or args.infer:
        if builder.build_inference_image():
            images_to_push.append("mnist-infer")
        else:
            success = False
    
    # Push images if requested
    if args.push and success:
        print(f"\n{'='*70}")
        print("Pushing images to registry...")
        print(f"{'='*70}\n")
        
        for image_name in images_to_push:
            if not builder.push_image(image_name):
                success = False
    
    # Summary
    print(f"\n{'='*70}")
    if success:
        print("✓ All operations completed successfully!")
    else:
        print("✗ Some operations failed. Check the output above for details.")
    print(f"{'='*70}\n")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
