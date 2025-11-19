"""
MNIST CNN Model Definition for Inference.

This module contains the same model architecture used during training,
plus a helper function to load the trained model for CPU-based inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class MnistCNN(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.
    
    Architecture:
    - Conv Layer 1: 1 -> 32 channels, 3x3 kernel
    - Conv Layer 2: 32 -> 64 channels, 3x3 kernel
    - Fully Connected 1: 3136 -> 128
    - Fully Connected 2: 128 -> 10 (output classes)
    
    This architecture is identical to the training model to ensure compatibility.
    """
    
    def __init__(self):
        super(MnistCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Fully connected layers
        # After 2 max pooling layers (2x2), 28x28 -> 14x14 -> 7x7
        # 64 channels * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
        
        Returns:
            Output logits of shape (batch_size, 10)
        """
        # Conv block 1: conv -> relu -> max_pool
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Conv block 2: conv -> relu -> max_pool
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Flatten for fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers with dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def load_model(model_path: str) -> nn.Module:
    """
    Load a trained MNIST model from disk for CPU inference.
    
    This function:
    1. Creates a new MnistCNN instance
    2. Loads the saved state_dict from the given path
    3. Sets the model to evaluation mode
    4. Ensures CPU-only execution (no CUDA)
    
    Args:
        model_path: Path to the saved model file (e.g., '/mnt/model/mnist_cnn.pt')
    
    Returns:
        Loaded model in evaluation mode, ready for inference
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    model_path = Path(model_path)
    
    # Check if model file exists
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Ensure the training job has completed and saved the model to the PVC."
        )
    
    # Create model instance
    model = MnistCNN()
    
    # Load state dict with CPU mapping (no CUDA)
    # map_location ensures the model loads on CPU even if it was trained on GPU
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    # Set to evaluation mode (disables dropout, batch norm, etc.)
    model.eval()
    
    print(f"✓ Model loaded successfully from {model_path}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    return model


def create_model() -> nn.Module:
    """
    Helper function to create and return a new MnistCNN model instance.
    
    Returns:
        MnistCNN model instance
    """
    return MnistCNN()
