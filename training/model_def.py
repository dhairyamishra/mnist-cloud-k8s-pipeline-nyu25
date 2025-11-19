"""
MNIST CNN Model Definition.
A simple convolutional neural network for MNIST digit classification.
CPU-friendly architecture suitable for training and inference without GPU.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistCNN(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.
    
    Architecture:
    - Conv Layer 1: 1 -> 32 channels, 3x3 kernel
    - Conv Layer 2: 32 -> 64 channels, 3x3 kernel
    - Fully Connected 1: 3136 -> 128
    - Fully Connected 2: 128 -> 10 (output classes)
    
    This architecture is designed to be CPU-friendly and will be used
    identically in both training and inference containers.
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


def create_model() -> nn.Module:
    """
    Helper function to create and return a new MnistCNN model instance.
    
    Returns:
        MnistCNN model instance
    """
    return MnistCNN()
