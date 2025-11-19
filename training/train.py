"""
MNIST Training Script for Kubernetes Job.

This script is designed to run inside a Kubernetes Job container with a 
PersistentVolumeClaim (PVC) mounted at /mnt/model. It will:
1. Download MNIST dataset automatically
2. Train a CNN model on CPU
3. Save the trained model to the PVC
4. Export training metadata (accuracy, hyperparameters)

Environment Variables:
- MODEL_DIR: Directory to save model (default: /mnt/model)
- EPOCHS: Number of training epochs (default: 5)
- BATCH_SIZE: Training batch size (default: 64)
- LR: Learning rate (default: 0.01)
"""
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model_def import create_model


def get_args():
    """Parse command line arguments and environment variables."""
    parser = argparse.ArgumentParser(description='Train MNIST CNN model')
    
    parser.add_argument('--model-dir', type=str, 
                        default=os.environ.get('MODEL_DIR', '/mnt/model'),
                        help='Directory to save the trained model')
    parser.add_argument('--epochs', type=int,
                        default=int(os.environ.get('EPOCHS', '5')),
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int,
                        default=int(os.environ.get('BATCH_SIZE', '64')),
                        help='Training batch size')
    parser.add_argument('--lr', type=float,
                        default=float(os.environ.get('LR', '0.01')),
                        help='Learning rate')
    parser.add_argument('--data-dir', type=str,
                        default='/tmp/mnist',
                        help='Directory to download/store MNIST data')
    
    return parser.parse_args()


def get_data_loaders(data_dir, batch_size):
    """
    Download MNIST dataset and create train/test data loaders.
    
    Args:
        data_dir: Directory to store MNIST data
        batch_size: Batch size for data loaders
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Define transforms (normalize to mean=0.1307, std=0.3081 for MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download training data
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # Download test data
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader


def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """
    Train the model for one epoch.
    
    Args:
        model: Neural network model
        device: Device to train on (CPU)
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        epoch: Current epoch number
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(model, device, test_loader, criterion):
    """
    Evaluate the model on test data.
    
    Args:
        model: Neural network model
        device: Device to evaluate on (CPU)
        test_loader: Test data loader
        criterion: Loss function
    
    Returns:
        Tuple of (test_loss, test_accuracy)
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    return test_loss, accuracy


def save_model_and_metadata(model, model_dir, metadata):
    """
    Save the trained model and metadata to disk.
    
    Args:
        model: Trained model
        model_dir: Directory to save model
        metadata: Dictionary containing training metadata
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    model_path = model_dir / 'mnist_cnn.pt'
    torch.save(model.state_dict(), model_path)
    print(f'\n✓ Model saved to: {model_path}')
    
    # Save metadata as JSON
    metadata_path = model_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'✓ Metadata saved to: {metadata_path}')


def main():
    """Main training function."""
    args = get_args()
    
    print('=' * 60)
    print('MNIST CNN Training')
    print('=' * 60)
    print(f'Model directory: {args.model_dir}')
    print(f'Epochs: {args.epochs}')
    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.lr}')
    print(f'Data directory: {args.data_dir}')
    print('=' * 60)
    
    # Set device (CPU only)
    device = torch.device('cpu')
    print(f'\nUsing device: {device}')
    
    # Load data
    print('\nLoading MNIST dataset...')
    train_loader, test_loader = get_data_loaders(args.data_dir, args.batch_size)
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')
    
    # Create model
    print('\nCreating model...')
    model = create_model().to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # Training loop
    print('\nStarting training...')
    print('-' * 60)
    
    train_history = []
    best_test_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)
        
        # Log results
        print(f'\nEpoch {epoch}/{args.epochs}:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%')
        print('-' * 60)
        
        # Track history
        train_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'test_loss': test_loss,
            'test_accuracy': test_acc
        })
        
        # Track best accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
    
    # Final evaluation
    final_test_loss, final_test_acc = evaluate(model, device, test_loader, criterion)
    
    print('\n' + '=' * 60)
    print('Training Complete!')
    print(f'Final Test Accuracy: {final_test_acc:.2f}%')
    print(f'Best Test Accuracy: {best_test_acc:.2f}%')
    print('=' * 60)
    
    # Prepare metadata
    metadata = {
        'model_name': 'MnistCNN',
        'training_date': datetime.now().isoformat(),
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'optimizer': 'SGD',
            'momentum': 0.9
        },
        'dataset': {
            'name': 'MNIST',
            'train_samples': len(train_loader.dataset),
            'test_samples': len(test_loader.dataset)
        },
        'results': {
            'final_test_accuracy': final_test_acc,
            'final_test_loss': final_test_loss,
            'best_test_accuracy': best_test_acc
        },
        'training_history': train_history
    }
    
    # Save model and metadata
    print('\nSaving model and metadata...')
    save_model_and_metadata(model, args.model_dir, metadata)
    
    print('\n✓ Training pipeline completed successfully!')


if __name__ == '__main__':
    main()
