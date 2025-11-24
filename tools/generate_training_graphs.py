"""
Generate Training Visualization Graphs

This script reads the training metadata and generates professional matplotlib graphs
for inclusion in the project report.

Usage:
    python tools/generate_training_graphs.py
    
Output:
    - training_accuracy.png
    - training_loss.png
    - training_combined.png
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_training_data(metadata_path):
    """Load training data from metadata.json"""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    history = metadata['training_history']
    epochs = [h['epoch'] for h in history]
    train_acc = [h['train_accuracy'] for h in history]
    test_acc = [h['test_accuracy'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    test_loss = [h['test_loss'] for h in history]
    
    return epochs, train_acc, test_acc, train_loss, test_loss

def plot_accuracy(epochs, train_acc, test_acc, output_path):
    """Generate accuracy vs epoch graph"""
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, train_acc, 'o-', linewidth=2, markersize=8, 
             label='Training Accuracy', color='#667eea')
    plt.plot(epochs, test_acc, 's-', linewidth=2, markersize=8, 
             label='Test Accuracy', color='#764ba2')
    
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Training and Test Accuracy vs. Epoch', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim([90, 100])
    
    # Add value labels on points
    for i, (e, ta, va) in enumerate(zip(epochs, train_acc, test_acc)):
        plt.annotate(f'{ta:.2f}%', (e, ta), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9, color='#667eea')
        plt.annotate(f'{va:.2f}%', (e, va), textcoords="offset points", 
                    xytext=(0,-15), ha='center', fontsize=9, color='#764ba2')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_loss(epochs, train_loss, test_loss, output_path):
    """Generate loss vs epoch graph"""
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, train_loss, 'o-', linewidth=2, markersize=8, 
             label='Training Loss', color='#667eea')
    plt.plot(epochs, test_loss, 's-', linewidth=2, markersize=8, 
             label='Test Loss', color='#764ba2')
    
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Training and Test Loss vs. Epoch', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels on points
    for i, (e, tl, vl) in enumerate(zip(epochs, train_loss, test_loss)):
        plt.annotate(f'{tl:.4f}', (e, tl), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9, color='#667eea')
        plt.annotate(f'{vl:.4f}', (e, vl), textcoords="offset points", 
                    xytext=(0,-15), ha='center', fontsize=9, color='#764ba2')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_combined(epochs, train_acc, test_acc, train_loss, test_loss, output_path):
    """Generate combined accuracy and loss graph"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy subplot
    ax1.plot(epochs, train_acc, 'o-', linewidth=2, markersize=8, 
             label='Training Accuracy', color='#667eea')
    ax1.plot(epochs, test_acc, 's-', linewidth=2, markersize=8, 
             label='Test Accuracy', color='#764ba2')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy vs. Epoch', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([90, 100])
    
    # Loss subplot
    ax2.plot(epochs, train_loss, 'o-', linewidth=2, markersize=8, 
             label='Training Loss', color='#667eea')
    ax2.plot(epochs, test_loss, 's-', linewidth=2, markersize=8, 
             label='Test Loss', color='#764ba2')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Loss vs. Epoch', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('MNIST CNN Training Progress', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def main():
    """Main function to generate all graphs"""
    # Paths
    project_root = Path(__file__).parent.parent
    metadata_path = project_root / 'local_model' / 'metadata.json'
    output_dir = project_root / 'figures'
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Check if metadata exists
    if not metadata_path.exists():
        print(f"✗ Error: {metadata_path} not found")
        print("  Please run training first to generate metadata.json")
        return
    
    print("=" * 60)
    print("Generating Training Visualization Graphs")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from: {metadata_path}")
    epochs, train_acc, test_acc, train_loss, test_loss = load_training_data(metadata_path)
    print(f"✓ Loaded {len(epochs)} epochs of training data")
    
    # Generate graphs
    print("\nGenerating graphs...")
    plot_accuracy(epochs, train_acc, test_acc, output_dir / 'training_accuracy.png')
    plot_loss(epochs, train_loss, test_loss, output_dir / 'training_loss.png')
    plot_combined(epochs, train_acc, test_acc, train_loss, test_loss, 
                  output_dir / 'training_combined.png')
    
    print("\n" + "=" * 60)
    print("✓ All graphs generated successfully!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  - training_accuracy.png  (Accuracy vs. Epoch)")
    print("  - training_loss.png      (Loss vs. Epoch)")
    print("  - training_combined.png  (Combined view)")
    print("\nYou can now insert these images into your report.")

if __name__ == '__main__':
    main()
