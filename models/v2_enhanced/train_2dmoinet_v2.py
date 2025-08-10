#!/usr/bin/env python3
"""
Enhanced 2DMOINet Training Script for Version 2
Implements advanced training techniques including:
- Enhanced model architecture with attention mechanisms
- Advanced loss functions (Focal + Dice + Boundary)
- Learning rate scheduling with warm restarts
- Mixed precision training
- Gradient clipping
- Early stopping
- Advanced metrics tracking
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.v2_enhanced.dataset import GrapheneSegmentationDataset
from models.v2_enhanced.transforms import AdvancedTransform
from models.v2_enhanced.model_architectures import get_enhanced_model, get_model_info
from models.v2_enhanced.losses import get_loss_function

# Training configuration
CONFIG = {
    'model_type': 'enhanced',  # 'enhanced', 'lightweight', 'standard'
    'num_classes': 4,
    'img_size': 224,
    'batch_size': 4,
    'num_epochs': 20,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'loss_type': 'combined',  # 'focal', 'dice', 'boundary', 'combined', 'weighted_ce', 'lovasz'
    'scheduler_type': 'cosine_warm_restart',  # 'step', 'cosine', 'cosine_warm_restart'
    'mixed_precision': True,
    'gradient_clip': 1.0,
    'early_stopping_patience': 5,
    'save_best_only': True,
    'num_workers': 0,
    'pin_memory': False,
    'train_img_dir': 'data/augmented/augmented_dataset/augmented_images',
    'train_mask_dir': 'data/augmented/augmented_dataset/augmented_masks'
}

class CosineAnnealingWarmRestarts:
    """Cosine annealing scheduler with warm restarts."""
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self):
        if self.T_cur >= self.T_0:
            self.T_0 *= self.T_mult
            self.T_cur = 0
        
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             (1 + np.cos(np.pi * self.T_cur / self.T_0)) / 2
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.T_cur += 1
        return lr
    
    def state_dict(self):
        """Return scheduler state for checkpointing."""
        return {
            'T_0': self.T_0,
            'T_mult': self.T_mult,
            'eta_min': self.eta_min,
            'T_cur': self.T_cur,
            'base_lr': self.base_lr
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint."""
        self.T_0 = state_dict['T_0']
        self.T_mult = state_dict['T_mult']
        self.eta_min = state_dict['eta_min']
        self.T_cur = state_dict['T_cur']
        self.base_lr = state_dict['base_lr']


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


class MetricsTracker:
    """Track and compute advanced training metrics."""
    
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.total_loss = 0
        self.total_iou = 0
        self.total_dice = 0
        self.total_pixel_acc = 0
        self.num_batches = 0
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, outputs, targets, loss):
        """Update metrics with batch results."""
        self.total_loss += loss
        
        # Convert to numpy for metric calculation
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # Calculate metrics
        batch_iou = self.calculate_iou(outputs, targets)
        batch_dice = self.calculate_dice(outputs, targets)
        batch_pixel_acc = self.calculate_pixel_accuracy(outputs, targets)
        
        self.total_iou += batch_iou
        self.total_dice += batch_dice
        self.total_pixel_acc += batch_pixel_acc
        
        # Update confusion matrix
        self.update_confusion_matrix(outputs, targets)
        
        self.num_batches += 1
    
    def calculate_iou(self, outputs, targets):
        """Calculate mean IoU for the batch."""
        ious = []
        for i in range(self.num_classes):
            intersection = np.logical_and(outputs == i, targets == i).sum()
            union = np.logical_or(outputs == i, targets == i).sum()
            if union > 0:
                iou = intersection / union
                ious.append(iou)
            else:
                ious.append(0.0)
        return np.mean(ious)
    
    def calculate_dice(self, outputs, targets):
        """Calculate mean Dice coefficient for the batch."""
        dices = []
        for i in range(self.num_classes):
            intersection = np.logical_and(outputs == i, targets == i).sum()
            total = (outputs == i).sum() + (targets == i).sum()
            if total > 0:
                dice = 2 * intersection / total
                dices.append(dice)
            else:
                dices.append(0.0)
        return np.mean(dices)
    
    def calculate_pixel_accuracy(self, outputs, targets):
        """Calculate pixel accuracy for the batch."""
        correct = (outputs == targets).sum()
        total = outputs.size
        return correct / total if total > 0 else 0.0
    
    def update_confusion_matrix(self, outputs, targets):
        """Update confusion matrix."""
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self.confusion_matrix[i, j] += np.logical_and(outputs == i, targets == j).sum()
    
    def get_metrics(self):
        """Get current metrics."""
        return {
            'loss': self.total_loss / self.num_batches if self.num_batches > 0 else 0,
            'iou': self.total_iou / self.num_batches if self.num_batches > 0 else 0,
            'dice': self.total_dice / self.num_batches if self.num_batches > 0 else 0,
            'pixel_acc': self.total_pixel_acc / self.num_batches if self.num_batches > 0 else 0,
            'confusion_matrix': self.confusion_matrix.copy()
        }


def get_2dmoinet_model_v2(model_type='enhanced', num_classes=4):
    """Get the enhanced 2DMOINet model."""
    model = get_enhanced_model(model_type=model_type, num_classes=num_classes, pretrained=True)
    
    # Print model information
    model_info = get_model_info(model)
    print(f"Model Type: {model_type}")
    print(f"Total Parameters: {model_info['total_parameters']:,}")
    print(f"Trainable Parameters: {model_info['trainable_parameters']:,}")
    print(f"Model Size: {model_info['model_size_mb']:.2f} MB")
    
    return model


def calculate_class_weights(dataset):
    """Calculate class weights based on dataset distribution."""
    class_counts = np.zeros(CONFIG['num_classes'])
    
    for _, mask in dataset:
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        for i in range(CONFIG['num_classes']):
            class_counts[i] += (mask == i).sum()
    
    # Calculate weights (inverse frequency)
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (CONFIG['num_classes'] * class_counts + 1e-8)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * CONFIG['num_classes']
    
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights}")
    
    return torch.FloatTensor(class_weights)


def train_epoch(model, train_loader, criterion, optimizer, device, scaler, metrics_tracker):
    """Train for one epoch."""
    model.train()
    metrics_tracker.reset()
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        if CONFIG['mixed_precision']:
            with torch.cuda.amp.autocast():
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            
            if CONFIG['gradient_clip'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip'])
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            
            loss.backward()
            
            if CONFIG['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip'])
            
            optimizer.step()
        
        # Calculate predictions
        predictions = torch.argmax(outputs, dim=1)
        
        # Update metrics
        metrics_tracker.update(predictions.cpu().numpy(), masks.cpu().numpy(), loss.item())
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
    
    return metrics_tracker.get_metrics()


def validate_epoch(model, val_loader, criterion, device, metrics_tracker):
    """Validate for one epoch."""
    model.eval()
    metrics_tracker.reset()
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            
            # Calculate predictions
            predictions = torch.argmax(outputs, dim=1)
            
            # Update metrics
            metrics_tracker.update(predictions.cpu().numpy(), masks.cpu().numpy(), loss.item())
    
    return metrics_tracker.get_metrics()


def plot_training_history(history, save_path):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU
    axes[0, 1].plot(history['train_iou'], label='Train IoU')
    axes[0, 1].plot(history['val_iou'], label='Validation IoU')
    axes[0, 1].set_title('Training and Validation IoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Dice
    axes[1, 0].plot(history['train_dice'], label='Train Dice')
    axes[1, 0].plot(history['val_dice'], label='Validation Dice')
    axes[1, 0].set_title('Training and Validation Dice')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Pixel Accuracy
    axes[1, 1].plot(history['train_pixel_acc'], label='Train Pixel Acc')
    axes[1, 1].plot(history['val_pixel_acc'], label='Validation Pixel Acc')
    axes[1, 1].set_title('Training and Validation Pixel Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Pixel Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main training function."""
    print("Enhanced 2DMOINet Training - Version 2")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    checkpoint_dir = project_root / "checkpoints" / "v2"
    results_dir = project_root / "results" / "v2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Data paths
    train_img_dir = project_root / "data/augmented/augmented_dataset/augmented_images"
    train_mask_dir = project_root / "data/augmented/augmented_dataset/augmented_masks"
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = GrapheneSegmentationDataset(
        str(train_img_dir), 
        str(train_mask_dir), 
        transform=AdvancedTransform(img_size=CONFIG['img_size'])
    )
    
    # Split into train/validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory']
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    print("Creating model...")
    model = get_2dmoinet_model_v2(
        model_type=CONFIG['model_type'], 
        num_classes=CONFIG['num_classes']
    )
    model = model.to(device)
    
    # Calculate class weights using raw dataset (no transforms)
    raw_dataset = GrapheneSegmentationDataset(
        image_dir=str(project_root / CONFIG['train_img_dir']),
        mask_dir=str(project_root / CONFIG['train_mask_dir']),
        transform=None  # No transforms for class weight calculation
    )
    class_weights = calculate_class_weights(raw_dataset)
    
    # Create loss function
    criterion = get_loss_function(
        loss_type=CONFIG['loss_type'],
        class_weights=class_weights.to(device)
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG['learning_rate'], 
        weight_decay=CONFIG['weight_decay']
    )
    
    # Create scheduler
    if CONFIG['scheduler_type'] == 'cosine_warm_restart':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
    elif CONFIG['scheduler_type'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])
    elif CONFIG['scheduler_type'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None
    
    # Create mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if CONFIG['mixed_precision'] else None
    
    # Create early stopping
    early_stopping = EarlyStopping(patience=CONFIG['early_stopping_patience'])
    
    # Create metrics tracker
    metrics_tracker = MetricsTracker(num_classes=CONFIG['num_classes'])
    
    # Training history
    history = {
        'train_loss': [], 'train_iou': [], 'train_dice': [], 'train_pixel_acc': [],
        'val_loss': [], 'val_iou': [], 'val_dice': [], 'val_pixel_acc': [],
        'learning_rates': []
    }
    
    # Training loop
    print(f"\nStarting training for {CONFIG['num_epochs']} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG['num_epochs']):
        start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        print("-" * 30)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, metrics_tracker
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, metrics_tracker
        )
        
        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, CosineAnnealingWarmRestarts):
                current_lr = scheduler.step()
            else:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}, "
              f"Dice: {train_metrics['dice']:.4f}, Pixel Acc: {train_metrics['pixel_acc']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}, "
              f"Dice: {val_metrics['dice']:.4f}, Pixel Acc: {val_metrics['pixel_acc']:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"Time: {time.time() - start_time:.2f}s")
        
        # Update history
        for key in ['loss', 'iou', 'dice', 'pixel_acc']:
            history[f'train_{key}'].append(train_metrics[key])
            history[f'val_{key}'].append(val_metrics[key])
        history['learning_rates'].append(current_lr)
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            if CONFIG['save_best_only']:
                torch.save(model.state_dict(), checkpoint_dir / "graphene_2dmoinet_v2_best.pth")
                print("Saved best model!")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_loss': best_val_loss,
                'config': CONFIG
            }, checkpoint_dir / f"graphene_2dmoinet_v2_checkpoint_{epoch+1}.pth")
        
        # Early stopping
        if early_stopping(val_metrics['loss'], model):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    torch.save(model.state_dict(), checkpoint_dir / "graphene_2dmoinet_v2_final.pth")
    
    # Plot training history
    plot_training_history(history, results_dir / "training_history_v2.png")
    
    # Save training history
    with open(results_dir / "training_history_v2.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save final confusion matrix
    final_confusion_matrix = val_metrics['confusion_matrix']
    plt.figure(figsize=(8, 6))
    # Convert to integers for proper formatting
    final_confusion_matrix_int = final_confusion_matrix.astype(int)
    sns.heatmap(final_confusion_matrix_int, annot=True, fmt='d', cmap='Blues')
    plt.title('Final Validation Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(results_dir / "confusion_matrix_v2.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation IoU: {val_metrics['iou']:.4f}")
    print(f"Final validation Dice: {val_metrics['dice']:.4f}")
    print(f"Final validation Pixel Accuracy: {val_metrics['pixel_acc']:.4f}")
    print(f"Results saved to: {results_dir}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main() 