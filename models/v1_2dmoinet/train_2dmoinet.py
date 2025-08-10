import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
from dataset import PreprocessedGrapheneDataset
from transforms import get_basic_transform
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross Entropy Loss for class imbalance"""
    def __init__(self, weights=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = weights

    def forward(self, inputs, targets):
        if self.weights is not None:
            return nn.CrossEntropyLoss(weight=self.weights)(inputs, targets)
        else:
            return nn.CrossEntropyLoss()(inputs, targets)

class DeepLabHead(nn.Sequential):
    """Enhanced DeepLab head with batch normalization"""
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # Add dropout for regularization
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

def get_2dmoinet_model(num_classes=4):
    """Create 2DMOINet model based on DeepLabV3 with VGG16-like features"""
    model = deeplabv3_resnet50(pretrained=True)
    
    # Replace the classifier head
    model.classifier = DeepLabHead(2048, num_classes)
    
    return model

class LearningRateScheduler:
    """Learning rate scheduler with warmup and decay"""
    def __init__(self, optimizer, initial_lr=1e-4, warmup_epochs=5, decay_factor=0.1):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.decay_factor = decay_factor
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.initial_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Decay phase
            lr = self.initial_lr * (self.decay_factor ** ((self.current_epoch - self.warmup_epochs) // 10))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr

def calculate_metrics(predictions, targets, num_classes=4):
    """Calculate various metrics for evaluation"""
    # Convert to numpy if they're tensors
    if torch.is_tensor(predictions):
        pred_np = predictions.cpu().numpy()
    else:
        pred_np = predictions
    
    if torch.is_tensor(targets):
        target_np = targets.cpu().numpy()
    else:
        target_np = targets
    
    # Flatten for metrics calculation
    pred_flat = pred_np.flatten()
    target_flat = target_np.flatten()
    
    # Calculate accuracy
    accuracy = accuracy_score(target_flat, pred_flat)
    
    # Calculate confusion matrix
    cm = confusion_matrix(target_flat, pred_flat, labels=range(num_classes))
    
    # Calculate IoU for each class
    ious = []
    for i in range(num_classes):
        intersection = np.logical_and(pred_flat == i, target_flat == i).sum()
        union = np.logical_or(pred_flat == i, target_flat == i).sum()
        iou = intersection / (union + 1e-8)
        ious.append(iou)
    
    mean_iou = np.mean(ious)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'ious': ious,
        'mean_iou': mean_iou
    }

def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy plot
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # IoU plot
    axes[1, 0].plot(history['train_iou'], label='Train IoU')
    axes[1, 0].plot(history['val_iou'], label='Val IoU')
    axes[1, 0].set_title('Mean IoU')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate plot
    axes[1, 1].plot(history['lr'])
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to: {save_path}")
    
    plt.show()

def train_2dmoinet():
    """Main training function for 2DMOINet"""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    num_classes = 4
    batch_size = 4
    num_epochs = 50
    learning_rate = 1e-4
    
    # Data paths
    train_img_dir = 'augmented_dataset/augmented_images'
    train_mask_dir = 'augmented_dataset/augmented_masks'
    
    # Create datasets
    train_dataset = PreprocessedGrapheneDataset(
        train_img_dir, 
        train_mask_dir, 
        transform=get_basic_transform(),
        num_classes=num_classes
    )
    
    # Calculate class weights
    class_weights = train_dataset.get_class_weights().to(device)
    print(f"Class weights: {class_weights}")
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False  # Set to False since we're not using GPU
    )
    
    # Create model
    model = get_2dmoinet_model(num_classes=num_classes).to(device)
    
    # Loss function (weighted cross entropy)
    criterion = WeightedCrossEntropyLoss(weights=class_weights)
    
    # Optimizer (SGD with momentum as per paper)
    optimizer = optim.SGD(
        model.parameters(), 
        lr=learning_rate, 
        momentum=0.9, 
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = LearningRateScheduler(optimizer, initial_lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_iou': [],
        'lr': []
    }
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Batch size: {batch_size}")
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_predictions = []
        epoch_targets = []
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            predictions = torch.argmax(outputs, dim=1)
            epoch_predictions.extend(predictions.cpu().numpy())
            epoch_targets.extend(masks.cpu().numpy())
            
            epoch_loss += loss.item()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(train_loader)
        metrics = calculate_metrics(
            np.array(epoch_predictions), 
            np.array(epoch_targets), 
            num_classes
        )
        
        # Update learning rate
        current_lr = scheduler.step()
        
        # Store history
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(metrics['accuracy'])
        history['train_iou'].append(metrics['mean_iou'])
        history['lr'].append(current_lr)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Class IoUs: {[f'{iou:.3f}' for iou in metrics['ious']]}")
        print("-" * 50)
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'metrics': metrics,
                'class_weights': class_weights
            }
            torch.save(checkpoint, f"graphene_2dmoinet_epoch_{epoch+1}.pth")
    
    # Save final model
    torch.save(model.state_dict(), "graphene_2dmoinet_final.pth")
    print("Final model saved as graphene_2dmoinet_final.pth")
    
    # Plot training history
    plot_training_history(history, "output/training_history.png")
    
    return model, history

if __name__ == "__main__":
    train_2dmoinet() 