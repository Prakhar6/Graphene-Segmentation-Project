import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in segmentation tasks.
    Reduces the relative loss for well-classified examples and focuses on hard examples.
    """
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for better handling of class imbalance and improving segmentation boundaries.
    """
    
    def __init__(self, smooth=1e-6, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Convert targets to one-hot encoding
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Apply softmax to inputs
        inputs_softmax = F.softmax(inputs, dim=1)
        
        # Calculate intersection and union
        intersection = (inputs_softmax * targets_one_hot).sum(dim=(2, 3))
        union = inputs_softmax.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        # Calculate Dice coefficient
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class BoundaryLoss(nn.Module):
    """
    Boundary Loss to improve segmentation boundary accuracy.
    Penalizes misclassification at object boundaries.
    """
    
    def __init__(self, reduction='mean'):
        super(BoundaryLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Convert targets to one-hot encoding
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Apply softmax to inputs
        inputs_softmax = F.softmax(inputs, dim=1)
        
        # Calculate boundary loss using gradient magnitude
        boundary_loss = 0
        for i in range(num_classes):
            # Calculate gradient magnitude for each class
            grad_x = torch.abs(inputs_softmax[:, i, :, 1:] - inputs_softmax[:, i, :, :-1])
            grad_y = torch.abs(inputs_softmax[:, i, 1:, :] - inputs_softmax[:, i, :-1, :])
            
            # Calculate target gradient magnitude
            target_grad_x = torch.abs(targets_one_hot[:, i, :, 1:] - targets_one_hot[:, i, :, :-1])
            target_grad_y = torch.abs(targets_one_hot[:, i, 1:, :] - targets_one_hot[:, i, :-1, :])
            
            # Boundary loss for this class
            class_boundary_loss = torch.mean(torch.abs(grad_x - target_grad_x)) + \
                                 torch.mean(torch.abs(grad_y - target_grad_y))
            boundary_loss += class_boundary_loss
        
        if self.reduction == 'mean':
            return boundary_loss / num_classes
        elif self.reduction == 'sum':
            return boundary_loss
        else:
            return boundary_loss


class CombinedLoss(nn.Module):
    """
    Combined Loss combining Focal Loss, Dice Loss, and Boundary Loss for optimal performance.
    """
    
    def __init__(self, focal_weight=0.4, dice_weight=0.4, boundary_weight=0.2, 
                 focal_alpha=1, focal_gamma=2, dice_smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.boundary_loss = BoundaryLoss()
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        boundary = self.boundary_loss(inputs, targets)
        
        total_loss = (self.focal_weight * focal + 
                     self.dice_weight * dice + 
                     self.boundary_weight * boundary)
        
        return total_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss with class weights to handle imbalanced data.
    """
    
    def __init__(self, class_weights=None, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        if self.class_weights is not None:
            # Apply class weights
            weights = self.class_weights.to(inputs.device)
            return F.cross_entropy(inputs, targets, weight=weights, reduction=self.reduction)
        else:
            return F.cross_entropy(inputs, targets, reduction=self.reduction)


class LovaszSoftmax(nn.Module):
    """
    Lovasz-Softmax loss for better optimization of the mean IoU metric.
    """
    
    def __init__(self, reduction='mean'):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Convert targets to one-hot encoding
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Apply softmax to inputs
        inputs_softmax = F.softmax(inputs, dim=1)
        
        # Calculate Lovasz loss
        loss = 0
        for i in range(num_classes):
            # Binary case for each class
            pred = inputs_softmax[:, i, :, :]
            target = targets_one_hot[:, i, :, :]
            
            # Calculate intersection over union
            intersection = (pred * target).sum(dim=(1, 2))
            union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) - intersection
            
            # IoU
            iou = (intersection + 1e-6) / (union + 1e-6)
            
            # Lovasz loss for this class
            class_loss = 1 - iou
            loss += class_loss.mean()
        
        if self.reduction == 'mean':
            return loss / num_classes
        elif self.reduction == 'sum':
            return loss
        else:
            return loss


def get_loss_function(loss_type='combined', class_weights=None, **kwargs):
    """
    Factory function to get the specified loss function.
    
    Args:
        loss_type: Type of loss function ('focal', 'dice', 'boundary', 'combined', 'weighted_ce', 'lovasz')
        class_weights: Class weights for weighted loss functions
        **kwargs: Additional arguments for loss functions
    
    Returns:
        Loss function instance
    """
    
    if loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'dice':
        return DiceLoss(**kwargs)
    elif loss_type == 'boundary':
        return BoundaryLoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedLoss(**kwargs)
    elif loss_type == 'weighted_ce':
        return WeightedCrossEntropyLoss(class_weights=class_weights, **kwargs)
    elif loss_type == 'lovasz':
        return LovaszSoftmax(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
