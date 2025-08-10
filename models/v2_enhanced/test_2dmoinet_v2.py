import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm

# Import V2 components
from dataset import GrapheneSegmentationDataset
from transforms import BasicTransform
from model_architectures import get_enhanced_model
from losses import get_loss_function

def load_model(model_path, num_classes=4, device='cuda'):
    """Load the trained V2 model"""
    print(f"Loading model from {model_path}")
    
    # Get the enhanced model
    model = get_enhanced_model(
        model_type='enhanced',
        num_classes=num_classes
    )
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both checkpoint format and direct state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully. Model info:")
    print_model_info(model)
    
    return model

def print_model_info(model):
    """Print model architecture information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model architecture: {model.__class__.__name__}")

def preprocess_image(image_path, transform):
    """Preprocess a single image for prediction"""
    # Load image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path
    
    # Apply transform
    image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def predict_single_image(model, image_tensor, device='cuda'):
    """Make prediction on a single image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        
        # Extract output from dictionary if needed
        if isinstance(output, dict):
            output = output['out']
        
        # Apply softmax to get probabilities
        probs = F.softmax(output, dim=1)
        
        # Get predicted class
        predictions = torch.argmax(output, dim=1)
        
        return predictions, probs

def visualize_predictions(image_path, predictions, probs, save_path=None, show_legend=True):
    """Visualize predictions with overlay and probability information"""
    try:
        # Handle image input
        if isinstance(image_path, str):
            # Load from file path
            original_image = cv2.imread(image_path)
            if original_image is None:
                print(f"Error: Could not load image {image_path}")
                return
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        elif torch.is_tensor(image_path):
            # Handle PyTorch tensor input
            img_np = image_path.cpu().numpy()
            
            # Handle different tensor formats
            if img_np.ndim == 4:  # [batch, channels, height, width]
                img_np = img_np[0]  # Take first batch
            if img_np.ndim == 3 and img_np.shape[0] == 3:  # [channels, height, width]
                img_np = np.transpose(img_np, (1, 2, 0))  # Convert to [height, width, channels]
            
            # Normalize to 0-255 range
            if img_np.max() <= 1.0:
                original_image = (img_np * 255).astype(np.uint8)
            else:
                original_image = img_np.astype(np.uint8)
        else:
            # Assume it's already a numpy array
            original_image = np.array(image_path)
            if original_image.ndim == 3 and original_image.shape[0] == 3:
                original_image = np.transpose(original_image, (1, 2, 0))
        
        # Handle predictions input
        if torch.is_tensor(predictions):
            pred_np = predictions.cpu().numpy()
        else:
            pred_np = np.array(predictions)
        
        # Remove batch dimension if present
        if pred_np.ndim > 2:
            pred_np = pred_np.squeeze()
        
        # Ensure prediction mask is 2D
        if pred_np.ndim != 2:
            print(f"Warning: Prediction mask has unexpected shape {pred_np.shape}, squeezing...")
            pred_np = pred_np.squeeze()
        
        # Verify dimensions match
        if pred_np.shape[:2] != original_image.shape[:2]:
            print(f"Warning: Shape mismatch - image: {original_image.shape}, mask: {pred_np.shape}")
            # Resize mask to match image if needed
            pred_np = cv2.resize(pred_np.astype(np.uint8), (original_image.shape[1], original_image.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
        
        # Create color map for 4 classes
        colors = [
            [0, 0, 0],      # Background - Black
            [255, 0, 0],    # 1 layer - Red
            [0, 255, 0],    # 2 layers - Green
            [0, 0, 255]     # 3+ layers - Blue
        ]
        
        # Create colored mask
        colored_mask = np.zeros_like(original_image)
        for i, color in enumerate(colors):
            mask = (pred_np == i)
            # Apply mask to each channel
            for c in range(3):
                colored_mask[:, :, c][mask] = color[c]
        
        # Create overlay
        overlay = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Prediction mask
        axes[1].imshow(colored_mask)
        axes[1].set_title('Prediction Mask', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        # Add legend if requested
        if show_legend:
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor='black', label='Background'),
                plt.Rectangle((0, 0), 1, 1, facecolor='red', label='1 Layer'),
                plt.Rectangle((0, 0), 1, 1, facecolor='green', label='2 Layers'),
                plt.Rectangle((0, 0), 1, 1, facecolor='blue', label='3+ Layers')
            ]
            axes[2].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        plt.close()
        
    except Exception as e:
        print(f"Error in visualize_predictions: {e}")
        print(f"Image shape: {original_image.shape if 'original_image' in locals() else 'Unknown'}")
        print(f"Mask shape: {pred_np.shape if 'pred_np' in locals() else 'Unknown'}")
        import traceback
        traceback.print_exc()

def evaluate_model(model, test_loader, device='cuda', save_dir='results/v2'):
    """Evaluate the model on test dataset"""
    print("Starting model evaluation...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_images = []
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            # Extract the output from the dictionary
            if isinstance(outputs, dict):
                outputs = outputs['out']
            predictions = torch.argmax(outputs, dim=1)
            
            # Store results
            all_predictions.append(predictions.cpu())
            all_targets.append(masks.cpu())
            all_images.append(images.cpu())
            
            # Visualize first few batches
            if batch_idx < 4:
                for i in range(min(images.size(0), 2)):  # Show max 2 images per batch
                    save_path = os.path.join(save_dir, f'v2_comparison_{batch_idx}_{i}.png')
                    visualize_predictions(
                        images[i].cpu().numpy(),
                        predictions[i:i+1],
                        None,
                        save_path=save_path
                    )
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    predictions_flat = all_predictions.flatten().numpy()
    targets_flat = all_targets.flatten().numpy()
    
    # Confusion matrix
    cm = confusion_matrix(targets_flat, predictions_flat, labels=[0, 1, 2, 3])
    
    # Calculate IoU for each class
    iou_scores = []
    for class_id in range(4):
        intersection = np.logical_and(targets_flat == class_id, predictions_flat == class_id).sum()
        union = np.logical_or(targets_flat == class_id, predictions_flat == class_id).sum()
        iou = intersection / union if union > 0 else 0
        iou_scores.append(iou)
    
    # Calculate overall metrics
    accuracy = (predictions_flat == targets_flat).mean()
    mean_iou = np.mean(iou_scores)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print("\nPer-class IoU:")
    class_names = ['Background', '1 Layer', '2 Layers', '3+ Layers']
    for i, (name, iou) in enumerate(zip(class_names, iou_scores)):
        print(f"  {name}: {iou:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - V2 Model', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    
    cm_save_path = os.path.join(save_dir, 'v2_confusion_matrix.png')
    plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {cm_save_path}")
    plt.show()
    plt.close()
    
    # Save detailed results
    results = {
        'accuracy': float(accuracy),
        'mean_iou': float(mean_iou),
        'per_class_iou': {name: float(iou) for name, iou in zip(class_names, iou_scores)},
        'confusion_matrix': cm.tolist()
    }
    
    import json
    results_path = os.path.join(save_dir, 'v2_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to {results_path}")
    
    return results

def main():
    """Main testing function"""
    print("Starting V2 Model Testing Pipeline")
    print("="*50)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model path
    model_path = 'checkpoints/v2/graphene_2dmoinet_v2_best.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        print("Please train the V2 model first or check the checkpoint path.")
        return
    
    # Load model
    model = load_model(model_path, num_classes=4, device=device)
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_transform = BasicTransform()
    
    # Use the augmented dataset for testing
    test_dataset = GrapheneSegmentationDataset(
        image_dir='data/augmented/augmented_dataset/augmented_images',
        mask_dir='data/augmented/augmented_dataset/augmented_masks',
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"Test dataset loaded: {len(test_dataset)} images")
    
    # Evaluate model
    results = evaluate_model(model, test_loader, device=device)
    
    print("\n" + "="*50)
    print("TESTING COMPLETED SUCCESSFULLY")
    print("="*50)
    print("Results have been saved to the results/v2/ directory.")
    print("Check the confusion matrix and comparison images for detailed analysis.")

if __name__ == "__main__":
    main() 