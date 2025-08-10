import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from train_2dmoinet import get_2dmoinet_model, calculate_metrics
from dataset import PreprocessedGrapheneDataset
from transforms import get_basic_transform

class GrapheneSegmenter:
    """Class for segmenting graphene images using trained 2DMOINet"""
    
    def __init__(self, model_path, num_classes=4, device=None):
        self.num_classes = num_classes
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = get_2dmoinet_model(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Using device: {self.device}")
    
    def preprocess_image(self, image_path):
        """Preprocess a single image for inference"""
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize to 224x224
        image = cv2.resize(image, (224, 224))
        
        # Convert to tensor and normalize
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = image.unsqueeze(0).to(self.device)
        
        return image
    
    def segment_image(self, image_path):
        """Segment a single image"""
        with torch.no_grad():
            # Preprocess image
            image = self.preprocess_image(image_path)
            
            # Forward pass
            output = self.model(image)['out']
            predictions = torch.argmax(output, dim=1)
            
            # Convert to numpy
            pred_mask = predictions.cpu().numpy()[0]
            
            return pred_mask
    
    def segment_batch(self, image_paths):
        """Segment multiple images"""
        results = []
        
        for image_path in image_paths:
            try:
                pred_mask = self.segment_image(image_path)
                results.append({
                    'image_path': image_path,
                    'prediction': pred_mask,
                    'success': True
                })
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'prediction': None,
                    'success': False,
                    'error': str(e)
                })
        
        return results

def create_colored_mask(mask, num_classes=4):
    """Create colored mask for visualization"""
    # Color map for different classes
    colors = [
        [0, 0, 0],      # Background - Black
        [255, 0, 0],    # 1 layer - Red
        [0, 255, 0],    # 2 layers - Green
        [0, 0, 255]     # 3+ layers - Blue
    ]
    
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    for class_id in range(num_classes):
        colored_mask[mask == class_id] = colors[class_id]
    
    return colored_mask

def visualize_predictions(image_path, pred_mask, true_mask=None, save_path=None):
    """Visualize predictions with original image and masks"""
    # Handle both image paths and preloaded images
    if isinstance(image_path, str):
        # Load original image from file
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Warning: Could not load image from {image_path}")
            return
        original_image = cv2.resize(original_image, (224, 224))
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    else:
        # Use preloaded image (assumed to be numpy array)
        original_image = image_path
        if len(original_image.shape) == 3 and original_image.shape[0] == 3:
            # Convert from CxHxW to HxWxC format
            original_image = np.transpose(original_image, (1, 2, 0))
        # Normalize to 0-255 range if needed
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        original_image_rgb = original_image
    
    # Create colored prediction mask
    colored_pred = create_colored_mask(pred_mask)
    
    # Create subplot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Prediction mask
    axes[1].imshow(colored_pred)
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    # Overlay
    overlay = original_image_rgb.copy()
    overlay = overlay.astype(np.float32) * 0.7
    colored_pred_float = colored_pred.astype(np.float32) * 0.3
    overlay = np.clip(overlay + colored_pred_float, 0, 255).astype(np.uint8)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

def evaluate_model(model_path, test_dataset, output_dir="test_results"):
    """Evaluate model on test dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize segmenter
    segmenter = GrapheneSegmenter(model_path)
    
    # Get test data
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    all_predictions = []
    all_targets = []
    all_images = []
    
    print("Evaluating model...")
    
    for i, (images, masks) in enumerate(test_loader):
        with torch.no_grad():
            # Move to device
            images = images.to(segmenter.device)
            masks = masks.to(segmenter.device)
            
            # Get predictions
            outputs = segmenter.model(images)['out']
            predictions = torch.argmax(outputs, dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(masks.cpu().numpy())
            all_images.extend(images.cpu().numpy())
            
            # Save visualization for first few samples
            if i < 10:
                pred_mask = predictions[0].cpu().numpy()
                true_mask = masks[0].cpu().numpy()
                
                # Save individual results
                save_path = os.path.join(output_dir, f"prediction_{i}.png")
                visualize_predictions(
                    images[0].cpu().numpy(), pred_mask, true_mask, save_path
                )
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    metrics = calculate_metrics(all_predictions, all_targets, segmenter.num_classes)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print("\nClass-wise IoU:")
    class_names = ['Background', '1 Layer', '2 Layers', '3+ Layers']
    for i, (name, iou) in enumerate(zip(class_names, metrics['ious'])):
        print(f"  {name}: {iou:.4f}")
    
    # Create confusion matrix visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics['confusion_matrix'], 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return metrics

def test_on_single_images(model_path, image_dir, output_dir="single_image_results"):
    """Test model on individual images"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize segmenter
    segmenter = GrapheneSegmenter(model_path)
    
    # Get list of images
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Testing on {len(image_files)} images...")
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        
        try:
            # Segment image
            pred_mask = segmenter.segment_image(image_path)
            
            # Create visualization
            save_path = os.path.join(output_dir, f"pred_{image_file.replace('.jpg', '.png').replace('.jpeg', '.png')}")
            visualize_predictions(image_path, pred_mask, save_path=save_path)
            
            print(f"Processed: {image_file}")
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

def main():
    """Main testing function"""
    print("2DMOINet Testing Pipeline")
    print("=" * 40)
    
    # Model path
    model_path = "graphene_2dmoinet_final.pth"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please train the model first using train_2dmoinet.py")
        return
    
    # Test on augmented dataset
    if os.path.exists("augmented_dataset/augmented_images"):
        print("Testing on augmented dataset...")
        test_dataset = PreprocessedGrapheneDataset(
            "augmented_dataset/augmented_images",
            "augmented_dataset/augmented_masks",
            transform=get_basic_transform(),
            num_classes=4
        )
        
        metrics = evaluate_model(model_path, test_dataset, "test_results")
    
    # Test on original quadrant images
    if os.path.exists("quadrant_dataset/images"):
        print("\nTesting on original quadrant images...")
        test_on_single_images(model_path, "quadrant_dataset/images", "single_image_results")
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main() 