import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def color_normalization(image):
    """Color normalization as described in the paper"""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Normalize L channel (lightness)
    l_norm = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
    
    # Normalize a and b channels
    a_norm = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
    b_norm = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    
    # Merge normalized channels
    lab_norm = cv2.merge([l_norm, a_norm, b_norm])
    
    # Convert back to BGR
    normalized = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)
    
    return normalized

def apply_clahe_enhanced(image):
    """Enhanced CLAHE with better parameters for graphene detection"""
    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel with optimized parameters
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Merge channels
    lab_enhanced = cv2.merge([l_clahe, a, b])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def resize_image(image, target_size=(224, 224)):
    """Resize image to target size as specified in the paper"""
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return resized

def preprocess_single_image(image_path, output_path, target_size=(224, 224)):
    """Complete preprocessing pipeline for a single image"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return False
    
    # Apply preprocessing pipeline
    # 1. Color normalization
    normalized = color_normalization(image)
    
    # 2. CLAHE enhancement
    enhanced = apply_clahe_enhanced(normalized)
    
    # 3. Resize to target size
    resized = resize_image(enhanced, target_size)
    
    # Save processed image
    cv2.imwrite(output_path, resized)
    
    return True

def preprocess_dataset(input_dir, output_dir, target_size=(224, 224)):
    """Preprocess entire dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    total_count = 0
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            total_count += 1
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            if preprocess_single_image(input_path, output_path, target_size):
                processed_count += 1
                print(f"Processed: {filename}")
            else:
                print(f"Failed to process: {filename}")
    
    print(f"\nPreprocessing complete!")
    print(f"Processed: {processed_count}/{total_count} images")
    print(f"Output directory: {output_dir}")

def visualize_preprocessing(image_path, save_path=None):
    """Visualize the preprocessing steps"""
    # Load original image
    original = cv2.imread(image_path)
    if original is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Apply preprocessing steps
    normalized = color_normalization(original)
    enhanced = apply_clahe_enhanced(normalized)
    resized = resize_image(enhanced, (224, 224))
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Normalized
    axes[0, 1].imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Color Normalized')
    axes[0, 1].axis('off')
    
    # CLAHE Enhanced
    axes[1, 0].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('CLAHE Enhanced')
    axes[1, 0].axis('off')
    
    # Final (Resized)
    axes[1, 1].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Final (224x224)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # Example usage
    print("Enhanced Preprocessing Pipeline")
    print("=" * 40)
    
    # Process quadrant dataset
    if os.path.exists("quadrant_dataset/images"):
        print("Processing quadrant dataset...")
        preprocess_dataset(
            "quadrant_dataset/images",
            "quadrant_dataset/processed_images"
        )
    
    # Visualize preprocessing on a sample image
    if os.path.exists("images/3-1-1-100x.jpg"):
        print("\nCreating preprocessing visualization...")
        visualize_preprocessing(
            "images/3-1-1-100x.jpg",
            "output/preprocessing_visualization.png"
        ) 