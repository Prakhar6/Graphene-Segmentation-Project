import os
import cv2
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

def random_rotation_augmentation(image, mask, max_angle=30):
    """Random rotation augmentation as described in the paper"""
    # Random angle between -max_angle and +max_angle
    angle = random.uniform(-max_angle, max_angle)
    
    # Get image dimensions
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation to image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                  borderMode=cv2.BORDER_REFLECT)
    
    # Apply rotation to mask
    rotated_mask = cv2.warpAffine(mask, rotation_matrix, (w, h), 
                                 borderMode=cv2.BORDER_CONSTANT, 
                                 borderValue=0)
    
    return rotated_image, rotated_mask

def horizontal_flip(image, mask):
    """Horizontal flip augmentation"""
    flipped_image = cv2.flip(image, 1)
    flipped_mask = cv2.flip(mask, 1)
    return flipped_image, flipped_mask

def vertical_flip(image, mask):
    """Vertical flip augmentation"""
    flipped_image = cv2.flip(image, 0)
    flipped_mask = cv2.flip(mask, 0)
    return flipped_image, flipped_mask

def brightness_contrast_augmentation(image, brightness_range=0.2, contrast_range=0.2):
    """Brightness and contrast augmentation"""
    # Random brightness and contrast adjustments
    brightness = random.uniform(1 - brightness_range, 1 + brightness_range)
    contrast = random.uniform(1 - contrast_range, 1 + contrast_range)
    
    # Apply adjustments
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness * 50)
    
    # Clip values to valid range
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    
    return adjusted

def gaussian_noise(image, std=5):
    """Add Gaussian noise to image"""
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def create_augmented_dataset(input_images_dir, input_masks_dir, output_dir, augmentations_per_image=4):
    """Create augmented dataset with multiple augmentations per image"""
    
    # Create output directories
    aug_images_dir = os.path.join(output_dir, "augmented_images")
    aug_masks_dir = os.path.join(output_dir, "augmented_masks")
    os.makedirs(aug_images_dir, exist_ok=True)
    os.makedirs(aug_masks_dir, exist_ok=True)
    
    # Get list of images
    image_files = [f for f in os.listdir(input_images_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    total_augmented = 0
    
    for image_file in image_files:
        # Load image and corresponding mask
        image_path = os.path.join(input_images_dir, image_file)
        # The mask files have the correct naming pattern: {base_name}_mask.png
        base_name = os.path.splitext(image_file)[0]
        mask_file = f"{base_name}_mask.png"
        mask_path = os.path.join(input_masks_dir, mask_file)
        
        if not os.path.exists(mask_path):
            print(f"Mask not found for {image_file}, skipping...")
            continue
        
        # Load image and mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            print(f"Could not load image or mask for {image_file}, skipping...")
            continue
        
        # Save original
        base_name = os.path.splitext(image_file)[0]
        cv2.imwrite(os.path.join(aug_images_dir, f"{base_name}_original.jpg"), image)
        cv2.imwrite(os.path.join(aug_masks_dir, f"{base_name}_original_mask.png"), mask)
        total_augmented += 1
        
        # Create augmentations
        for i in range(augmentations_per_image):
            aug_image = image.copy()
            aug_mask = mask.copy()
            
            # Apply random augmentations
            augmentation_applied = False
            
            # Random rotation (always applied as per paper)
            aug_image, aug_mask = random_rotation_augmentation(aug_image, aug_mask)
            augmentation_applied = True
            
            # Random horizontal flip (50% chance)
            if random.random() < 0.5:
                aug_image, aug_mask = horizontal_flip(aug_image, aug_mask)
                augmentation_applied = True
            
            # Random vertical flip (30% chance)
            if random.random() < 0.3:
                aug_image, aug_mask = vertical_flip(aug_image, aug_mask)
                augmentation_applied = True
            
            # Brightness/contrast adjustment (40% chance)
            if random.random() < 0.4:
                aug_image = brightness_contrast_augmentation(aug_image)
                augmentation_applied = True
            
            # Gaussian noise (20% chance)
            if random.random() < 0.2:
                aug_image = gaussian_noise(aug_image)
                augmentation_applied = True
            
            # Save augmented image and mask
            aug_suffix = f"_aug_{i+1}"
            cv2.imwrite(os.path.join(aug_images_dir, f"{base_name}{aug_suffix}.jpg"), aug_image)
            cv2.imwrite(os.path.join(aug_masks_dir, f"{base_name}{aug_suffix}_mask.png"), aug_mask)
            total_augmented += 1
        
        print(f"Processed {image_file}: {augmentations_per_image + 1} total images")
    
    print(f"\nAugmentation complete!")
    print(f"Total images created: {total_augmented}")
    print(f"Output directory: {output_dir}")
    
    return output_dir

def visualize_augmentations(image_path, mask_path, output_path=None):
    """Visualize different augmentation techniques"""
    # Load original image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None or mask is None:
        print("Could not load image or mask")
        return
    
    # Create different augmentations
    augmentations = []
    
    # Original
    augmentations.append(("Original", image, mask))
    
    # Rotation
    rot_img, rot_mask = random_rotation_augmentation(image, mask, 15)
    augmentations.append(("Rotation", rot_img, rot_mask))
    
    # Horizontal flip
    hflip_img, hflip_mask = horizontal_flip(image, mask)
    augmentations.append(("Horizontal Flip", hflip_img, hflip_mask))
    
    # Brightness/contrast
    bc_img = brightness_contrast_augmentation(image)
    augmentations.append(("Brightness/Contrast", bc_img, mask))
    
    # Gaussian noise
    noise_img = gaussian_noise(image)
    augmentations.append(("Gaussian Noise", noise_img, mask))
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, (title, aug_img, aug_mask) in enumerate(augmentations):
        row = i // 3
        col = i % 3
        
        # Show image
        axes[row, col].imshow(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(title)
        axes[row, col].axis('off')
    
    # Remove empty subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Augmentation visualization saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    print("Enhanced Data Augmentation Pipeline")
    print("=" * 40)
    
    # Create augmented dataset from quadrant dataset
    if os.path.exists("quadrant_dataset/images"):
        print("Creating augmented dataset from quadrants...")
        create_augmented_dataset(
            "quadrant_dataset/images",
            "quadrant_dataset/masks",
            "augmented_dataset",
            augmentations_per_image=4
        )
    
    # Visualize augmentations on a sample
    if os.path.exists("quadrant_dataset/images/3-1-1-100x_tl.jpg"):
        print("\nCreating augmentation visualization...")
        visualize_augmentations(
            "quadrant_dataset/images/3-1-1-100x_tl.jpg",
            "quadrant_dataset/masks/3-1-1-100x_tl_mask.png",
            "output/augmentation_visualization.png"
        ) 