#!/usr/bin/env python3
"""
Enhanced Data Augmentation Pipeline
Apply advanced augmentation techniques to expand the quadrant dataset
"""

import os
import cv2
import numpy as np
from pathlib import Path
import random
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A

def apply_elastic_deformation(image, mask, alpha=1, sigma=50, alpha_affine=50):
    """Apply elastic deformation to image and mask"""
    transform = A.ElasticTransform(
        alpha=alpha, 
        sigma=sigma, 
        alpha_affine=alpha_affine, 
        p=1.0
    )
    
    if len(image.shape) == 3:
        augmented = transform(image=image, mask=mask)
    else:
        augmented = transform(image=image, mask=mask)
    
    return augmented['image'], augmented['mask']

def apply_grid_distortion(image, mask, num_steps=5, distort_limit=0.3):
    """Apply grid distortion to image and mask"""
    transform = A.GridDistortion(
        num_steps=num_steps,
        distort_limit=distort_limit,
        p=1.0
    )
    
    if len(image.shape) == 3:
        augmented = transform(image=image, mask=mask)
    else:
        augmented = transform(image=image, mask=mask)
    
    return augmented['image'], augmented['mask']

def apply_color_augmentation(image):
    """Apply color-based augmentations"""
    # Convert to PIL Image for color operations
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    # Random brightness
    brightness_factor = random.uniform(0.8, 1.2)
    pil_image = ImageEnhance.Brightness(pil_image).enhance(brightness_factor)
    
    # Random contrast
    contrast_factor = random.uniform(0.8, 1.2)
    pil_image = ImageEnhance.Contrast(pil_image).enhance(contrast_factor)
    
    # Random saturation (if color image)
    if pil_image.mode == 'RGB':
        saturation_factor = random.uniform(0.8, 1.2)
        pil_image = ImageEnhance.Color(pil_image).enhance(saturation_factor)
    
    # Random sharpness
    sharpness_factor = random.uniform(0.8, 1.2)
    pil_image = ImageEnhance.Sharpness(pil_image).enhance(sharpness_factor)
    
    # Random blur
    if random.random() < 0.3:
        blur_radius = random.uniform(0.5, 2.0)
        pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return np.array(pil_image)

def apply_noise_augmentation(image):
    """Apply noise-based augmentations"""
    # Salt and pepper noise
    if random.random() < 0.3:
        noise = np.random.random(image.shape[:2])
        salt_mask = noise > 0.95
        pepper_mask = noise < 0.05
        
        if len(image.shape) == 3:
            image[salt_mask] = [255, 255, 255]
            image[pepper_mask] = [0, 0, 0]
        else:
            image[salt_mask] = 255
            image[pepper_mask] = 0
    
    # Gaussian noise
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(5, 15), image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return image

def apply_geometric_augmentation(image, mask):
    """Apply geometric augmentations"""
    # Random rotation
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        transform = A.Rotate(limit=angle, p=1.0)
        augmented = transform(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']
    
    # Random flip
    if random.random() < 0.5:
        transform = A.HorizontalFlip(p=1.0)
        augmented = transform(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']
    
    # Random crop and resize
    if random.random() < 0.3:
        h, w = image.shape[:2]
        crop_size = min(h, w)
        crop_size = int(crop_size * random.uniform(0.8, 0.95))
        
        transform = A.RandomCrop(
            height=crop_size, 
            width=crop_size, 
            p=1.0
        )
        augmented = transform(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']
        
        # Resize back to original size
        transform = A.Resize(height=h, width=w, p=1.0)
        augmented = transform(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']
    
    return image, mask

def augment_image_and_mask(image_path, mask_path, output_dir, base_name):
    """Apply all augmentation techniques to create multiple versions"""
    
    # Read image and mask
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
    if image is None or mask is None:
        print(f"Could not read image or mask: {image_path}")
        return 0
        
        # Save original
    orig_image_path = output_dir / "augmented_images" / f"{base_name}_original.jpg"
    orig_mask_path = output_dir / "augmented_masks" / f"{base_name}_original.png"
    
    cv2.imwrite(str(orig_image_path), image)
    cv2.imwrite(str(orig_mask_path), mask)
    
    count = 1  # Original image
    
    # Apply different augmentation combinations
    augmentations = [
        ("hflip", lambda img, msk: (cv2.flip(img, 1), cv2.flip(msk, 1))),
        ("vflip", lambda img, msk: (cv2.flip(img, 0), cv2.flip(msk, 0))),
        ("rot90", lambda img, msk: (cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(msk, cv2.ROTATE_90_CLOCKWISE))),
        ("elastic", lambda img, msk: apply_elastic_deformation(img, msk)),
        ("grid", lambda img, msk: apply_grid_distortion(img, msk)),
        ("color", lambda img, msk: (apply_color_augmentation(img), msk)),
        ("noise", lambda img, msk: (apply_noise_augmentation(img), msk)),
        ("geometric", lambda img, msk: apply_geometric_augmentation(img, msk))
    ]
    
    # Apply 4 augmentations per image
    selected_augs = random.sample(augmentations, 4)
    
    for aug_name, aug_func in selected_augs:
        try:
            aug_image, aug_mask = aug_func(image.copy(), mask.copy())
            
            # Save augmented image and mask
            aug_image_path = output_dir / "augmented_images" / f"{base_name}_{aug_name}.jpg"
            aug_mask_path = output_dir / "augmented_masks" / f"{base_name}_{aug_name}.png"
            
            cv2.imwrite(str(aug_image_path), aug_image)
            cv2.imwrite(str(aug_mask_path), aug_mask)
            
            count += 1
            
        except Exception as e:
            print(f"Error applying {aug_name} augmentation: {e}")
            continue
    
    return count

def main():
    """Main function to apply enhanced augmentation"""
    print("Enhanced Data Augmentation Pipeline")
    print("=" * 40)
    
    # Define paths
    project_root = Path(__file__).parent.parent
    quadrant_dir = project_root / "quadrant_dataset"
    output_dir = project_root / "data" / "augmented" / "augmented_dataset"
    
    # Create output directories
    (output_dir / "augmented_images").mkdir(parents=True, exist_ok=True)
    (output_dir / "augmented_masks").mkdir(parents=True, exist_ok=True)
    
    # Get quadrant images
    quadrant_images_dir = quadrant_dir / "images"
    quadrant_masks_dir = quadrant_dir / "masks"
    
    if not quadrant_images_dir.exists():
        print(f"Quadrant dataset not found: {quadrant_images_dir}")
        print("Please run create_quadrant_dataset.py first")
        return
    
    image_files = list(quadrant_images_dir.glob("*.jpg"))
    
    if not image_files:
        print(f"No quadrant images found in {quadrant_images_dir}")
        return
    
    print(f"Found {len(image_files)} quadrant images")
    print(f"Applying 4 augmentations per image...")
    
    total_images = 0
    
    for image_file in image_files:
        # Find corresponding mask
        mask_name = f"{image_file.stem}_mask.png"
        mask_file = quadrant_masks_dir / mask_name
        
        if mask_file.exists():
            base_name = image_file.stem
            count = augment_image_and_mask(image_file, mask_file, output_dir, base_name)
            total_images += count
            print(f"Processed {image_file.name}: {count} total images")
        else:
            print(f"Mask not found for {image_file.name}: {mask_file}")
    
    print(f"\nAugmentation complete!")
    print(f"Total images created: {total_images}")
    print(f"Output directory: {output_dir}")
    
    # Create augmentation visualization
    print("\nCreating augmentation visualization...")
    try:
        # Show example of augmented images
        sample_images = list((output_dir / "augmented_images").glob("*.jpg"))[:3]
        if sample_images:
            print(f"Sample augmented images: {[img.name for img in sample_images]}")
    except Exception as e:
        print(f"Could not load image or mask")

if __name__ == "__main__":
    main() 