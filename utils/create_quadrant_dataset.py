#!/usr/bin/env python3
"""
Create Quadrant Dataset
Split original images and masks into 4 quadrants to expand the dataset
"""

import os
import cv2
import numpy as np
from pathlib import Path

def create_quadrants(image_path, mask_path, output_dir):
    """Create 4 quadrants from an image and its corresponding mask"""
    
    # Read image and mask
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    if image is None or mask is None:
        print(f"Could not read image or mask: {image_path}")
        return False
    
    # Get dimensions
    h, w = image.shape[:2]
    h_mid = h // 2
    w_mid = w // 2
    
    # Create quadrants
    quadrants = [
        ('tl', (0, 0, w_mid, h_mid)),           # Top-left
        ('tr', (w_mid, 0, w, h_mid)),           # Top-right
        ('bl', (0, h_mid, w_mid, h)),           # Bottom-left
        ('br', (w_mid, h_mid, w, h))            # Bottom-right
    ]
    
    base_name = image_path.stem
    
    for suffix, (x1, y1, x2, y2) in quadrants:
        # Extract quadrant from image
        quad_image = image[y1:y2, x1:x2]
        quad_mask = mask[y1:y2, x1:x2]
        
        # Save quadrant image
        image_filename = f"{base_name}_{suffix}.jpg"
        image_path_out = output_dir / "images" / image_filename
        cv2.imwrite(str(image_path_out), quad_image)
        
        # Save quadrant mask
        mask_filename = f"{base_name}_{suffix}_mask.png"
        mask_path_out = output_dir / "masks" / mask_filename
        cv2.imwrite(str(mask_path_out), quad_mask)
    
    return True

def main():
    """Main function to create quadrant dataset"""
    print("Creating Quadrant Dataset")
    print("=" * 40)
    
    # Define paths
    project_root = Path(__file__).parent.parent
    images_dir = project_root / "data" / "raw" / "images"
    masks_dir = project_root / "data" / "raw" / "masks"
    output_dir = project_root / "quadrant_dataset"
    
    # Create output directories
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    
    # Get image files
    image_files = list(images_dir.glob("*.jpg"))
    
    if not image_files:
        print(f"No image files found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Creating quadrants for each image...")
    
    successful = 0
    total = len(image_files)
    
    for image_file in image_files:
        # Find corresponding mask
        mask_name = f"{image_file.stem}_mask.png"
        mask_file = masks_dir / mask_name
        
        if mask_file.exists():
            if create_quadrants(image_file, mask_file, output_dir):
                successful += 1
                print(f"Created quadrants for {image_file.name}")
            else:
                print(f"Failed to create quadrants for {image_file.name}")
        else:
            print(f"Mask not found for {image_file.name}: {mask_file}")
    
    print(f"\nQuadrant dataset creation complete!")
    print(f"Successfully processed: {successful}/{total} images")
    print(f"Total quadrants created: {successful * 4}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main() 