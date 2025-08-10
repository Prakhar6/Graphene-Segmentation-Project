import os
import json
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_coco_annotations(json_path):
    """Load COCO format annotations"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def create_mask_from_annotations(image_info, annotations, image_size):
    """Create mask from COCO annotations"""
    mask = np.zeros(image_size, dtype=np.uint8)
    
    for ann in annotations:
        if ann['image_id'] == image_info['id']:
            category_id = ann['category_id']
            # Convert category_id to mask value (0=background, 1=1_layer, 2=2_layer, 3=3+_layer)
            mask_value = category_id - 1  # Adjust for 0-based indexing
            
            # Create polygon mask from segmentation
            if 'segmentation' in ann and len(ann['segmentation']) > 0:
                polygon = np.array(ann['segmentation'][0]).reshape(-1, 2)
                polygon = polygon.astype(np.int32)
                cv2.fillPoly(mask, [polygon], mask_value)
    
    return mask

def split_image_and_mask(image_path, mask, output_dir, base_name):
    """Split image and mask into 4 quadrants"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return []
    
    h, w = image.shape[:2]
    h_mid, w_mid = h // 2, w // 2
    
    # Define quadrants
    quadrants = [
        (0, 0, w_mid, h_mid),           # Top-left
        (w_mid, 0, w, h_mid),           # Top-right
        (0, h_mid, w_mid, h),           # Bottom-left
        (w_mid, h_mid, w, h)            # Bottom-right
    ]
    
    quadrant_names = ['tl', 'tr', 'bl', 'br']
    results = []
    
    for i, (x1, y1, x2, y2) in enumerate(quadrants):
        # Crop image
        img_crop = image[y1:y2, x1:x2]
        
        # Crop mask
        mask_crop = mask[y1:y2, x1:x2]
        
        # Save image
        img_filename = f"{base_name}_{quadrant_names[i]}.jpg"
        img_path_out = os.path.join(output_dir, 'images', img_filename)
        cv2.imwrite(img_path_out, img_crop)
        
        # Save mask
        mask_filename = f"{base_name}_{quadrant_names[i]}_mask.png"
        mask_path_out = os.path.join(output_dir, 'masks', mask_filename)
        cv2.imwrite(mask_path_out, mask_crop)
        
        results.append({
            'image': img_filename,
            'mask': mask_filename,
            'quadrant': quadrant_names[i]
        })
    
    return results

def create_quadrant_dataset():
    """Create quadrant dataset from original images and annotations"""
    
    # Create output directories
    output_dir = "quadrant_dataset"
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    
    # Load annotations
    coco_data = load_coco_annotations("annotations/coco.json")
    
    # Process each image
    all_quadrants = []
    
    for image_info in coco_data['images']:
        image_name = image_info['file_name']
        image_path = os.path.join("images", image_name)
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        # Get annotations for this image
        image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_info['id']]
        
        # Create mask
        mask = create_mask_from_annotations(image_info, image_annotations, (image_info['height'], image_info['width']))
        
        # Split into quadrants
        base_name = os.path.splitext(image_name)[0]
        quadrants = split_image_and_mask(image_path, mask, output_dir, base_name)
        
        all_quadrants.extend(quadrants)
        
        print(f"Processed {image_name} -> {len(quadrants)} quadrants")
    
    # Save dataset info
    dataset_info = {
        'total_quadrants': len(all_quadrants),
        'quadrants': all_quadrants,
        'classes': ['background', '1_layer', '2_layer', '3+_layer']
    }
    
    with open(os.path.join(output_dir, "dataset_info.json"), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\nDataset created successfully!")
    print(f"Total quadrants: {len(all_quadrants)}")
    print(f"Output directory: {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    create_quadrant_dataset() 