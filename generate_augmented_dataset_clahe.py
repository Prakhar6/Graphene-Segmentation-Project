import os
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import numpy as np
import random

# Set seed for reproducibility
random.seed(42)

# Paths
CLAHE_IMG_DIR = "clahe_images"
MASK_DIR = "masks"  # original masks
AUG_IMG_DIR = "aug_images_clahe"
AUG_MASK_DIR = "aug_masks_clahe"

os.makedirs(AUG_IMG_DIR, exist_ok=True)
os.makedirs(AUG_MASK_DIR, exist_ok=True)

# Augmentation options
def augment(image, mask):
    image = TF.resize(image, (256, 256))
    mask = TF.resize(mask, (256, 256), interpolation=Image.NEAREST)

    aug_type = random.choice(['original', 'hflip', 'vflip', 'rotate'])
    if aug_type == 'hflip':
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    elif aug_type == 'vflip':
        image = TF.vflip(image)
        mask = TF.vflip(mask)
    elif aug_type == 'rotate':
        angle = random.choice([90, 180, 270])
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)
    return image, mask

for filename in sorted(os.listdir(CLAHE_IMG_DIR)):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(CLAHE_IMG_DIR, filename)
        
        base_name = os.path.splitext(filename)[0]
        mask_name = base_name + "_mask.png"
        mask_path = os.path.join(MASK_DIR, mask_name)

        if not os.path.exists(mask_path):
            print(f"❌ Mask not found for {filename}: expected {mask_path}")
            continue

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Generate 4 augmented versions
        for i in range(4):
            aug_img, aug_mask = augment(image, mask)
            aug_img.save(os.path.join(AUG_IMG_DIR, f"{base_name}_aug{i}.png"))
            aug_mask.save(os.path.join(AUG_MASK_DIR, f"{base_name}_aug{i}.png"))

print("✅ Augmented CLAHE dataset created in 'aug_images_clahe/' and 'aug_masks_clahe/'")
