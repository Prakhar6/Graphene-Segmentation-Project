import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import numpy as np
import torch
import cv2

class GrapheneSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, num_classes=4):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.num_classes = num_classes
        
        # Get sorted lists of images and masks
        self.images = sorted([f for f in os.listdir(image_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.masks = sorted([f for f in os.listdir(mask_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Verify matching pairs
        self._verify_pairs()
        
        print(f"Dataset initialized with {len(self.images)} image-mask pairs")
        print(f"Number of classes: {self.num_classes}")

    def _verify_pairs(self):
        """Verify that image and mask files match"""
        if len(self.images) != len(self.masks):
            print(f"Warning: Number of images ({len(self.images)}) != number of masks ({len(self.masks)})")
        
        # Check for matching pairs
        for img_file in self.images:
            base_name = os.path.splitext(img_file)[0]
            mask_file = f"{base_name}_mask.png"
            if mask_file not in self.masks:
                print(f"Warning: No matching mask found for {img_file}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale
        
        # Convert mask to class indices (0-3 for 4 classes)
        # Assuming mask values: 0=background, 1=1_layer, 2=2_layer, 3=3+_layer
        mask = np.array(mask)
        mask = mask.astype(np.uint8)
        
        # Ensure mask values are in valid range
        mask = np.clip(mask, 0, self.num_classes - 1)
        
        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            # Convert to tensors if no transform
            image = TF.to_tensor(image)
            mask = torch.from_numpy(mask).long()

        return image, mask

    def get_class_weights(self):
        """Calculate class weights for handling class imbalance"""
        class_counts = np.zeros(self.num_classes)
        
        for idx in range(len(self)):
            _, mask = self.__getitem__(idx)
            mask_np = mask.numpy()
            
            for class_id in range(self.num_classes):
                class_counts[class_id] += np.sum(mask_np == class_id)
        
        # Calculate weights (inverse frequency)
        total_pixels = np.sum(class_counts)
        class_weights = total_pixels / (self.num_classes * class_counts + 1e-8)
        
        # Normalize weights
        class_weights = class_weights / np.sum(class_weights)
        
        print(f"Class counts: {class_counts}")
        print(f"Class weights: {class_weights}")
        
        return torch.FloatTensor(class_weights)

class PreprocessedGrapheneDataset(GrapheneSegmentationDataset):
    """Dataset class for preprocessed images (224x224, normalized)"""
    
    def __init__(self, image_dir, mask_dir, transform=None, num_classes=4):
        super().__init__(image_dir, mask_dir, transform, num_classes)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Load preprocessed image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # Convert mask to class indices
        mask = np.array(mask)
        mask = mask.astype(np.uint8)
        mask = np.clip(mask, 0, self.num_classes - 1)
        
        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            # Convert to tensors if no transform
            image = TF.to_tensor(image)
            mask = torch.from_numpy(mask).long()

        return image, mask
