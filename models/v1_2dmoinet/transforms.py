import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import torch
import random

class BasicTransform:
    def __init__(self, size=(256, 256)):
        self.size = size
    
    def __call__(self, image, mask):
        # Convert mask back to PIL Image for resizing
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask.astype(np.uint8))
        
        image = TF.resize(image, self.size)
        mask = TF.resize(mask, self.size, interpolation=Image.NEAREST)
        image = TF.to_tensor(image)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image, mask

class AugmentedTransform:
    def __init__(self, size=(256, 256)):
        self.size = size
    
    def __call__(self, image, mask):
        # Convert mask back to PIL Image for resizing
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask.astype(np.uint8))
        
        image = TF.resize(image, self.size)
        mask = TF.resize(mask, self.size, interpolation=Image.NEAREST)

        # Data augmentation (4 variations per image)
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

        image = TF.to_tensor(image)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image, mask

def get_basic_transform():
    return BasicTransform()

def get_augmented_transform():
    return AugmentedTransform()
