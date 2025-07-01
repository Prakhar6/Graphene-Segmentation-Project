import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import torch
import random

def get_augmented_transform():
    def transform(image, mask):
        image = TF.resize(image, (256, 256))
        mask = TF.resize(mask, (256, 256), interpolation=Image.NEAREST)

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
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Image normailization 
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image, mask
    return transform

def get_basic_transform():
    def transform(image, mask):
        image = TF.resize(image, (256, 256))
        mask = TF.resize(mask, (256, 256), interpolation=Image.NEAREST)
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Image normailization 
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image, mask
    return transform
