import torch
import torchvision.transforms as TF
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import random
import math

class AdvancedTransform:
    """
    Advanced transform class for v2 with sophisticated augmentation techniques
    including elastic deformations, grid distortions, and advanced color augmentations.
    """
    
    def __init__(self, img_size=224, p=0.5):
        self.img_size = img_size
        self.p = p
        
    def __call__(self, image, mask=None):
        # Convert to PIL if needed
        if isinstance(image, torch.Tensor):
            image = F.to_pil_image(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if mask is not None and isinstance(mask, torch.Tensor):
            mask = F.to_pil_image(mask)
        elif mask is not None and isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask.astype(np.uint8))
        
        # Apply advanced augmentations
        if random.random() < self.p:
            image, mask = self._elastic_deformation(image, mask)
        
        if random.random() < self.p:
            image, mask = self._grid_distortion(image, mask)
        
        if random.random() < self.p:
            image = self._advanced_color_augmentation(image)
        
        if random.random() < self.p:
            image = self._advanced_noise_augmentation(image)
        
        # Standard augmentations
        if random.random() < self.p:
            image, mask = self._random_rotation(image, mask, angle_range=45)
        
        if random.random() < self.p:
            image, mask = self._random_flip(image, mask)
        
        if random.random() < self.p:
            image, mask = self._random_crop_and_resize(image, mask)
        
        # Resize to target size
        image = F.resize(image, (self.img_size, self.img_size), antialias=True)
        if mask is not None:
            mask = F.resize(mask, (self.img_size, self.img_size), antialias=False)
        
        # Convert to tensors
        image = F.to_tensor(image)
        if mask is not None:
            mask = torch.tensor(np.array(mask), dtype=torch.long)
        
        return image, mask
    
    def _elastic_deformation(self, image, mask=None):
        """Apply elastic deformation to image and mask."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        if mask is not None and isinstance(mask, Image.Image):
            mask = np.array(mask)
        
        # Elastic deformation parameters
        alpha = random.uniform(0.5, 2.0)
        sigma = random.uniform(0.5, 1.5)
        
        # Create displacement fields
        h, w = image.shape[:2]
        dx = np.random.randn(h, w) * alpha
        dy = np.random.randn(h, w) * alpha
        
        # Smooth displacement fields
        dx = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma)
        
        # Apply displacement
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x = np.clip(x + dx, 0, w - 1).astype(np.int32)
        y = np.clip(y + dy, 0, h - 1).astype(np.int32)
        
        # Ensure indices are within bounds
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        
        # Warp image and mask
        if len(image.shape) == 3:
            image = image[y, x]
        else:
            image = image[y, x, :]
        
        if mask is not None:
            mask = mask[y, x]
        
        # Convert back to PIL
        image = Image.fromarray(image)
        if mask is not None:
            mask = Image.fromarray(mask)
        
        return image, mask
    
    def _grid_distortion(self, image, mask=None):
        """Apply grid distortion to image and mask."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        if mask is not None and isinstance(mask, Image.Image):
            mask = np.array(mask)
        
        h, w = image.shape[:2]
        
        # Grid distortion parameters
        num_steps = random.randint(5, 10)
        xsteps = [1 + random.uniform(-0.1, 0.1) for _ in range(num_steps)]
        ysteps = [1 + random.uniform(-0.1, 0.1) for _ in range(num_steps)]
        
        # Apply grid distortion
        x_step = w // num_steps
        y_step = h // num_steps
        
        for i in range(num_steps):
            for j in range(num_steps):
                x1 = i * x_step
                y1 = j * y_step
                x2 = min((i + 1) * x_step, w)
                y2 = min((j + 1) * y_step, h)
                
                # Distort this grid cell
                if len(image.shape) == 3:
                    cell = image[y1:y2, x1:x2]
                else:
                    cell = image[y1:y2, x1:x2, :]
                
                if cell.size > 0:
                    # Resize cell with distortion
                    new_w = int(cell.shape[1] * xsteps[i])
                    new_h = int(cell.shape[0] * ysteps[j])
                    if new_w > 0 and new_h > 0:
                        cell = cv2.resize(cell, (new_w, new_h))
                        
                        # Place back in image - ensure dimensions match
                        actual_h = min(new_h, y2-y1)
                        actual_w = min(new_w, x2-x1)
                        
                        if len(image.shape) == 3:
                            if actual_h > 0 and actual_w > 0:
                                image[y1:y1+actual_h, x1:x1+actual_w] = cell[:actual_h, :actual_w]
                        else:
                            if actual_h > 0 and actual_w > 0:
                                image[y1:y1+actual_h, x1:x1+actual_w, :] = cell[:actual_h, :actual_w]
        
        # Convert back to PIL
        image = Image.fromarray(image)
        if mask is not None:
            mask = Image.fromarray(mask)
        
        return image, mask
    
    def _advanced_color_augmentation(self, image):
        """Apply advanced color augmentations."""
        # Brightness
        if random.random() < 0.5:
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Brightness(image).enhance(factor)
        
        # Contrast
        if random.random() < 0.5:
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Contrast(image).enhance(factor)
        
        # Saturation
        if random.random() < 0.5:
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Color(image).enhance(factor)
        
        # Sharpness
        if random.random() < 0.3:
            factor = random.uniform(0.5, 1.5)
            image = ImageEnhance.Sharpness(image).enhance(factor)
        
        # Gaussian blur
        if random.random() < 0.2:
            radius = random.uniform(0.5, 1.5)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        
        return image
    
    def _advanced_noise_augmentation(self, image):
        """Apply advanced noise augmentations."""
        if random.random() < 0.3:
            # Salt and pepper noise
            image = np.array(image)
            h, w = image.shape[:2]
            num_salt = np.ceil(0.01 * h * w)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            image[coords[0], coords[1]] = 255
            
            num_pepper = np.ceil(0.01 * h * w)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            image[coords[0], coords[1]] = 0
            
            image = Image.fromarray(image)
        
        if random.random() < 0.3:
            # Gaussian noise
            image = np.array(image)
            noise = np.random.normal(0, random.uniform(5, 15), image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        return image
    
    def _random_rotation(self, image, mask=None, angle_range=30):
        """Apply random rotation to image and mask."""
        angle = random.uniform(-angle_range, angle_range)
        
        # Ensure image and mask are PIL Images
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if mask is not None and isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask.astype(np.uint8))
        
        image = F.rotate(image, angle, fill=0)
        if mask is not None:
            mask = F.rotate(mask, angle, fill=0)
        
        return image, mask

    def _random_flip(self, image, mask=None):
        """Apply random horizontal and vertical flips."""
        # Ensure image and mask are PIL Images
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if mask is not None and isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask.astype(np.uint8))
        
        if random.random() < 0.5:
            image = F.hflip(image)
            if mask is not None:
                mask = F.hflip(mask)
        
        if random.random() < 0.5:
            image = F.vflip(image)
            if mask is not None:
                mask = F.vflip(mask)
        
        return image, mask
    
    def _random_crop_and_resize(self, image, mask=None):
        """Apply random crop and resize."""
        # Ensure image and mask are PIL Images
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if mask is not None and isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask.astype(np.uint8))
        
        # Random crop ratio
        crop_ratio = random.uniform(0.8, 1.0)
        
        w, h = image.size
        crop_w = int(w * crop_ratio)
        crop_h = int(h * crop_ratio)
        
        # Random crop position
        left = random.randint(0, w - crop_w)
        top = random.randint(0, h - crop_h)
        right = left + crop_w
        bottom = top + crop_h
        
        # Crop
        image = F.crop(image, top, left, crop_h, crop_w)
        if mask is not None:
            mask = F.crop(mask, top, left, crop_h, crop_w)
        
        # Resize back to original size
        image = F.resize(image, (h, w), antialias=True)
        if mask is not None:
            mask = F.resize(mask, (h, w), antialias=False)
        
        return image, mask


class BasicTransform:
    """
    Basic transform for validation/testing - no augmentation, just resizing and tensor conversion.
    """
    
    def __init__(self, img_size=224):
        self.img_size = img_size
        
    def __call__(self, image, mask=None):
        # Convert to PIL if needed
        if isinstance(image, torch.Tensor):
            image = F.to_pil_image(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if mask is not None and isinstance(mask, torch.Tensor):
            mask = F.to_pil_image(mask)
        
        # Ensure mask is PIL Image for resize
        if mask is not None and isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask.astype(np.uint8))
        
        # Resize
        image = F.resize(image, (self.img_size, self.img_size), antialias=True)
        if mask is not None:
            mask = F.resize(mask, (self.img_size, self.img_size), antialias=False)
        
        # Convert to tensors
        image = F.to_tensor(image)
        if mask is not None:
            mask = torch.tensor(np.array(mask), dtype=torch.long)
        
        return image, mask
