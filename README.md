# 🧪 Graphene Layer Segmentation from Optical Microscope Images

This project uses semantic segmentation to detect and classify different graphene layer regions from optical microscope images using **DeepLabV3** (PyTorch). The model distinguishes background, monolayer graphene, and multilayer graphene areas pixel by pixel.
> We also performed testing using the SMP U-Net-based model (see results section) for comparison


---

## 🔬 Problem Statement

Optical images of graphene flakes often contain overlapping regions with varying numbers of atomic layers. The goal is to:
- **Segment each flake region** from the image.
- **Classify the number of graphene layers** in each segmented area:
  - `0`: Background
  - `1`: 1 Layer
  - `2`: 2+ Layers

This allows researchers to quickly analyze high-resolution microscope images without manual inspection.

---


## 🧠 Model Architecture: DeepLabV3

This project uses a **DeepLabV3-ResNet50** model from `torchvision.models.segmentation`:

- Pretrained on ImageNet
- Fine-tuned for 3-class segmentation
- Custom classifier head (2 conv layers)
- Trained on augmented microscope data

---

## 🔁 Training Pipeline

### ✅ 1. Dataset & Mask Preparation

- Images are cropped from high-res 2048×1536 microscope scans.
- Masks are created using [MakeSense.ai](https://www.makesense.ai/) and exported as PNG or COCO format.
- Pixel labels:
  - `0` → background (blue)
  - `1` → 1 layer (green)
  - `2` → 2+ layers (red)

### ✅ 2. Data Augmentation

Implemented in `transforms.py`:
- Resizing (256x256)
- Random flips & rotations
- Converted into PyTorch tensors
