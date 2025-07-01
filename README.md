# Graphene Layer Segmentation from Optical Microscope Images

This project uses semantic segmentation to detect and classify different graphene layer regions from optical microscope images using **DeepLabV3** (PyTorch). The model distinguishes background, monolayer graphene, and multilayer graphene areas pixel by pixel.
> We also performed testing using the SMP U-Net-based model (see results section) for comparison


---

## Problem Statement

Optical images of graphene flakes often contain overlapping regions with varying numbers of atomic layers. The goal is to:
- **Segment each flake region** from the image.
- **Classify the number of graphene layers** in each segmented area:
  - `0`: Background
  - `1`: 1 Layer
  - `2`: 2+ Layers

This allows researchers to quickly analyze high-resolution microscope images without manual inspection.

---


## Model Architecture: DeepLabV3

This project uses a **DeepLabV3-ResNet50** model from `torchvision.models.segmentation`:

- Pretrained on ImageNet
- Fine-tuned for 3-class segmentation
- Custom classifier head (2 conv layers)
- Trained on augmented microscope data

---

## Training Pipeline

### 1. Dataset & Mask Preparation

- Images are cropped from high-res 2048×1536 microscope scans.
- Masks are created using [MakeSense.ai](https://www.makesense.ai/) and exported as COCO format.
- Pixel labels:
  - `0` → background (blue)
  - `1` → 1 layer (green)
  - `2` → 2+ layers (red)
- 'generate_augmented_dataset.py' unpacks the COCO format to create the masks for pytorch in PNG format

### 2. Data Augmentation (`transforms.py`)
- Resizing (256x256)
- Random flips & rotations
- Converted into PyTorch tensors
- Total of 16 images created from the original dataset of 4

### 3. Model Training (`train_deeplabv3.py`)
- 20 epochs using CrossEntropyLoss
- Adam optimizer with lr=1e-4
- Final model saved to graphene_deeplabv3.pth


### 4. Testing & Evaluation (`test_deeplabv3.py`)
- Loads the saved DeepLabV3 model
- Generates predictions on the test set
- Saves side-by-side comparisons to outputs_deeplabv3/
- Computes metrics:
  -   Per-class IoU
  -   Pixel accuracy

## Results
| Model       | Background IoU | 1 Layer IoU | 2+ Layers IoU | Pixel Accuracy |
|-------------|----------------|-------------|----------------|----------------|
| **DeepLabV3** | **0.9636**     | **0.7915**   | **0.9741**      | **0.9812**     |
| U-Net       | 0.9507         | 0.3574      | 0.9597         | 0.9726         |
> DeepLabV3 achieved significantly better results on all classes, especially on the 1-layer class which is more subtle and harder to detect.

## Key Takeaways
- Even with limited image data, segmentation of graphene layers is possible using pretrained models and careful data augmentation.
- DeepLabV3 generalizes well to fine-grained visual differences in scientific images.
- Labeling with even ~4–5 high-quality annotated images, combined with flips/rotations, can yield strong performance.

## Acknowledgments 
- MakeSense.ai for fast online annotation
- DeepLabV3 by Google, available via torchvision.models.segmentation
- Research guidance and data from Dr. Shao’s Lab
