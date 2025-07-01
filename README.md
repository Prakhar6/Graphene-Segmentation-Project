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
#### <ins> Model Architecture </ins>
This script trains a DeepLabV3 segmentation model on the preprocessed graphene dataset using PyTorch. The model used is a DeepLabV3 with a ResNet-50 backbone, pretrained on ImageNet:
```python
from torchvision.models.segmentation import deeplabv3_resnet50
model = deeplabv3_resnet50(pretrained=True)
```

To adapt this model to a 3-class problem (Background, 1 Layer, 2+ Layers), the classifier head is replaced with a custom head:
```python
class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
```
This adds two convolutional layers:
- A 3x3 conv followed by batch norm and ReLU
- A final 1x1 conv to produce the desired number of output classes (in this case, 3)

#### <ins> Dataset & Dataloader </ins>
Next, the training images and masks are loaded from:
```python
aug_images/   # Augmented training images
aug_masks/    # Corresponding training masks
```
The dataset uses a  resizing and tensor conversion transform:
```
from transforms import get_basic_transform
train_dataset = GrapheneSegmentationDataset(train_img_dir, train_mask_dir, transform=get_basic_transform())
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
```
Each image and mask is resized to 256x256 and converted to PyTorch tensors.

#### <ins> Loss, Optimizer, and Device </ins>

The model uses:
- CrossEntropyLoss() for multi-class pixel classification
- Adam optimizer with lr=1e-4
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

#### <ins> Training Loop </ins>

This model is then trained for 20 epochs over thr training dataset:
```python
for epoch in range(20):
    total_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        output = model(imgs)['out']  # Get raw logits from model
        loss = criterion(output, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}/20 - Loss: {total_loss:.4f}")
```
- The model outputs logits (output = model(imgs)['out'])
- Loss is computed pixel-wise between logits and true mask
- Backpropagation updates model weights
- Epoch loss is printed for monitoring

After training, the model’s weights are saved as `graphene_deeplabv3.pth`. You can reload it for testing/inference later using the same model architecture. A similar process is used to train the SMP Unet model as well.

### 4. Testing & Evaluation (`test_deeplabv3.py`)
After training, the model is evaluated on a set of unseen test images using pixel-level metrics and visualization.

#### <ins> Model Loading </ins>
The model is re-initialized with the same architecture used during training and the saved weights are loaded:
- weights=None disables loading of pretrained weights (we use our own).
- .eval() sets the model to inference mode (important for layers like BatchNorm and Dropout).

```python
model = deeplabv3_resnet50(weights=None, aux_loss=True)
model.classifier = DeepLabHead(2048, num_classes)
model.load_state_dict(torch.load("graphene_model.pth"))
model.eval()
```

#### <ins> Evaluation Dataset </ins>
A DataLoader is created with batch size = 1 (one image at a time for visualization and metric tracking):
```python
test_dataset = GrapheneSegmentationDataset(test_img_dir, test_mask_dir, transform=get_basic_transform())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
```

#### <ins> Prediction & Visualization </ins>
For each test image:
- The model produces class scores for each pixel.
- The class with the highest score is selected (argmax) to form a prediction mask.
- The predicted mask is decoded to RGB colors
- A comparison plot is saved to `outputs_deeplabv3/ ` showing:
  -  Input Image
  -  Ground Truth Mask (grayscale)
  -  Predicted RGB Mask (color-coded)
![alt text](https://github.com/Prakhar6/Graphene-Segmentation-Project/blob/main/outputs_deeplabv3/comparison_2.png?raw=true)


#### <ins> Evaluation Metric </ins>
Two important metrics are computed for each image:
1. Mean Intersection over Union (IoU)
Measures how well the predicted mask overlaps with the ground truth mask for each class:
- Calculated for each class (Background, 1 Layer, 2+ Layers)
- Skips classes not present in the ground truth
- The value ranges from 1.0 to 0.0 (perfect overlap to no overlap).
$$
\text{IoU} = \frac{\text{Intersection}}{\text{Union}} = \frac{TP}{TP + FP + FN}
$$

2. Pixel Accuracy
Measures how many pixels were classified correctly out of all pixels:
$$
\text{Pixel Accuracy} = \frac{\text{Number of Correctly Classified Pixels}}{\text{Total Number of Pixels}}
$$

#### <ins> Final Results </ins>
After evaluating all test images, the script computes the mean IoU per class and average pixel accuracy. This gives a quantitative measure of how well the model performs across each segmentation class.
```python
mean_ious = np.nanmean(np.array(all_ious), axis=0)
mean_acc = np.mean(all_accs)

print(f"DeepLabV3 Mean IoU per class: Background: {mean_ious[0]:.4f}, 1 Layer: {mean_ious[1]:.4f}, 2+ Layers: {mean_ious[2]:.4f}")
print(f"DeepLabV3 Mean Pixel Accuracy: {mean_acc:.4f}")
```




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
- [MakeSense.ai](https://www.makesense.ai/) for fast online annotation
- DeepLabV3 by Google, available via torchvision.models.segmentation
- Research guidance and data from Dr. Yinming Shao’s Lab
