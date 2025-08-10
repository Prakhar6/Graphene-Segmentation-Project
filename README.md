# Graphene Layer Segmentation with 2DMOINet

This project implements a machine learning model for semantic segmentation of graphene flakes from optical microscope images, based on techniques from the 2020 Han et al. paper "Deep-Learning-Enabled Fast Optical Identification and Characterization of 2D Materials".

## Problem Statement

Graphene flakes exhibit different optical properties based on their layer thickness:
- **Background**: Substrate or empty areas
- **1 Layer**: Single-layer graphene (monolayer)
- **2 Layers**: Bilayer graphene
- **3+ Layers**: Trilayer or thicker graphene

The goal is to automatically segment these regions from optical microscope images to enable rapid characterization of graphene samples.

## Model Architecture

### V1: Baseline 2DMOINet
- **Architecture**: DeepLabV3+ with ResNet50 backbone
- **Classes**: 4 (background, 1 layer, 2 layers, 3+ layers)
- **Input Size**: 224x224 pixels
- **Loss Function**: CrossEntropyLoss with class weights

### V2: Enhanced 2DMOINet
- **Architecture**: Enhanced DeepLabV3+ with attention mechanisms
- **Advanced Features**:
  - Enhanced ASPP with attention modules
  - Improved decoder with skip connections
  - Multiple loss function options (Focal, Dice, Boundary, Combined)
  - Advanced data augmentation (elastic deformation, grid distortion)
  - Mixed precision training
  - Advanced learning rate scheduling
  - Gradient clipping and early stopping

## Project Structure

```
Graphene-Segmentation-Project/
├── models/
│   ├── v1_2dmoinet/           # Baseline implementation
│   │   ├── dataset.py         # Dataset loading
│   │   ├── transforms.py      # Basic transforms
│   │   ├── train_2dmoinet.py # Training script
│   │   └── test_2dmoinet.py  # Testing script
│   └── v2_enhanced/           # Enhanced implementation
│       ├── dataset.py         # Enhanced dataset
│       ├── transforms.py      # Advanced transforms
│       ├── losses.py          # Multiple loss functions
│       ├── model_architectures.py # Enhanced architectures
│       ├── train_2dmoinet_v2.py # Enhanced training
│       └── test_2dmoinet_v2.py  # Enhanced testing
├── utils/                     # Utility scripts
│   ├── create_quadrant_dataset.py
│   ├── enhanced_augmentation.py
│   └── enhanced_preprocess.py
├── scripts/                   # Pipeline runners
│   ├── run_pipeline.py       # V1 pipeline
│   └── run_v2_pipeline.py    # V2 pipeline
├── checkpoints/              # Model checkpoints
│   ├── v1/                  # V1 models
│   └── v2/                  # V2 models
├── results/                  # Evaluation results
│   ├── v1/                  # V1 results
│   └── v2/                  # V2 results
├── images/                   # Original images
├── masks/                    # Ground truth masks
├── aug_images/              # Augmented images
├── aug_masks/               # Augmented masks
└── annotations/             # COCO/VGG annotations
```

## Training Pipeline

### V1 Pipeline
```bash
python scripts/run_pipeline.py
```

### V2 Pipeline
```bash
python scripts/run_v2_pipeline.py
```

## Data Preparation

1. **Quadrant Splitting**: Original images are split into 4 quadrants (4 → 16 images)
2. **Enhanced Augmentation**: 4 augmentations per quadrant (16 → 64 total images)
3. **Augmentation Techniques**:
   - Horizontal/Vertical flips
   - 90° rotations
   - Elastic deformation
   - Grid distortion
   - Color augmentation
   - Noise injection

## Training Configuration

### V1 Training
- **Epochs**: 50
- **Batch Size**: 4
- **Learning Rate**: 0.001 with warmup and decay
- **Optimizer**: SGD with momentum
- **Loss**: CrossEntropyLoss

### V2 Training
- **Epochs**: 100
- **Batch Size**: 4
- **Learning Rate**: CosineAnnealingWarmRestarts
- **Optimizer**: AdamW
- **Loss**: Combined Loss (Focal + Dice + Boundary)
- **Advanced Features**: Mixed precision, gradient clipping, early stopping

## Evaluation Metrics

- **Mean IoU**: Intersection over Union across all classes
- **Pixel Accuracy**: Overall pixel-level accuracy
- **Per-class IoU**: Individual class performance
- **Confusion Matrix**: Detailed class-wise predictions
- **Dice Coefficient**: Boundary accuracy measure

## Results

### V1 Performance
- **Mean IoU**: ~0.75
- **Accuracy**: ~0.85
- **Training Time**: ~30 minutes

### V2 Performance (Expected Improvements)
- **Mean IoU**: Target >0.80
- **Accuracy**: Target >0.90
- **Training Time**: ~45 minutes (due to enhanced features)

## Key Features from Research Paper

1. **Multi-scale Processing**: ASPP module captures context at multiple scales
2. **Attention Mechanisms**: Focus on relevant image regions
3. **Advanced Augmentation**: Elastic deformation for realistic variations
4. **Loss Function Combination**: Multiple loss functions for better optimization
5. **Learning Rate Scheduling**: Adaptive learning rate for convergence

## Installation

```bash
# Clone repository
git clone <repository-url>
cd Graphene-Segmentation-Project

# Install dependencies
pip install -r requirements.txt

# Activate virtual environment (if using)
source .venv/Scripts/activate  # Windows
source .venv/bin/activate      # Linux/Mac
```

## Usage

### Quick Start
```bash
# Run V1 pipeline
python scripts/run_pipeline.py

# Run V2 pipeline
python scripts/run_v2_pipeline.py
```

### Individual Components
```bash
# Train V1 model
python models/v1_2dmoinet/train_2dmoinet.py

# Train V2 model
python models/v2_enhanced/train_2dmoinet_v2.py

# Test V1 model
python models/v1_2dmoinet/test_2dmoinet.py

# Test V2 model
python models/v2_enhanced/test_2dmoinet_v2.py
```

## Dependencies

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- OpenCV >= 4.5.0
- NumPy >= 1.21.0
- PIL (Pillow) >= 8.3.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0
- scikit-learn >= 1.0.0

## Model Versions

### Version 1 (Baseline)
- Basic DeepLabV3+ implementation
- Standard data augmentation
- CrossEntropyLoss
- Basic training pipeline

### Version 2 (Enhanced)
- Enhanced architecture with attention
- Advanced augmentation techniques
- Multiple loss function options
- Advanced training strategies
- Comprehensive evaluation

### Future Versions
- **V3**: Additional architectural improvements
- **V4**: Ensemble methods and advanced post-processing

## Key Takeaways

1. **Quadrant splitting** effectively expands the dataset from 4 to 64 images
2. **Advanced augmentation** improves model generalization
3. **Attention mechanisms** enhance feature extraction
4. **Loss function combination** improves optimization
5. **Mixed precision training** reduces memory usage and training time

## Acknowledgments 

This implementation is based on the research paper:
"Deep-Learning-Enabled Fast Optical Identification and Characterization of 2D Materials" by Han et al., Advanced Materials, 2020.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement improvements
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
