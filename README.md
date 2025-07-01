# 🧪 Graphene Layer Segmentation from Optical Microscope Images

This project uses semantic segmentation to detect and classify different graphene layer regions from optical microscope images using DeepLabV3 and SMP Unet (PyTorch). The model distinguishes background, monolayer graphene, and multilayer graphene areas pixel by pixel.

# Problem Statement 

Optical images of graphene flakes often contain overlapping regions with varying numbers of atomic layers. The goal is to:

Segment each flake region from the image.

Classify the number of graphene layers in each segmented area:

0: Background

1: 1 Layer

2: 2+ Layers

This allows researchers to quickly analyze high-resolution microscope images without manual inspection.
