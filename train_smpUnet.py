import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim
from dataset import GrapheneSegmentationDataset
from transforms import get_basic_transform
from tqdm import tqdm

# --- Configuration ---
image_dir = "aug_images"
mask_dir = "aug_masks"
batch_size = 2
num_classes = 3
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset ---
train_dataset = GrapheneSegmentationDataset(image_dir, mask_dir, transform=get_basic_transform())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# --- Model: U-Net with ResNet34 backbone ---
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=num_classes, activation=None)
model.to(device)

# --- Loss and optimizer ---
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --- Training ---
model.train()
for epoch in range(epochs):
    total_loss = 0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "graphene_unet.pth")
print("✅ U-Net model saved.")
