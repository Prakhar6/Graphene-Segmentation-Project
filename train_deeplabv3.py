import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import DataLoader
from dataset import GrapheneSegmentationDataset
from transforms import get_basic_transform

# Custom DeepLab head
class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

def get_model(num_classes):
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier = DeepLabHead(2048, num_classes)
    return model

def train():
    train_img_dir = 'aug_images'
    train_mask_dir = 'aug_masks'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    train_dataset = GrapheneSegmentationDataset(train_img_dir, train_mask_dir, transform=get_basic_transform())
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # Model
    model = get_model(num_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(20):
        total_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            output = model(imgs)['out']
            loss = criterion(output, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/20 - Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "graphene_model.pth")
    print("✅ Model saved as graphene_model.pth")

if __name__ == "__main__":
    train()
