import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import DataLoader
from dataset import GrapheneSegmentationDataset
from transforms import get_basic_transform
from train_deeplabv3 import DeepLabHead

def get_model(num_classes):
    model = deeplabv3_resnet50(weights=None, aux_loss=True)
    model.classifier = DeepLabHead(2048, num_classes)
    model.load_state_dict(torch.load("graphene_model.pth", map_location='cpu'))
    model.eval()
    return model

def decode_mask(mask):
    colors = {
        0: [0, 0, 255],     # Background - Blue
        1: [0, 255, 0],     # 1 Layer - Green
        2: [255, 0, 0]      # 2+ Layers - Red
    }
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in colors.items():
        rgb_mask[mask == class_id] = color
    return rgb_mask

def compute_iou(pred, target, num_classes=3):
    ious = []
    pred = pred.flatten()
    target = target.flatten()

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            ious.append(float('nan'))  # class not present
        else:
            ious.append(intersection / union)
    return ious

def pixel_accuracy(pred, target):
    return (pred == target).sum() / target.size

def test():
    test_img_dir = 'images'
    test_mask_dir = 'masks'
    output_dir = 'outputs_deeplabv3'
    os.makedirs(output_dir, exist_ok=True)

    test_dataset = GrapheneSegmentationDataset(test_img_dir, test_mask_dir, transform=get_basic_transform())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=3).to(device)

    all_ious = []
    all_accs = []

    with torch.no_grad():
        for i, (img, mask) in enumerate(test_loader):
            img = img.to(device)
            pred = model(img)['out'].argmax(dim=1).squeeze().cpu().numpy()
            mask = mask.squeeze().numpy()

            all_ious.append(compute_iou(pred, mask))
            all_accs.append(pixel_accuracy(pred, mask))

            pred_rgb = decode_mask(pred)

            fig, axs = plt.subplots(1, 3, figsize=(13, 4))
            axs[0].imshow(img.squeeze().cpu().permute(1, 2, 0))
            axs[0].set_title("Input Image")
            axs[1].imshow(mask, cmap='gray')
            axs[1].set_title("Ground Truth")
            axs[2].imshow(pred_rgb)
            axs[2].set_title("Prediction")

            red_patch = mpatches.Patch(color='red', label='2+ Layers')
            green_patch = mpatches.Patch(color='green', label='1 Layer')
            blue_patch = mpatches.Patch(color='blue', label='Background')
            axs[2].legend(handles=[red_patch, green_patch, blue_patch], loc='lower right')

            for ax in axs:
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(f"{output_dir}/comparison_{i}.png")
            plt.close()

    mean_ious = np.nanmean(np.array(all_ious), axis=0)
    mean_acc = np.mean(all_accs)

    print("✅ Predictions saved to 'outputs_deeplabv3/'")
    print(f"DeepLabV3 Mean IoU per class: Background: {mean_ious[0]:.4f}, 1 Layer: {mean_ious[1]:.4f}, 2+ Layers: {mean_ious[2]:.4f}")
    print(f"DeepLabV3 Mean Pixel Accuracy: {mean_acc:.4f}")

if __name__ == "__main__":
    test()
