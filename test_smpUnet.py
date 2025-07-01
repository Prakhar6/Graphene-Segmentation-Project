import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import GrapheneSegmentationDataset
from transforms import get_basic_transform
import segmentation_models_pytorch as smp
from matplotlib.patches import Patch

PALETTE = {
    0: (0, 0, 255),      # Background - Blue
    1: (0, 255, 0),      # 1 layer - Green
    2: (255, 0, 0),      # 2+ layers - Red
}

def decode_segmap(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in PALETTE.items():
        rgb[mask == cls_idx] = color
    return rgb

def get_unet_model(num_classes):
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, classes=num_classes, activation=None)
    model.load_state_dict(torch.load("graphene_unet.pth", map_location='cpu'))
    model.eval()
    return model

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
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return ious

def pixel_accuracy(pred, target):
    return (pred == target).sum() / target.size

def test():
    test_img_dir = 'images'
    test_mask_dir = 'masks'
    output_dir = 'outputs_unet'
    os.makedirs(output_dir, exist_ok=True)

    test_dataset = GrapheneSegmentationDataset(test_img_dir, test_mask_dir, transform=get_basic_transform())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_unet_model(num_classes=3).to(device)

    all_ious = []
    all_accs = []

    with torch.no_grad():
        for i, (img, mask) in enumerate(test_loader):
            img = img.to(device)
            pred = model(img).argmax(dim=1).squeeze().cpu().numpy()
            mask = mask.squeeze().numpy()

            all_ious.append(compute_iou(pred, mask))
            all_accs.append(pixel_accuracy(pred, mask))

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            axs[0].imshow(img.squeeze().cpu().permute(1, 2, 0))
            axs[0].set_title("Input Image")

            axs[1].imshow(decode_segmap(mask))
            axs[1].set_title("Ground Truth")

            axs[2].imshow(decode_segmap(pred))
            axs[2].set_title("U-Net Prediction")

            for ax in axs:
                ax.axis("off")

            legend_patches = [
                Patch(color=np.array(PALETTE[2])/255.0, label='2+ layers (Red)'),
                Patch(color=np.array(PALETTE[1])/255.0, label='1 layer (Green)'),
                Patch(color=np.array(PALETTE[0])/255.0, label='Background (Blue)'),
            ]
            fig.legend(handles=legend_patches, loc='lower center', ncol=3)

            plt.tight_layout()
            plt.savefig(f"{output_dir}/unet_comparison_{i}.png", bbox_inches='tight')
            plt.close()

    mean_ious = np.nanmean(np.array(all_ious), axis=0)
    mean_acc = np.mean(all_accs)

    print("✅ U-Net predictions saved to 'outputs_unet/'")
    print(f"U-Net Mean IoU per class: Background: {mean_ious[0]:.4f}, 1 Layer: {mean_ious[1]:.4f}, 2+ Layers: {mean_ious[2]:.4f}")
    print(f"U-Net Mean Pixel Accuracy: {mean_acc:.4f}")

if __name__ == "__main__":
    test()
