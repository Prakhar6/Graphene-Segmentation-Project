import os
from PIL import Image
import torchvision.transforms.functional as TF

image_dir = 'images'
mask_dir = 'masks'
out_img_dir = 'aug_images'
out_mask_dir = 'aug_masks'

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_mask_dir, exist_ok=True)

ops = {
    'original': lambda x: x,
    'hflip': TF.hflip,
    'vflip': TF.vflip,
    'rot90': lambda x: TF.rotate(x, 90),
}

for img_name in sorted(os.listdir(image_dir)):
    img_path = os.path.join(image_dir, img_name)

    base_name = os.path.splitext(img_name)[0]  # e.g., "3-1-1-100x"
    mask_name = base_name + '_mask.png'        # e.g., "3-1-1-100x_mask.png"
    mask_path = os.path.join(mask_dir, mask_name)

    if not os.path.exists(mask_path):
        print(f"Mask not found for {img_name}: expected {mask_path}")
        continue

    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path)

    for op_name, op in ops.items():
        img_aug = op(image)
        mask_aug = op(mask)

        img_aug.save(f"{out_img_dir}/{base_name}_{op_name}.png")
        mask_aug.save(f"{out_mask_dir}/{base_name}_{op_name}.png")

print("✅ Augmented dataset created.")
