import os
import cv2
import numpy as np
import json
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

IMAGE_DIR = "images"
ANNOTATION_FILE = "annotations/coco.json"
OUTPUT_MASK_DIR = "masks"
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

coco = COCO(ANNOTATION_FILE)

category_mapping = {cat['id']: idx for idx, cat in enumerate(coco.loadCats(coco.getCatIds()))}
print("Class Mapping:", category_mapping)

for image_info in coco.loadImgs(coco.getImgIds()):
    image_id = image_info['id']
    filename = image_info['file_name']
    height, width = image_info['height'], image_info['width']

    mask = np.zeros((height, width), dtype=np.uint8)

    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        cat_id = ann['category_id']
        class_idx = category_mapping[cat_id]  # start at 0
        rle = coco.annToRLE(ann)
        binary_mask = coco_mask.decode(rle)
        mask[binary_mask == 1] = class_idx

    # Save mask
    mask_path = os.path.join(OUTPUT_MASK_DIR, os.path.splitext(filename)[0] + "_mask.png")
    cv2.imwrite(mask_path, mask)
    print(f"Saved: {mask_path}")
