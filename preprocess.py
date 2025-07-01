import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def contrast_stretch(gray):
    """Stretch intensity range to 0–255 before CLAHE."""
    p2, p98 = np.percentile(gray, (2, 98))
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

def apply_clahe(image_bgr):
    # --- convert to LAB, enhance L channel ---
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l = contrast_stretch(l)                               # (optional but helps)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    cl   = clahe.apply(np.uint8(l))

    lab_enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)  # back to BGR

def preprocess_images(input_dir="images",
                      output_dir="clahe_images",
                      visualize=False):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(input_dir, fname)
            img      = cv2.imread(img_path)

            clahe_img = apply_clahe(img)
            cv2.imwrite(os.path.join(output_dir, fname), clahe_img)

            if visualize:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8,4))
                plt.subplot(1,2,1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB));    plt.title("Original"); plt.axis("off")
                plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB)); plt.title("CLAHE");    plt.axis("off")
                plt.suptitle(fname); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    preprocess_images(visualize=True)   # set False when batch‑processing
