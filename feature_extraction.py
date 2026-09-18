import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_DIR = "d:/Skin/skin_output"
OUTPUT_DIR = "d:/Skin/feature_extraction_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
image_files = sorted([f for f in os.listdir(INPUT_DIR) if "_lesion.png" in f])

print(f"Found {len(image_files)} lesion files in {INPUT_DIR}")

def crop_to_content(gray, thresh=10):
    """Remove black padding around the lesion."""
    mask = gray > thresh
    if not mask.any():
        return gray
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    return gray[y0:y1+1, x0:x1+1]

def make_blob_feature_image(gray, out_size=300, blob_thresh=0.35, blur_bg=True):
    """
    Turn a grayscale patch into a 'feature highlight' image:
    - salient/edge regions get colored (viridis)
    - flat/background regions stay dark purple
    This is what gives the blob-in-cell look.
    """
    gray = crop_to_content(gray)
    gray = cv2.resize(gray, (out_size, out_size))
    gray_f = gray.astype(np.float32)

    # --- "saliency" = local contrast (blob-of-Gaussian style) ---
    blur1 = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=2)
    blur2 = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=6)
    dog = np.abs(blur1 - blur2)                     # difference of Gaussians
    grad = cv2.Laplacian(gray_f, cv2.CV_32F, ksize=3)
    saliency = dog + 0.5 * np.abs(grad)

    # normalize 0..1
    saliency -= saliency.min()
    if saliency.max() > 0:
        saliency /= saliency.max()

    # threshold: only strong regions become "blobs", rest forced to background
    feature = np.where(saliency > blob_thresh, saliency, 0.0)

    # optional: soften blob edges a touch
    if blur_bg:
        feature = cv2.GaussianBlur(feature, (0, 0), sigmaX=1.2)

    return feature  # values 0..1

def draw_grid(ax, size, grid_size, color='white', lw=0.8):
    step = size / grid_size
    for k in range(grid_size + 1):
        ax.axhline(k * step, color=color, lw=lw)
        ax.axvline(k * step, color=color, lw=lw)

for file in image_files:
    path = os.path.join(INPUT_DIR, file)
    bgr = cv2.imread(path)
    if bgr is None:
        continue
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    out_size = 300
    grid_size = 10
    feature = make_blob_feature_image(gray, out_size=out_size)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(feature, cmap='viridis', vmin=0, vmax=1, extent=[0, out_size, out_size, 0])
    draw_grid(ax, out_size, grid_size, color='white', lw=0.8)
    ax.set_xlim(0, out_size)
    ax.set_ylim(out_size, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Extracted Features - {file}")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, file.replace(".png", "_features.png"))
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

print(f"\nAll feature extraction plots saved to: {OUTPUT_DIR}")