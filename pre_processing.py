# ================================================================
# SKIN IMAGE AMB PREPROCESSING (DIRECT 9 IMAGES)
# ================================================================

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================================================================
# 1. PATH CONFIGURATION
# ================================================================

output_dir = r"d:\Skin\skin_AMB_Preprocessed"
os.makedirs(output_dir, exist_ok=True)

image_paths = [
    r"d:\Skin\skin\unnamed (1).png",
    r"d:\Skin\skin\unnamed (2).png",
    r"d:\Skin\skin\unnamed (3).png",
    r"d:\Skin\skin\unnamed (4).png",
    r"d:\Skin\skin\unnamed (5).png",
    r"d:\Skin\skin\unnamed (6).png",
    r"d:\Skin\skin\unnamed (7).png",
    r"d:\Skin\skin\unnamed (8).png",
    r"d:\Skin\skin\unnamed (9).png"
]

# ================================================================
# 2. ANISOTROPIC DIFFUSION
# ================================================================

def anisotropic_diffusion(image, iterations=10, kappa=30, gamma=0.15):
    img = image.astype(np.float32) / 255.0

    for _ in range(iterations):
        north = np.zeros_like(img)
        south = np.zeros_like(img)
        east  = np.zeros_like(img)
        west  = np.zeros_like(img)

        north[1:] = img[1:] - img[:-1]
        south[:-1] = img[:-1] - img[1:]
        east[:, :-1] = img[:, :-1] - img[:, 1:]
        west[:, 1:] = img[:, 1:] - img[:, :-1]

        c_n = np.exp(-(north / kappa) ** 2)
        c_s = np.exp(-(south / kappa) ** 2)
        c_e = np.exp(-(east / kappa) ** 2)
        c_w = np.exp(-(west / kappa) ** 2)

        img += gamma * (c_n * north + c_s * south + c_e * east + c_w * west)
        img = np.clip(img, 0, 1)

    return (img * 255).astype(np.uint8)

# ================================================================
# 3. AMB FILTER (ANISOTROPIC + MEDIAN + BILATERAL)
# ================================================================

def anisotropic_median_bilateral(image):
    anisotropic = anisotropic_diffusion(
        image,
        iterations=10,
        kappa=0.08,
        gamma=0.12
    )

    median = cv2.medianBlur(anisotropic, 3)

    bilateral = cv2.bilateralFilter(
        median,
        d=7,
        sigmaColor=50,
        sigmaSpace=50
    )

    return bilateral

# ================================================================
# 4. PROCESS & SAVE 9 IMAGES
# ================================================================

print("=" * 70)
print("AMB PREPROCESSING PIPELINE (9 IMAGES)")
print("=" * 70)
print(f"Output folder: {output_dir}")
print("=" * 70)

for i, path in enumerate(image_paths, 1):
    image_bgr = cv2.imread(path)

    if image_bgr is None:
        print(f"[ERROR] Could not read image: {path}")
        continue

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    processed = anisotropic_median_bilateral(image_rgb)

    # 1. Save preprocessed PNG image file
    out_img_path = os.path.join(output_dir, f"unnamed ({i}).png")
    cv2.imwrite(out_img_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))

    # 2. Save comparison plot
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title(f"Original: unnamed ({i}).png", fontsize=11, fontweight="bold")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(processed)
    plt.title("AMB Filter Output", fontsize=11, fontweight="bold")
    plt.axis("off")

    plt.suptitle(f"Image {i} of 9 - AMB Preprocessing", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_fig_path = os.path.join(output_dir, f"AMB_Comparison_{i}.png")
    plt.savefig(out_fig_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Processed [{i}/9] -> Saved image: {os.path.basename(out_img_path)} and plot: {os.path.basename(out_fig_path)}")

print("=" * 70)
print(f"[DONE] All 9 AMB images and comparison plots saved to: {output_dir}")
print("=" * 70)
