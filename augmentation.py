# ================================================================
# SKIN IMAGE AUGMENTATION (FOR 9 AMB IMAGES ONLY)
# ================================================================

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================================================================
# 1. YOUR AMB OUTPUT IMAGE PATHS
# ================================================================

image_paths = [
    r"d:\Skin\skin_AMB_Preprocessed\unnamed (1).png",
    r"d:\Skin\skin_AMB_Preprocessed\unnamed (2).png",
    r"d:\Skin\skin_AMB_Preprocessed\unnamed (3).png",
    r"d:\Skin\skin_AMB_Preprocessed\unnamed (4).png",
    r"d:\Skin\skin_AMB_Preprocessed\unnamed (5).png",
    r"d:\Skin\skin_AMB_Preprocessed\unnamed (6).png",
    r"d:\Skin\skin_AMB_Preprocessed\unnamed (7).png",
    r"d:\Skin\skin_AMB_Preprocessed\unnamed (8).png",
    r"d:\Skin\skin_AMB_Preprocessed\unnamed (9).png"
]

# ================================================================
# 2. AUGMENTATION FUNCTIONS (UNCHANGED)
# ================================================================

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_REFLECT
    )


def horizontal_flip(image):
    return cv2.flip(image, 1)


def vertical_flip(image):
    return cv2.flip(image, 0)


def zoom_image(image, zoom_factor=1.15):
    h, w = image.shape[:2]

    new_h = int(h / zoom_factor)
    new_w = int(w / zoom_factor)

    y1 = (h - new_h) // 2
    x1 = (w - new_w) // 2

    cropped = image[y1:y1 + new_h, x1:x1 + new_w]

    return cv2.resize(cropped, (w, h))


def brightness_change(image, factor):
    img = image.astype(np.float32) * factor
    return np.clip(img, 0, 255).astype(np.uint8)


def contrast_change(image, factor):
    img = image.astype(np.float32)
    mean = np.mean(img, axis=(0,1), keepdims=True)
    result = (img - mean) * factor + mean
    return np.clip(result, 0, 255).astype(np.uint8)


# ================================================================
# 3. CREATE 9 AUGMENTATIONS
# ================================================================

def create_augmentations(image):

    return [
        ("Original", image),
        ("Rotate_P15", rotate_image(image, 15)),
        ("Rotate_M15", rotate_image(image, -15)),
        ("Horizontal_Flip", horizontal_flip(image)),
        ("Vertical_Flip", vertical_flip(image)),
        ("Zoom", zoom_image(image, 1.15)),
        ("Brightness_High", brightness_change(image, 1.15)),
        ("Brightness_Low", brightness_change(image, 0.85)),
        ("Contrast", contrast_change(image, 1.15)),
    ]


# ================================================================
# 4. DISPLAY & SAVE 9 IMAGES WITH AUGMENTATION
# ================================================================

output_save_dir = r"d:\Skin\Augmented_AMB_Output"
import os
os.makedirs(output_save_dir, exist_ok=True)

print("\nDisplaying and saving augmentation for 9 AMB images...\n")

for idx, path in enumerate(image_paths):

    image_bgr = cv2.imread(path)

    if image_bgr is None:
        print(f"Error loading: {path}")
        continue

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    augmented_images = create_augmentations(image_rgb)

    plt.figure(figsize=(15, 15))

    for i, (name, aug_img) in enumerate(augmented_images):
        plt.subplot(3, 3, i+1)
        plt.imshow(aug_img)
        plt.title(name, fontsize=12, fontweight="bold")
        plt.axis("off")

        # Save individual augmented image file
        single_img_path = os.path.join(output_save_dir, f"unnamed_{idx+1}_{name}.png")
        cv2.imwrite(single_img_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

    img_name = os.path.basename(path)
    plt.suptitle(
        f"Image {idx+1} of 9 - {img_name}",
        fontsize=18,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig_path = os.path.join(output_save_dir, f"Augmented_AMB_Image_{idx+1}.png")
    plt.savefig(save_fig_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_fig_path}")
    plt.close()

print(f"\nAll 9 augmentation figures saved to: {output_save_dir}")