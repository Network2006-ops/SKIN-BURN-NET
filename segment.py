# ================================================================
# SKIN LESION SEGMENTATION
# GRABCUT-DRIVEN
#
# INPUT:
#   AMB preprocessed skin images
#
# OUTPUT:
#   1. AMB Preprocessed Image
#   2. Segmented Lesion Mask
#
# REMOVED:
#   - Segmentation Overlay
#   - Green boundary
#   - Red highlighted lesion
#   - Infection percentage
#   - Detected Infection graph
# ================================================================


# ================================================================
# 1. IMPORT LIBRARIES
# ================================================================

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ================================================================
# 2. INPUT IMAGE PATHS
# ================================================================

image_paths = [
    "d:/Skin/skin_AMB_Preprocessed/unnamed (1).png",
    "d:/Skin/skin_AMB_Preprocessed/unnamed (2).png",
    "d:/Skin/skin_AMB_Preprocessed/unnamed (3).png",
    "d:/Skin/skin_AMB_Preprocessed/unnamed (4).png",
    "d:/Skin/skin_AMB_Preprocessed/unnamed (5).png",
    "d:/Skin/skin_AMB_Preprocessed/unnamed (6).png",
    "d:/Skin/skin_AMB_Preprocessed/unnamed (7).png",
    "d:/Skin/skin_AMB_Preprocessed/unnamed (8).png",
    "d:/Skin/skin_AMB_Preprocessed/unnamed (9).png"
]


# ================================================================
# 3. OUTPUT DIRECTORY
# ================================================================

OUTPUT_DIR = "d:/Skin/skin_GrabCut_Segmentation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# 4. REMOVE HAIR / DARK ARTIFACTS
# ================================================================

def remove_hair(bgr):

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    ks = max(
        9,
        int(round(min(gray.shape[:2]) * 0.015)) | 1
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (ks, ks)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    _, hair_mask = cv2.threshold(
        blackhat,
        10,
        255,
        cv2.THRESH_BINARY
    )

    cleaned = cv2.inpaint(
        bgr,
        hair_mask,
        3,
        cv2.INPAINT_TELEA
    )

    return cleaned


# ================================================================
# 5. FILL HOLES IN MASK
# ================================================================

def fill_holes(mask):

    h, w = mask.shape

    flood = mask.copy()

    flood_mask = np.zeros(
        (h + 2, w + 2),
        np.uint8
    )

    cv2.floodFill(
        flood,
        flood_mask,
        (0, 0),
        255
    )

    flood_inverse = cv2.bitwise_not(flood)

    filled = mask | flood_inverse

    return filled


# ================================================================
# 6. SMOOTH SEGMENTATION MASK
# ================================================================

def smooth_mask(mask):

    # Median filtering
    smoothed = cv2.medianBlur(
        mask,
        9
    )

    # Find contours
    contours, _ = cv2.findContours(
        smoothed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return smoothed

    # Select largest contour
    largest = max(
        contours,
        key=cv2.contourArea
    )

    # Contour approximation
    epsilon = (
        0.003 *
        cv2.arcLength(
            largest,
            True
        )
    )

    approx = cv2.approxPolyDP(
        largest,
        epsilon,
        True
    )

    # Create final mask
    output = np.zeros_like(smoothed)

    cv2.drawContours(
        output,
        [approx],
        -1,
        255,
        -1
    )

    return output


# ================================================================
# 7. GRABCUT SEGMENTATION
# ================================================================

def grabcut_rect(
    bgr,
    margin_frac=0.06
):

    h, w = bgr.shape[:2]

    # Rectangle margins
    margin_x = int(
        w * margin_frac
    )

    margin_y = int(
        h * margin_frac
    )

    rect = (
        margin_x,
        margin_y,
        w - 2 * margin_x,
        h - 2 * margin_y
    )

    # GrabCut mask
    gc_mask = np.zeros(
        (h, w),
        np.uint8
    )

    # Background model
    bgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    # Foreground model
    fgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    # Apply GrabCut
    cv2.grabCut(
        bgr,
        gc_mask,
        rect,
        bgd_model,
        fgd_model,
        8,
        cv2.GC_INIT_WITH_RECT
    )

    # Keep definite foreground
    # and probable foreground
    result = np.where(
        (gc_mask == cv2.GC_FGD) |
        (gc_mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    return result


# ================================================================
# 8. COLOR PRIOR SEGMENTATION
# ================================================================

def color_prior_mask(bgr):

    h, w = bgr.shape[:2]

    # ------------------------------------------------------------
    # BGR channels
    # ------------------------------------------------------------

    b, g, r = cv2.split(
        bgr.astype(np.float32)
    )

    # Excess red
    excess_red = (
        r -
        (g + b) / 2.0
    )

    # ------------------------------------------------------------
    # HSV
    # ------------------------------------------------------------

    hsv = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2HSV
    ).astype(np.float32)

    saturation = hsv[:, :, 1]

    value = hsv[:, :, 2]

    # ------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------

    features = np.stack(
        [
            excess_red.flatten(),
            saturation.flatten(),
            (255 - value).flatten()
        ],
        axis=1
    ).astype(np.float32)

    # ------------------------------------------------------------
    # Standardization
    # ------------------------------------------------------------

    features = (
        features -
        features.mean(axis=0)
    ) / (
        features.std(axis=0) +
        1e-6
    )

    # ------------------------------------------------------------
    # K-Means
    # ------------------------------------------------------------

    k = 3

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        20,
        0.5
    )

    _, labels, centers = cv2.kmeans(
        features,
        k,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS
    )

    # ------------------------------------------------------------
    # Select lesion cluster
    # ------------------------------------------------------------

    cluster_scores = centers.sum(
        axis=1
    )

    lesion_cluster = int(
        np.argmax(cluster_scores)
    )

    # ------------------------------------------------------------
    # Create binary mask
    # ------------------------------------------------------------

    mask = (
        (
            labels.flatten() ==
            lesion_cluster
        )
        .astype(np.uint8)
        .reshape(h, w)
        * 255
    )

    return mask


# ================================================================
# 9. COMBINE GRABCUT + COLOR PRIOR
# ================================================================

def segment_lesion(bgr):

    h, w = bgr.shape[:2]

    # ------------------------------------------------------------
    # Step 1: Remove hair
    # ------------------------------------------------------------

    clean = remove_hair(bgr)

    # ------------------------------------------------------------
    # Step 2: GrabCut
    # ------------------------------------------------------------

    gc_result = grabcut_rect(
        clean
    )

    # ------------------------------------------------------------
    # Step 3: Color prior
    # ------------------------------------------------------------

    prior = color_prior_mask(
        clean
    )

    # ------------------------------------------------------------
    # Step 4: Calculate areas internally
    # ------------------------------------------------------------

    gc_area = (
        np.sum(
            gc_result == 255
        )
        /
        (h * w)
    )

    prior_area = (
        np.sum(
            prior == 255
        )
        /
        (h * w)
    )

    # ------------------------------------------------------------
    # Step 5: Check whether GrabCut is degenerate
    # ------------------------------------------------------------

    degenerate = (
        gc_area < 0.02
        or
        gc_area > 0.98
    )

    # ------------------------------------------------------------
    # Step 6: Compare with color prior
    # ------------------------------------------------------------

    disagrees = (
        abs(
            gc_area -
            prior_area
        )
        > 0.30
    )

    # ------------------------------------------------------------
    # Step 7: Select segmentation
    # ------------------------------------------------------------

    if degenerate and disagrees:

        chosen = prior

    else:

        chosen = gc_result

    # ============================================================
    # MORPHOLOGICAL CLEANING
    # ============================================================

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    # Opening
    chosen = cv2.morphologyEx(
        chosen,
        cv2.MORPH_OPEN,
        kernel
    )

    # Closing
    chosen = cv2.morphologyEx(
        chosen,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Fill holes
    chosen = fill_holes(
        chosen
    )

    # ============================================================
    # CONTOUR FILTERING
    # ============================================================

    contours, _ = cv2.findContours(
        chosen,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        min_area = (
            0.002 *
            h *
            w
        )

        contours = [
            c
            for c in contours
            if cv2.contourArea(c)
            > min_area
        ]

    # ============================================================
    # KEEP LARGEST LESION REGION
    # ============================================================

    if contours:

        largest = max(
            contours,
            key=cv2.contourArea
        )

        final = np.zeros_like(
            chosen
        )

        cv2.drawContours(
            final,
            [largest],
            -1,
            255,
            -1
        )

    else:

        final = chosen

    # ============================================================
    # FINAL SMOOTHING
    # ============================================================

    final = smooth_mask(
        final
    )

    return final


# ================================================================
# 10. START PROCESSING
# ================================================================

print("=" * 70)
print("SKIN LESION SEGMENTATION")
print("GRABCUT-DRIVEN SEGMENTATION")
print("=" * 70)

print(
    "Number of images:",
    len(image_paths)
)

print(
    "Output directory:",
    OUTPUT_DIR
)

print("=" * 70)


# ================================================================
# 11. PROCESS EACH IMAGE
# ================================================================

processed_count = 0

for idx, path in enumerate(
    image_paths
):

    # ------------------------------------------------------------
    # Check file
    # ------------------------------------------------------------

    if not os.path.exists(path):

        print(
            "FILE NOT FOUND:",
            path
        )

        continue

    # ------------------------------------------------------------
    # Read image
    # ------------------------------------------------------------

    image_bgr = cv2.imread(
        path
    )

    if image_bgr is None:

        print(
            "ERROR READING:",
            path
        )

        continue

    # ------------------------------------------------------------
    # Convert BGR -> RGB
    # ------------------------------------------------------------

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    # ------------------------------------------------------------
    # Segment lesion
    # ------------------------------------------------------------

    mask = segment_lesion(
        image_bgr
    )

    # ------------------------------------------------------------
    # File name
    # ------------------------------------------------------------

    filename = os.path.basename(
        path
    )

    name = os.path.splitext(
        filename
    )[0]


    # ============================================================
    # 12. SAVE SEGMENTED MASK
    # ============================================================

    mask_path = os.path.join(
        OUTPUT_DIR,
        name +
        "_Segmented_Mask.png"
    )

    cv2.imwrite(
        mask_path,
        mask
    )


    # ============================================================
    # 13. DISPLAY ONLY TWO IMAGES
    # ============================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )


    # ------------------------------------------------------------
    # IMAGE 1
    # AMB PREPROCESSED IMAGE
    # ------------------------------------------------------------

    axes[0].imshow(
        image_rgb
    )

    axes[0].set_title(
        "AMB Preprocessed Image",
        fontsize=16,
        fontweight="bold"
    )

    axes[0].axis(
        "off"
    )


    # ------------------------------------------------------------
    # IMAGE 2
    # SEGMENTED LESION
    # ------------------------------------------------------------

    axes[1].imshow(
        mask,
        cmap="gray"
    )

    axes[1].set_title(
        "Segmented Lesion",
        fontsize=16,
        fontweight="bold"
    )

    axes[1].axis(
        "off"
    )


    # ============================================================
    # MAIN FIGURE TITLE
    # ============================================================

    fig.suptitle(
        f"Skin Lesion Segmentation - {filename}",
        fontsize=18,
        fontweight="bold"
    )


    # ============================================================
    # LAYOUT
    # ============================================================

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94
        ]
    )


    # ============================================================
    # SAVE COMPARISON FIGURE
    # ============================================================

    comparison_path = os.path.join(
        OUTPUT_DIR,
        name +
        "_Segmentation_Result.png"
    )

    plt.savefig(
        comparison_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )


    # ============================================================
    # DISPLAY
    # ============================================================

    plt.close()

    plt.close()


    # ============================================================
    # PROGRESS
    # ============================================================

    processed_count += 1

    print(
        f"[{idx + 1}/{len(image_paths)}] "
        f"Completed: {filename}"
    )

    print(
        "Segmented mask:",
        mask_path
    )

    print(
        "Comparison:",
        comparison_path
    )

    print("-" * 70)


# ================================================================
# 14. FINAL SUMMARY
# ================================================================

print()
print("=" * 70)
print("SEGMENTATION COMPLETED")
print("=" * 70)

print(
    "Successfully processed:",
    processed_count
)

print(
    "Total requested:",
    len(image_paths)
)

print()
print("Generated outputs:")
print("1. Segmented lesion masks")
print("2. Two-image comparison figures")

print()
print("NOT GENERATED:")
print("X Segmentation Overlay")
print("X Green boundary")
print("X Red highlighted overlay")
print("X Infection percentage")
print("X Detected Infection graph")

print()
print(
    "Saved in:",
    OUTPUT_DIR
)

print("=" * 70)