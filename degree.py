import os
import cv2
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

INPUT_DIR  = "d:/Skin/selected_features"
OUTPUT_DIR = "d:/Skin/classification_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_files = sorted([f for f in os.listdir(INPUT_DIR) if "_selected.png" in f])

# ============================================================
# FEATURE EXTRACTION FROM SELECTED MAP
# ============================================================

def extract_features(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # normalize
    gray = gray.astype(np.float32) / 255.0

    features = []

    # 1. Mean intensity
    features.append(np.mean(gray))

    # 2. Std deviation
    features.append(np.std(gray))

    # 3. Max intensity
    features.append(np.max(gray))

    # 4. Energy
    features.append(np.sum(gray**2))

    # 5. Non-zero ratio (how much lesion present)
    features.append(np.sum(gray > 0.2) / gray.size)

    return np.array(features)


# ============================================================
# SIMPLE RULE-BASED CLASSIFIER (FAST + NO TRAINING NEEDED)
# ============================================================

def classify(features):
    mean_val = features[0]
    density  = features[4]

    # You can tune these thresholds
    if mean_val < 0.2 and density < 0.2:
        return "Degree I"
    elif mean_val < 0.4:
        return "Degree II"
    else:
        return "Degree III"


# ============================================================
# SAVE RESULTS
# ============================================================

results = []

print("Running Classification...\n")

for file in image_files:

    path = os.path.join(INPUT_DIR, file)
    img = cv2.imread(path)

    if img is None:
        continue

    # extract features
    feat = extract_features(img)

    # classify
    label = classify(feat)

    # save image with label
    output_img = img.copy()

    cv2.putText(output_img,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)

    save_path = os.path.join(OUTPUT_DIR, file.replace("_selected.png", "_classified.png"))
    cv2.imwrite(save_path, output_img)

    # store results
    results.append([file, label] + feat.tolist())

    print(f"{file} -> {label}")


# ============================================================
# SAVE CSV
# ============================================================

columns = ["Image", "Class", "Mean", "Std", "Max", "Energy", "Density"]

df = pd.DataFrame(results, columns=columns)

csv_path = os.path.join(OUTPUT_DIR, "classification_results.csv")
df.to_csv(csv_path, index=False)

print("\n[DONE]")
print("Saved in:", OUTPUT_DIR)