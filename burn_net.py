# ================================================================
# BURN-NET TRAINING + TESTING
# TRAINING/TESTING ACCURACY AND LOSS
#
# Dataset:
# d:/Skin/skin
#
# NOTE:
# This dataset has no class labels, so the accuracy below is
# RECONSTRUCTION ACCURACY, not burn-severity classification accuracy.
# ================================================================

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from tqdm import tqdm
from tensorflow.keras import layers, Model

# ================================================================
# 1. LOCAL WORKSPACE PATHS (VS CODE)
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "skin")

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "BURN_NET_Training_Testing"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.1
TEST_SIZE = 0.20
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 70)
print("BURN-NET TRAINING + TESTING")
print("=" * 70)

print("\nDataset path:")
print(DATASET_PATH)
print("Epochs       :", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Dropout rate :", DROPOUT_RATE)

# ================================================================
# 3. FIND IMAGE FILES
# ================================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
)

image_paths = []

for root, dirs, files in os.walk(DATASET_PATH):

    # Do not search generated output folder
    if OUTPUT_DIR in root:
        continue

    for file in files:

        if file.lower().endswith(IMAGE_EXTENSIONS):

            image_paths.append(
                os.path.join(root, file)
            )

image_paths = sorted(image_paths)

print("\nTotal images found:", len(image_paths))

if len(image_paths) == 0:
    raise ValueError(
        "No images found in the dataset."
    )

# ================================================================
# 4. LOAD IMAGES
# ================================================================

print("\nLoading images...")

images = []
valid_paths = []

for path in tqdm(image_paths):

    try:

        img = cv2.imread(path)

        if img is None:
            continue

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        img = cv2.resize(
            img,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA
        )

        img = img.astype(
            np.float32
        ) / 255.0

        images.append(img)
        valid_paths.append(path)

    except Exception as e:

        print(
            "Skipped:",
            path,
            "Reason:",
            e
        )

X = np.array(
    images,
    dtype=np.float32
)

print("\nSuccessfully loaded:", len(X))
print("Image shape:", X.shape)

# ================================================================
# 5. DATA CHECK
# ================================================================

if len(X) < 2:

    raise ValueError(
        "At least 2 images are required."
    )

print("\nPixel range:")
print("Minimum:", X.min())
print("Maximum:", X.max())

# ================================================================
# 6. TRAIN / TEST SPLIT
# ================================================================

indices = np.arange(len(X))

np.random.shuffle(indices)

test_count = int(
    len(X) * TEST_SIZE
)

test_indices = indices[:test_count]
train_indices = indices[test_count:]

X_train = X[train_indices]
X_test = X[test_indices]

print("\n")
print("=" * 70)
print("DATA SPLIT")
print("=" * 70)

print("Total images    :", len(X))
print("Training images :", len(X_train))
print("Testing images  :", len(X_test))

# ================================================================
# 7. BURN-NET MODEL
# ================================================================

print("\n")
print("=" * 70)
print("BURN-NET MODEL")
print("=" * 70)

input_layer = layers.Input(
    shape=(IMG_SIZE, IMG_SIZE, 3),
    name="Skin_Image_Input"
)

# ------------------------------------------------
# ENCODER
# ------------------------------------------------

x = layers.Conv2D(
    32,
    (3, 3),
    padding="same",
    activation="relu"
)(input_layer)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D(
    (2, 2)
)(x)


x = layers.Conv2D(
    64,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D(
    (2, 2)
)(x)


x = layers.Conv2D(
    128,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D(
    (2, 2)
)(x)

# ------------------------------------------------
# BURN-NET FEATURE EXTRACTION
# ------------------------------------------------

features = layers.Conv2D(
    256,
    (3, 3),
    padding="same",
    activation="relu",
    name="BURN_NET_Features"
)(x)

# ------------------------------------------------
# DROPOUT REGULARIZATION
# ------------------------------------------------

dropout = layers.Dropout(
    DROPOUT_RATE,
    name="BURN_NET_Dropout"
)(features)

# ------------------------------------------------
# DECODER
# ------------------------------------------------

x = layers.Conv2D(
    128,
    (3, 3),
    padding="same",
    activation="relu"
)(dropout)

x = layers.UpSampling2D(
    (2, 2)
)(x)


x = layers.Conv2D(
    64,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = layers.UpSampling2D(
    (2, 2)
)(x)


x = layers.Conv2D(
    32,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = layers.UpSampling2D(
    (2, 2)
)(x)

# ------------------------------------------------
# OUTPUT
# ------------------------------------------------

output_layer = layers.Conv2D(
    3,
    (3, 3),
    padding="same",
    activation="sigmoid",
    name="Reconstructed_Skin_Image"
)(x)

# ================================================================
# 8. CREATE MODEL
# ================================================================

model = Model(
    inputs=input_layer,
    outputs=output_layer,
    name="BURN_NET"
)

# ================================================================
# 9. RECONSTRUCTION ACCURACY
# ================================================================
#
# Accuracy = 1 - Mean Absolute Error
#
# This is NOT burn-class classification accuracy.
# ================================================================

def reconstruction_accuracy(
    y_true,
    y_pred
):

    mae = tf.reduce_mean(
        tf.abs(
            y_true - y_pred
        )
    )

    accuracy = 1.0 - mae

    return accuracy

# ================================================================
# 10. COMPILE MODEL
# ================================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="mse",

    metrics=[
        reconstruction_accuracy
    ]
)

# ================================================================
# 11. MODEL SUMMARY
# ================================================================

model.summary()

# ================================================================
# 12. TRAINING
# ================================================================

print("\n")
print("=" * 70)
print("BURN-NET TRAINING STARTED")
print("=" * 70)

history = model.fit(

    X_train,
    X_train,

    validation_data=(
        X_test,
        X_test
    ),

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

    shuffle=True,

    verbose=1
)

# ================================================================
# 13. TRAINING COMPLETED
# ================================================================

print("\n")
print("=" * 70)
print("BURN-NET TRAINING COMPLETED")
print("=" * 70)

print(
    "Epochs completed:",
    len(history.history["loss"])
)

# ================================================================
# 14. GET TRAINING / TESTING RESULTS
# ================================================================

train_loss = history.history["loss"]

test_loss = history.history["val_loss"]

train_accuracy = history.history[
    "reconstruction_accuracy"
]

test_accuracy = history.history[
    "val_reconstruction_accuracy"
]

epochs_range = np.arange(
    1,
    len(train_loss) + 1
)

# ================================================================
# 15. PRINT FINAL RESULTS
# ================================================================

print("\n")
print("=" * 70)
print("FINAL TRAINING / TESTING RESULTS")
print("=" * 70)

print(
    "\nFinal Training Accuracy :",
    f"{train_accuracy[-1] * 100:.2f}%"
)

print(
    "Final Testing Accuracy  :",
    f"{test_accuracy[-1] * 100:.2f}%"
)

print(
    "Final Training Loss     :",
    f"{train_loss[-1]:.6f}"
)

print(
    "Final Testing Loss      :",
    f"{test_loss[-1]:.6f}"
)

# ================================================================
# 16. CREATE ACCURACY + LOSS GRAPH
# ================================================================

plt.figure(
    figsize=(14, 6)
)

# ------------------------------------------------
# LEFT: ACCURACY
# ------------------------------------------------

plt.subplot(1, 2, 1)

plt.plot(
    epochs_range,
    train_accuracy,
    linestyle="--",
    linewidth=1.8,
    label="Training"
)

plt.plot(
    epochs_range,
    test_accuracy,
    linestyle="--",
    linewidth=1.8,
    label="Testing"
)

plt.xlabel(
    "Epochs",
    fontsize=14
)

plt.ylabel(
    "Accuracy",
    fontsize=14
)

plt.title(
    "BURN-NET Accuracy",
    fontsize=16,
    fontweight="bold"
)

plt.xticks(
    fontsize=11
)

plt.yticks(
    fontsize=11
)

plt.grid(
    True,
    alpha=0.5
)

plt.legend(
    fontsize=11
)

# ------------------------------------------------
# RIGHT: LOSS
# ------------------------------------------------

plt.subplot(1, 2, 2)

plt.plot(
    epochs_range,
    train_loss,
    linestyle="--",
    linewidth=1.8,
    label="Training"
)

plt.plot(
    epochs_range,
    test_loss,
    linestyle="--",
    linewidth=1.8,
    label="Testing"
)

plt.xlabel(
    "Epochs",
    fontsize=14
)

plt.ylabel(
    "Loss",
    fontsize=14
)

plt.title(
    "BURN-NET Loss",
    fontsize=16,
    fontweight="bold"
)

plt.xticks(
    fontsize=11
)

plt.yticks(
    fontsize=11
)

plt.grid(
    True,
    alpha=0.5
)

plt.legend(
    fontsize=11
)

plt.tight_layout()

# ================================================================
# 17. SAVE GRAPH
# ================================================================

GRAPH_PATH = os.path.join(
    OUTPUT_DIR,
    "BURN_NET_Training_Testing_Accuracy_Loss.png"
)

plt.savefig(
    GRAPH_PATH,
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print("\nGraph saved:")
print(GRAPH_PATH)

# ================================================================
# 18. SAVE HISTORY CSV
# ================================================================

history_df = pd.DataFrame({

    "Epoch": epochs_range,

    "Training_Accuracy":
        train_accuracy,

    "Testing_Accuracy":
        test_accuracy,

    "Training_Loss":
        train_loss,

    "Testing_Loss":
        test_loss
})

CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "BURN_NET_Training_Testing_History.csv"
)

history_df.to_csv(
    CSV_PATH,
    index=False
)

print("\nHistory saved:")
print(CSV_PATH)

# ================================================================
# 19. SAVE MODEL
# ================================================================

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "BURN_NET_Training_Testing.keras"
)

model.save(
    MODEL_PATH
)

print("\nModel saved:")
print(MODEL_PATH)

# ================================================================
# 20. FINAL OUTPUT
# ================================================================

print("\n")
print("=" * 70)
print("FINAL OUTPUT")
print("=" * 70)

print(
    "Total images       :",
    len(X)
)

print(
    "Training images    :",
    len(X_train)
)

print(
    "Testing images     :",
    len(X_test)
)

print(
    "Epochs             :",
    len(epochs_range)
)

print(
    "Training Accuracy  :",
    f"{train_accuracy[-1] * 100:.2f}%"
)

print(
    "Testing Accuracy   :",
    f"{test_accuracy[-1] * 100:.2f}%"
)

print(
    "Training Loss      :",
    f"{train_loss[-1]:.6f}"
)

print(
    "Testing Loss       :",
    f"{test_loss[-1]:.6f}"
)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")
print("1. BURN_NET_Training_Testing.keras")
print("2. BURN_NET_Training_Testing_Accuracy_Loss.png")
print("3. BURN_NET_Training_Testing_History.csv")

print("=" * 70)