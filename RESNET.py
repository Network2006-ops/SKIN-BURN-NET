# ================================================================
# RESNET-101
# TRAINING + TESTING ACCURACY + LOSS
# ================================================================
#
# Dataset:
# d:/Skin/skin
#
# Current dataset:
# 1238 unlabeled skin images
#
# This version:
# - Does NOT treat output folders as classes
# - Uses ResNet-101 encoder
# - Uses image reconstruction
# - 80% Training / 20% Testing
# - Produces Training Accuracy
# - Produces Testing Accuracy
# - Produces Training Loss
# - Produces Testing Loss
# - Saves combined graph
#
# NOTE:
# Accuracy here is reconstruction accuracy.
# It is NOT burn-severity classification accuracy.
# ================================================================


# ================================================================
# 1. IMPORT LIBRARIES
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


from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)


# ================================================================
# 3. LOCAL WORKSPACE PATHS (VS CODE)
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "skin")

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ResNet101_Training_Testing"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

IMG_SIZE = 224

BATCH_SIZE = 16

EPOCHS = 100

LEARNING_RATE = 0.001

DROPOUT_RATE = 0.1

TEST_SIZE = 0.20

SEED = 42


# ================================================================
# 4. RANDOM SEEDS
# ================================================================

np.random.seed(SEED)

tf.random.set_seed(SEED)


# ================================================================
# 5. HEADER
# ================================================================

print("=" * 80)
print("RESNET-101 TRAINING AND TESTING")
print("=" * 80)

print("\nDataset path:")
print(DATASET_PATH)

print("\nOutput path:")
print(OUTPUT_DIR)

print("\nEpochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Dropout rate:", DROPOUT_RATE)


# ================================================================
# 6. IMAGE EXTENSIONS
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


# ================================================================
# 7. OUTPUT FOLDERS TO IGNORE
# ================================================================

IGNORE_FOLDERS = {

    "BURN_NET_Training",

    "BURN_NET_Training_Testing",

    "BURN_NET_All_Models",

    "Training_Testing_Graphs",

    "ResNet101_Training_Testing"

}


# ================================================================
# 8. FIND ORIGINAL IMAGES
# ================================================================

print("\n")
print("=" * 80)
print("SEARCHING FOR ORIGINAL IMAGES")
print("=" * 80)

image_paths = []


for root, dirs, files in os.walk(
    DATASET_PATH
):

    # ------------------------------------------------------------
    # Remove generated folders from search
    # ------------------------------------------------------------

    dirs[:] = [

        d for d in dirs

        if d not in IGNORE_FOLDERS

    ]


    # ------------------------------------------------------------
    # Find images
    # ------------------------------------------------------------

    for file in files:

        if file.lower().endswith(
            IMAGE_EXTENSIONS
        ):

            image_paths.append(

                os.path.join(
                    root,
                    file
                )

            )


# Sort paths
image_paths = sorted(
    image_paths
)


print(
    "\nTotal original images found:",
    len(image_paths)
)


# ================================================================
# 9. CHECK DATASET
# ================================================================

if len(image_paths) < 10:

    raise ValueError(

        """
Not enough images were found.

Please check:

d:/Skin/skin

The dataset must contain at least 10 images.
"""

    )


# ================================================================
# 10. LOAD IMAGES
# ================================================================

print("\n")
print("=" * 80)
print("LOADING IMAGES")
print("=" * 80)


images = []

valid_paths = []


for path in tqdm(
    image_paths
):

    try:

        # --------------------------------------------------------
        # Read image
        # --------------------------------------------------------

        img = cv2.imread(
            path
        )


        # --------------------------------------------------------
        # Check image
        # --------------------------------------------------------

        if img is None:

            continue


        # --------------------------------------------------------
        # BGR -> RGB
        # --------------------------------------------------------

        img = cv2.cvtColor(

            img,

            cv2.COLOR_BGR2RGB

        )


        # --------------------------------------------------------
        # Resize
        # --------------------------------------------------------

        img = cv2.resize(

            img,

            (
                IMG_SIZE,
                IMG_SIZE
            ),

            interpolation=cv2.INTER_AREA

        )


        # --------------------------------------------------------
        # Normalize
        # --------------------------------------------------------

        img = img.astype(
            np.float32
        ) / 255.0


        # --------------------------------------------------------
        # Store
        # --------------------------------------------------------

        images.append(
            img
        )

        valid_paths.append(
            path
        )


    except Exception as e:

        print(
            "\nSkipped:",
            path
        )

        print(
            "Reason:",
            e
        )


# ================================================================
# 11. CONVERT TO NUMPY
# ================================================================

X = np.array(

    images,

    dtype=np.float32

)


print("\n")
print(
    "Successfully loaded:",
    len(X)
)

print(
    "Image shape:",
    X.shape
)

print(
    "Minimum pixel value:",
    X.min()
)

print(
    "Maximum pixel value:",
    X.max()
)


# ================================================================
# 12. CHECK DATA
# ================================================================

if len(X) < 10:

    raise ValueError(
        "Too few valid images were loaded."
    )


if X.ndim != 4:

    raise ValueError(
        f"Unexpected image shape: {X.shape}"
    )


# ================================================================
# 13. TRAIN / TEST SPLIT
# ================================================================

print("\n")
print("=" * 80)
print("TRAINING / TESTING SPLIT")
print("=" * 80)


indices = np.arange(
    len(X)
)


# Shuffle
rng = np.random.default_rng(
    SEED
)

rng.shuffle(
    indices
)


# Calculate test size
test_count = int(

    len(X) * TEST_SIZE

)


# Test indices
test_indices = indices[
    :test_count
]


# Training indices
train_indices = indices[
    test_count:
]


# Create datasets
X_train = X[
    train_indices
]

X_test = X[
    test_indices
]


print(
    "\nTotal images:",
    len(X)
)

print(
    "Training images:",
    len(X_train)
)

print(
    "Testing images:",
    len(X_test)
)

print(
    "Training percentage:",
    f"{len(X_train) / len(X) * 100:.2f}%"
)

print(
    "Testing percentage:",
    f"{len(X_test) / len(X) * 100:.2f}%"
)


# ================================================================
# 14. RESNET-101
# ================================================================

print("\n")
print("=" * 80)
print("RESNET-101 MODEL")
print("=" * 80)


# ------------------------------------------------
# ResNet-101 pretrained feature extractor
# ------------------------------------------------

base_model = ResNet101(

    include_top=False,

    weights="imagenet",

    input_shape=(

        IMG_SIZE,
        IMG_SIZE,
        3

    )

)


# ------------------------------------------------
# Freeze ResNet-101
# ------------------------------------------------

base_model.trainable = False


print(
    "\nResNet-101 loaded successfully."
)

print(
    "ImageNet pretrained weights: YES"
)

print(
    "Base model frozen: YES"
)


# ================================================================
# 15. BUILD RESNET-101 AUTOENCODER
# ================================================================

inputs = layers.Input(

    shape=(

        IMG_SIZE,
        IMG_SIZE,
        3

    ),

    name="Skin_Image_Input"

)


# ------------------------------------------------
# ResNet-101 encoder
# ------------------------------------------------

features = base_model(

    inputs,

    training=False

)


# ResNet output:
# 7 x 7 x 2048


# ================================================================
# 16. FEATURE BOTTLENECK
# ================================================================

x = layers.Conv2D(

    512,

    (3, 3),

    padding="same",

    activation="relu",

    name="ResNet101_Features"

)(features)


x = layers.Dropout(
    DROPOUT_RATE,
    name="Bottleneck_Dropout"
)(x)


# ================================================================
# 17. DECODER
# ================================================================

x = layers.UpSampling2D(

    size=(2, 2)

)(x)

# 14 x 14


x = layers.Conv2D(

    256,

    (3, 3),

    padding="same",

    activation="relu"

)(x)


x = layers.UpSampling2D(

    size=(2, 2)

)(x)

# 28 x 28


x = layers.Conv2D(

    128,

    (3, 3),

    padding="same",

    activation="relu"

)(x)


x = layers.UpSampling2D(

    size=(2, 2)

)(x)

# 56 x 56


x = layers.Conv2D(

    64,

    (3, 3),

    padding="same",

    activation="relu"

)(x)


x = layers.UpSampling2D(

    size=(2, 2)

)(x)

# 112 x 112


x = layers.Conv2D(

    32,

    (3, 3),

    padding="same",

    activation="relu"

)(x)


x = layers.UpSampling2D(

    size=(2, 2)

)(x)

# 224 x 224


# ================================================================
# 18. RECONSTRUCTED IMAGE
# ================================================================

outputs = layers.Conv2D(

    3,

    (3, 3),

    padding="same",

    activation="sigmoid",

    name="Reconstructed_Skin_Image"

)(x)


# ================================================================
# 19. CREATE MODEL
# ================================================================

model = Model(

    inputs=inputs,

    outputs=outputs,

    name="ResNet101_BURN_NET"

)


# ================================================================
# 20. RECONSTRUCTION ACCURACY
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
# 21. COMPILE MODEL
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
# 22. MODEL SUMMARY
# ================================================================

print("\n")

model.summary()


# ================================================================
# 23. CALLBACKS
# ================================================================

callbacks = [

    EarlyStopping(

        monitor="val_loss",

        patience=15,

        restore_best_weights=True,

        verbose=1

    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=5,

        min_lr=1e-7,

        verbose=1

    )

]


# ================================================================
# 24. TRAINING
# ================================================================

print("\n")
print("=" * 80)
print("RESNET-101 TRAINING STARTED")
print("=" * 80)


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

    callbacks=callbacks,

    verbose=1

)


# ================================================================
# 25. TRAINING COMPLETED
# ================================================================

print("\n")
print("=" * 80)
print("TRAINING COMPLETED")
print("=" * 80)


epochs_completed = len(

    history.history["loss"]

)


print(
    "\nEpochs completed:",
    epochs_completed
)


# ================================================================
# 26. EXTRACT HISTORY
# ================================================================

training_loss = np.array(

    history.history[
        "loss"
    ]

)


testing_loss = np.array(

    history.history[
        "val_loss"
    ]

)


training_accuracy = np.array(

    history.history[
        "reconstruction_accuracy"
    ]

)


testing_accuracy = np.array(

    history.history[
        "val_reconstruction_accuracy"
    ]

)


epochs_range = np.arange(

    1,

    epochs_completed + 1

)


# ================================================================
# 27. FINAL EVALUATION
# ================================================================

print("\n")
print("=" * 80)
print("FINAL MODEL EVALUATION")
print("=" * 80)


train_results = model.evaluate(

    X_train,

    X_train,

    verbose=0

)


test_results = model.evaluate(

    X_test,

    X_test,

    verbose=0

)


final_training_loss = train_results[0]

final_training_accuracy = train_results[1]


final_testing_loss = test_results[0]

final_testing_accuracy = test_results[1]


# ================================================================
# 28. DISPLAY RESULTS
# ================================================================

print("\nFinal Training Accuracy:")
print(
    f"{final_training_accuracy * 100:.2f}%"
)


print("\nFinal Testing Accuracy:")
print(
    f"{final_testing_accuracy * 100:.2f}%"
)


print("\nFinal Training Loss:")
print(
    f"{final_training_loss:.6f}"
)


print("\nFinal Testing Loss:")
print(
    f"{final_testing_loss:.6f}"
)


# ================================================================
# 29. SAVE MODEL
# ================================================================

MODEL_PATH = os.path.join(

    OUTPUT_DIR,

    "ResNet101_Training_Testing.keras"

)


model.save(

    MODEL_PATH

)


print("\n")
print("Model saved:")
print(
    MODEL_PATH
)


# ================================================================
# 30. ACCURACY GRAPH
# ================================================================

plt.figure(

    figsize=(10, 7)

)


plt.plot(

    epochs_range,

    training_accuracy * 100,

    linestyle="--",

    linewidth=2,

    marker="o",

    markersize=3,

    label="Training Accuracy"

)


plt.plot(

    epochs_range,

    testing_accuracy * 100,

    linestyle="--",

    linewidth=2,

    marker="s",

    markersize=3,

    label="Testing Accuracy"

)


plt.xlabel(

    "Epochs",

    fontsize=16,

    fontweight="bold"

)


plt.ylabel(

    "Accuracy (%)",

    fontsize=16,

    fontweight="bold"

)


plt.title(

    "ResNet-101 Training and Testing Accuracy",

    fontsize=18,

    fontweight="bold"

)


plt.xticks(

    fontsize=13

)


plt.yticks(

    fontsize=13

)


plt.ylim(

    0,

    100

)


plt.grid(

    True,

    linestyle=":",

    alpha=0.6

)


plt.legend(

    fontsize=13

)


plt.tight_layout()


ACCURACY_GRAPH = os.path.join(

    OUTPUT_DIR,

    "ResNet101_Training_Testing_Accuracy.png"

)


plt.savefig(

    ACCURACY_GRAPH,

    dpi=600,

    bbox_inches="tight"

)


plt.close()


# ================================================================
# 31. LOSS GRAPH
# ================================================================

plt.figure(

    figsize=(10, 7)

)


plt.plot(

    epochs_range,

    training_loss,

    linestyle="--",

    linewidth=2,

    marker="o",

    markersize=3,

    label="Training Loss"

)


plt.plot(

    epochs_range,

    testing_loss,

    linestyle="--",

    linewidth=2,

    marker="s",

    markersize=3,

    label="Testing Loss"

)


plt.xlabel(

    "Epochs",

    fontsize=16,

    fontweight="bold"

)


plt.ylabel(

    "Loss",

    fontsize=16,

    fontweight="bold"

)


plt.title(

    "ResNet-101 Training and Testing Loss",

    fontsize=18,

    fontweight="bold"

)


plt.xticks(

    fontsize=13

)


plt.yticks(

    fontsize=13

)


plt.grid(

    True,

    linestyle=":",

    alpha=0.6

)


plt.legend(

    fontsize=13

)


plt.tight_layout()


LOSS_GRAPH = os.path.join(

    OUTPUT_DIR,

    "ResNet101_Training_Testing_Loss.png"

)


plt.savefig(

    LOSS_GRAPH,

    dpi=600,

    bbox_inches="tight"

)


plt.close()


# ================================================================
# 32. COMBINED ACCURACY + LOSS GRAPH
# ================================================================

plt.figure(

    figsize=(16, 7)

)


# ------------------------------------------------
# Accuracy
# ------------------------------------------------

plt.subplot(

    1,

    2,

    1

)


plt.plot(

    epochs_range,

    training_accuracy * 100,

    linestyle="--",

    linewidth=2,

    marker="o",

    markersize=3,

    label="Training"

)


plt.plot(

    epochs_range,

    testing_accuracy * 100,

    linestyle="--",

    linewidth=2,

    marker="s",

    markersize=3,

    label="Testing"

)


plt.xlabel(

    "Epochs",

    fontsize=15,

    fontweight="bold"

)


plt.ylabel(

    "Accuracy (%)",

    fontsize=15,

    fontweight="bold"

)


plt.title(

    "Accuracy",

    fontsize=17,

    fontweight="bold"

)


plt.ylim(

    0,

    100

)


plt.xticks(

    fontsize=12

)


plt.yticks(

    fontsize=12

)


plt.grid(

    True,

    linestyle=":",

    alpha=0.6

)


plt.legend(

    fontsize=12

)


# ------------------------------------------------
# Loss
# ------------------------------------------------

plt.subplot(

    1,

    2,

    2

)


plt.plot(

    epochs_range,

    training_loss,

    linestyle="--",

    linewidth=2,

    marker="o",

    markersize=3,

    label="Training"

)


plt.plot(

    epochs_range,

    testing_loss,

    linestyle="--",

    linewidth=2,

    marker="s",

    markersize=3,

    label="Testing"

)


plt.xlabel(

    "Epochs",

    fontsize=15,

    fontweight="bold"

)


plt.ylabel(

    "Loss",

    fontsize=15,

    fontweight="bold"

)


plt.title(

    "Loss",

    fontsize=17,

    fontweight="bold"

)


plt.xticks(

    fontsize=12

)


plt.yticks(

    fontsize=12

)


plt.grid(

    True,

    linestyle=":",

    alpha=0.6

)


plt.legend(

    fontsize=12

)


plt.tight_layout()


COMBINED_GRAPH = os.path.join(

    OUTPUT_DIR,

    "ResNet101_Training_Testing_Accuracy_Loss.png"

)


plt.savefig(

    COMBINED_GRAPH,

    dpi=600,

    bbox_inches="tight"

)


plt.close()


# ================================================================
# 33. SAVE TRAINING HISTORY
# ================================================================

history_df = pd.DataFrame({

    "Epoch":
        epochs_range,

    "Training_Accuracy":
        training_accuracy * 100,

    "Testing_Accuracy":
        testing_accuracy * 100,

    "Training_Loss":
        training_loss,

    "Testing_Loss":
        testing_loss

})


HISTORY_PATH = os.path.join(

    OUTPUT_DIR,

    "ResNet101_Training_Testing_History.csv"

)


history_df.to_csv(

    HISTORY_PATH,

    index=False

)


# ================================================================
# 34. SAVE FINAL RESULTS
# ================================================================

final_results = pd.DataFrame({

    "Metric": [

        "Training Accuracy",

        "Testing Accuracy",

        "Training Loss",

        "Testing Loss"

    ],

    "Value": [

        final_training_accuracy * 100,

        final_testing_accuracy * 100,

        final_training_loss,

        final_testing_loss

    ]

})


FINAL_RESULTS_PATH = os.path.join(

    OUTPUT_DIR,

    "ResNet101_Final_Results.csv"

)


final_results.to_csv(

    FINAL_RESULTS_PATH,

    index=False

)


# ================================================================
# 35. FINAL OUTPUT
# ================================================================

print("\n")
print("=" * 80)
print("RESNET-101 PROCESS COMPLETED")
print("=" * 80)


print("\nDataset images:")
print(
    len(X)
)


print("\nTraining images:")
print(
    len(X_train)
)


print("\nTesting images:")
print(
    len(X_test)
)


print("\nEpochs completed:")
print(
    epochs_completed[-1]
)


print("\n")
print("-" * 80)

print(
    "Training Accuracy :",
    f"{final_training_accuracy * 100:.2f}%"
)

print(
    "Testing Accuracy  :",
    f"{final_testing_accuracy * 100:.2f}%"
)

print(
    "Training Loss     :",
    f"{final_training_loss:.6f}"
)

print(
    "Testing Loss      :",
    f"{final_testing_loss:.6f}"
)

print("-" * 80)


print("\n")
print("GENERATED FILES")
print("-" * 80)


print(
    "\n1. Accuracy graph:"
)

print(
    ACCURACY_GRAPH
)


print(
    "\n2. Loss graph:"
)

print(
    LOSS_GRAPH
)


print(
    "\n3. Combined Accuracy + Loss graph:"
)

print(
    COMBINED_GRAPH
)


print(
    "\n4. Training history CSV:"
)

print(
    HISTORY_PATH
)


print(
    "\n5. Final results CSV:"
)

print(
    FINAL_RESULTS_PATH
)


print(
    "\n6. ResNet-101 model:"
)

print(
    MODEL_PATH
)


print("\n")
print("=" * 80)