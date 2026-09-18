# ================================================================
# CNN WITH VGG16
# 100 EPOCH TRAINING / TESTING
# ACCURACY AND LOSS
# FOR UNLABELED SKIN IMAGES
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

from tqdm import tqdm


import tensorflow as tf

from tensorflow.keras.applications import VGG16

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    UpSampling2D,
    BatchNormalization,
    Activation,
    Dropout
)

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    ModelCheckpoint
)

from sklearn.model_selection import train_test_split


# ================================================================
# 2. LOCAL WORKSPACE PATHS (VS CODE)
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "skin")

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "VGG16_Training_Testing"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# 4. PARAMETERS
# ================================================================

IMG_SIZE = 224

BATCH_SIZE = 16

# IMPORTANT: 100 EPOCHS
EPOCHS = 100

LEARNING_RATE = 0.001

DROPOUT_RATE = 0.1

SEED = 42


np.random.seed(SEED)

tf.random.set_seed(SEED)


# ================================================================
# 5. IMAGE EXTENSIONS
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
# 6. FOLDERS TO IGNORE
# ================================================================

IGNORE_FOLDERS = {

    "BURN_NET_Training",

    "BURN_NET_Training_Testing",

    "BURN_NET_All_Models",

    "Training_Testing_Graphs",

    "ResNet101_Training_Testing",

    "VGG16_Training_Testing"

}


# ================================================================
# 7. FIND ORIGINAL IMAGES
# ================================================================

print("=" * 70)

print("CNN WITH VGG16")

print("IMAGE SEARCH")

print("=" * 70)


image_paths = []


for root, dirs, files in os.walk(DATASET_PATH):

    # Ignore generated output folders
    dirs[:] = [
        d for d in dirs
        if d not in IGNORE_FOLDERS
    ]


    for file in files:

        if file.lower().endswith(
            IMAGE_EXTENSIONS
        ):

            image_path = os.path.join(
                root,
                file
            )

            image_paths.append(
                image_path
            )


# Remove duplicate paths
image_paths = sorted(
    list(set(image_paths))
)


print()

print(
    "Total image files found:",
    len(image_paths)
)


if len(image_paths) == 0:

    raise ValueError(
        "No image files found."
    )


# ================================================================
# 8. LOAD AND PREPROCESS IMAGES
# ================================================================

print()

print("=" * 70)

print("LOADING AND PREPROCESSING IMAGES")

print("=" * 70)


X = []

valid_paths = []


for image_path in tqdm(
    image_paths,
    desc="Processing images"
):

    image = cv2.imread(
        image_path
    )


    if image is None:

        continue


    # BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # Resize
    image = cv2.resize(
        image,
        (
            IMG_SIZE,
            IMG_SIZE
        ),
        interpolation=cv2.INTER_AREA
    )


    # Normalize
    image = image.astype(
        np.float32
    ) / 255.0


    X.append(
        image
    )


    valid_paths.append(
        image_path
    )


# Convert to NumPy
X = np.asarray(
    X,
    dtype=np.float32
)


print()

print(
    "Dataset shape:",
    X.shape
)

print(
    "Valid images:",
    len(X)
)


# ================================================================
# 9. TRAIN / TEST SPLIT
# ================================================================

X_train, X_test = train_test_split(

    X,

    test_size=0.20,

    random_state=SEED
)


print()

print("=" * 70)

print("DATASET SPLIT")

print("=" * 70)


print(
    "Total images   :",
    len(X)
)

print(
    "Training images:",
    len(X_train)
)

print(
    "Testing images :",
    len(X_test)
)


# ================================================================
# 10. LOAD VGG16
# ================================================================

print()

print("=" * 70)

print("LOADING VGG16")

print("=" * 70)


vgg16_base = VGG16(

    weights="imagenet",

    include_top=False,

    input_shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    )

)


# ================================================================
# 11. FREEZE VGG16 BASE
# ================================================================

vgg16_base.trainable = False


print()

print("VGG16 base model loaded")

print("Weights : ImageNet")

print("Base model : Frozen")


# ================================================================
# 12. INPUT
# ================================================================

inputs = Input(

    shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    ),

    name="Input_Image"

)


# ================================================================
# 13. VGG16 FEATURE EXTRACTION
# ================================================================

x = vgg16_base(

    inputs,

    training=False

)


# ================================================================
# 14. BOTTLENECK
# ================================================================

x = Conv2D(

    512,

    (3, 3),

    padding="same",

    name="VGG16_Features"

)(x)


x = BatchNormalization()(x)


x = Activation(
    "relu"
)(x)


x = Dropout(
    DROPOUT_RATE,
    name="Bottleneck_Dropout"
)(x)


# ================================================================
# 15. DECODER
# ================================================================

# ------------------------------------------------
# 7x7 -> 14x14
# ------------------------------------------------

x = UpSampling2D(

    size=(2, 2)

)(x)


x = Conv2D(

    256,

    (3, 3),

    padding="same"

)(x)


x = BatchNormalization()(x)


x = Activation(
    "relu"
)(x)


# ------------------------------------------------
# 14x14 -> 28x28
# ------------------------------------------------

x = UpSampling2D(

    size=(2, 2)

)(x)


x = Conv2D(

    128,

    (3, 3),

    padding="same"

)(x)


x = BatchNormalization()(x)


x = Activation(
    "relu"
)(x)


# ------------------------------------------------
# 28x28 -> 56x56
# ------------------------------------------------

x = UpSampling2D(

    size=(2, 2)

)(x)


x = Conv2D(

    64,

    (3, 3),

    padding="same"

)(x)


x = BatchNormalization()(x)


x = Activation(
    "relu"
)(x)


# ------------------------------------------------
# 56x56 -> 112x112
# ------------------------------------------------

x = UpSampling2D(

    size=(2, 2)

)(x)


x = Conv2D(

    32,

    (3, 3),

    padding="same"

)(x)


x = BatchNormalization()(x)


x = Activation(
    "relu"
)(x)


# ------------------------------------------------
# 112x112 -> 224x224
# ------------------------------------------------

x = UpSampling2D(

    size=(2, 2)

)(x)


x = Conv2D(

    16,

    (3, 3),

    padding="same"

)(x)


x = BatchNormalization()(x)


x = Activation(
    "relu"
)(x)


# ================================================================
# 16. OUTPUT
# ================================================================

outputs = Conv2D(

    3,

    (3, 3),

    padding="same",

    activation="sigmoid",

    name="Reconstructed_Image"

)(x)


# ================================================================
# 17. CREATE MODEL
# ================================================================

model = Model(

    inputs=inputs,

    outputs=outputs,

    name="CNN_VGG16_Reconstruction"

)


# ================================================================
# 18. MODEL SUMMARY
# ================================================================

print()

print("=" * 70)

print("MODEL SUMMARY")

print("=" * 70)

model.summary()


# ================================================================
# 19. RECONSTRUCTION ACCURACY
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
# 20. COMPILE MODEL
# ================================================================

model.compile(

    optimizer=Adam(

        learning_rate=LEARNING_RATE

    ),

    loss="mean_squared_error",

    metrics=[
        reconstruction_accuracy
    ]

)


# ================================================================
# 21. CALLBACKS
# ================================================================

best_model_path = os.path.join(

    OUTPUT_DIR,

    "CNN_VGG16_Best_Model.keras"

)


callbacks = [

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=8,

        min_lr=1e-7,

        verbose=1

    ),

    ModelCheckpoint(

        best_model_path,

        monitor="val_loss",

        save_best_only=True,

        verbose=1

    )

]


# ================================================================
# 22. TRAIN MODEL FOR 100 EPOCHS
# ================================================================

print()

print("=" * 70)

print("CNN WITH VGG16")

print("100 EPOCH TRAINING")

print("=" * 70)


history = model.fit(

    X_train,

    X_train,

    validation_data=(

        X_test,

        X_test

    ),

    epochs=100,

    batch_size=BATCH_SIZE,

    callbacks=callbacks,

    verbose=1

)


# ================================================================
# 23. SAVE HISTORY
# ================================================================

history_df = pd.DataFrame(

    history.history

)


history_csv = os.path.join(

    OUTPUT_DIR,

    "VGG16_100_Epoch_Training_History.csv"

)


history_df.to_csv(

    history_csv,

    index=False

)


# ================================================================
# 24. EXTRACT ACCURACY
# ================================================================

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


# ================================================================
# 25. EXTRACT LOSS
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


# ================================================================
# 26. ACCURACY TO PERCENTAGE
# ================================================================

training_accuracy_percent = (

    training_accuracy * 100

)


testing_accuracy_percent = (

    testing_accuracy * 100

)


# ================================================================
# 27. EPOCH NUMBERS
# ================================================================

epochs_completed = range(

    1,

    len(training_accuracy) + 1

)


print()

print(
    "Epochs completed:",
    len(training_accuracy)
)


# ================================================================
# 28. ACCURACY GRAPH
# ================================================================

plt.figure(

    figsize=(12, 7)

)


plt.plot(

    epochs_completed,

    training_accuracy_percent,

    linewidth=2.5,

    label="Training Accuracy"

)


plt.plot(

    epochs_completed,

    testing_accuracy_percent,

    linewidth=2.5,

    label="Testing Accuracy"

)


plt.title(

    "CNN with VGG16 - Training and Testing Accuracy",

    fontsize=18,

    fontweight="bold"

)


plt.xlabel(

    "Epoch",

    fontsize=15

)


plt.ylabel(

    "Reconstruction Accuracy (%)",

    fontsize=15

)


plt.xlim(

    1,

    100

)


plt.legend(

    fontsize=12

)


plt.grid(

    True,

    linestyle="--",

    alpha=0.5

)


plt.xticks(

    np.arange(
        0,
        101,
        10
    ),

    fontsize=12

)


plt.yticks(

    fontsize=12

)


plt.tight_layout()


accuracy_path = os.path.join(

    OUTPUT_DIR,

    "VGG16_100_Epoch_Training_Testing_Accuracy.png"

)


plt.savefig(

    accuracy_path,

    dpi=600,

    bbox_inches="tight"

)


plt.close()


# ================================================================
# 29. LOSS GRAPH
# ================================================================

plt.figure(

    figsize=(12, 7)

)


plt.plot(

    epochs_completed,

    training_loss,

    linewidth=2.5,

    label="Training Loss"

)


plt.plot(

    epochs_completed,

    testing_loss,

    linewidth=2.5,

    label="Testing Loss"

)


plt.title(

    "CNN with VGG16 - Training and Testing Loss",

    fontsize=18,

    fontweight="bold"

)


plt.xlabel(

    "Epoch",

    fontsize=15

)


plt.ylabel(

    "Loss",

    fontsize=15

)


plt.xlim(

    1,

    100

)


plt.legend(

    fontsize=12

)


plt.grid(

    True,

    linestyle="--",

    alpha=0.5

)


plt.xticks(

    np.arange(
        0,
        101,
        10
    ),

    fontsize=12

)


plt.yticks(

    fontsize=12

)


plt.tight_layout()


loss_path = os.path.join(

    OUTPUT_DIR,

    "VGG16_100_Epoch_Training_Testing_Loss.png"

)


plt.savefig(

    loss_path,

    dpi=600,

    bbox_inches="tight"

)


plt.close()


# ================================================================
# 30. COMBINED GRAPH
# ================================================================

fig, ax1 = plt.subplots(

    figsize=(13, 7)

)


# Accuracy
ax1.plot(

    epochs_completed,

    training_accuracy_percent,

    linewidth=2.5,

    label="Training Accuracy"

)


ax1.plot(

    epochs_completed,

    testing_accuracy_percent,

    linewidth=2.5,

    label="Testing Accuracy"

)


ax1.set_xlabel(

    "Epoch",

    fontsize=15

)


ax1.set_ylabel(

    "Accuracy (%)",

    fontsize=15

)


ax1.set_xlim(

    1,

    100

)


ax1.set_xticks(

    np.arange(
        0,
        101,
        10
    )

)


ax1.grid(

    True,

    linestyle="--",

    alpha=0.5

)


# Loss
ax2 = ax1.twinx()


ax2.plot(

    epochs_completed,

    training_loss,

    linewidth=2.5,

    linestyle=":",

    label="Training Loss"

)


ax2.plot(

    epochs_completed,

    testing_loss,

    linewidth=2.5,

    linestyle=":",

    label="Testing Loss"

)


ax2.set_ylabel(

    "Loss",

    fontsize=15

)


plt.title(

    "CNN with VGG16 - 100 Epoch Training and Testing Performance",

    fontsize=18,

    fontweight="bold"

)


lines1, labels1 = (

    ax1.get_legend_handles_labels()

)


lines2, labels2 = (

    ax2.get_legend_handles_labels()

)


ax1.legend(

    lines1 + lines2,

    labels1 + labels2,

    fontsize=11,

    loc="best"

)


plt.tight_layout()


combined_path = os.path.join(

    OUTPUT_DIR,

    "VGG16_100_Epoch_Training_Testing_Combined.png"

)


plt.savefig(

    combined_path,

    dpi=600,

    bbox_inches="tight"

)


plt.close()


# ================================================================
# 31. FINAL TRAINING EVALUATION
# ================================================================

train_results = model.evaluate(

    X_train,

    X_train,

    batch_size=BATCH_SIZE,

    verbose=0

)


# ================================================================
# 32. FINAL TESTING EVALUATION
# ================================================================

test_results = model.evaluate(

    X_test,

    X_test,

    batch_size=BATCH_SIZE,

    verbose=0

)


# ================================================================
# 33. FINAL VALUES
# ================================================================

final_train_loss = train_results[0]

final_train_accuracy = train_results[1]


final_test_loss = test_results[0]

final_test_accuracy = test_results[1]


# ================================================================
# 34. SAVE FINAL RESULTS
# ================================================================

results_df = pd.DataFrame({

    "Metric": [

        "Training Accuracy",

        "Testing Accuracy",

        "Training Loss",

        "Testing Loss"

    ],

    "Value": [

        final_train_accuracy,

        final_test_accuracy,

        final_train_loss,

        final_test_loss

    ]

})


results_path = os.path.join(

    OUTPUT_DIR,

    "VGG16_100_Epoch_Final_Results.csv"

)


results_df.to_csv(

    results_path,

    index=False

)


# ================================================================
# 35. SAVE MODEL
# ================================================================

final_model_path = os.path.join(

    OUTPUT_DIR,

    "CNN_VGG16_100_Epoch_Final.keras"

)


model.save(

    final_model_path

)


# ================================================================
# 36. SAVE DATASET SPLIT INFORMATION
# ================================================================

split_info = pd.DataFrame({

    "Dataset": [

        "Total",

        "Training",

        "Testing"

    ],

    "Number_of_Images": [

        len(X),

        len(X_train),

        len(X_test)

    ]

})


split_path = os.path.join(

    OUTPUT_DIR,

    "VGG16_100_Epoch_Dataset_Split.csv"

)


split_info.to_csv(

    split_path,

    index=False

)


# ================================================================
# 37. FINAL OUTPUT
# ================================================================

print()

print("=" * 70)

print("VGG16 CNN - 100 EPOCH PROCESS COMPLETED")

print("=" * 70)


print()

print("Total Images      :", len(X))

print("Training Images   :", len(X_train))

print("Testing Images    :", len(X_test))

print("Epochs            :", len(training_accuracy))


print()

print("-" * 70)

print("FINAL PERFORMANCE")

print("-" * 70)


print(

    f"Training Accuracy : "

    f"{final_train_accuracy * 100:.2f}%"

)


print(

    f"Testing Accuracy  : "

    f"{final_test_accuracy * 100:.2f}%"

)


print(

    f"Training Loss     : "

    f"{final_train_loss:.6f}"

)


print(

    f"Testing Loss      : "

    f"{final_test_loss:.6f}"

)


print()

print("-" * 70)

print("OUTPUT FILES")

print("-" * 70)


print(

    "Accuracy graph :",

    accuracy_path

)


print(

    "Loss graph     :",

    loss_path

)


print(

    "Combined graph :",

    combined_path

)


print(

    "History CSV    :",

    history_csv

)


print(

    "Results CSV    :",

    results_path

)


print(

    "Model          :",

    final_model_path

)


print("=" * 70)