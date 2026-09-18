# ================================================================
# ABDOLAHNEJAD-STYLE CNN
# EFFICIENTNET-B7 FOR BURN SEVERITY CLASSIFICATION
#
# TRAINING / TESTING:
#   - Training Accuracy
#   - Testing Accuracy
#   - Training Loss
#   - Testing Loss
#
# 4 CLASSES:
#   SPF
#   SPT
#   DPT
#   FT
#
# EPOCHS = 100
#
# NOTE:
# This is an implementation inspired by the published
# EfficientNet-B7 burn-severity CNN approach.
# It is NOT a claim of exact reproduction of the paper's
# proprietary/training configuration.
# ================================================================


# ================================================================
# 1. IMPORT LIBRARIES
# ================================================================

import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import seaborn as sns


# ================================================================
# 2. LOCAL WORKSPACE PATHS (VS CODE)
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "skin")
OUTPUT_DIR = os.path.join(BASE_DIR, "Abdolahnejad_EfficientNetB7_CNN")

# ================================================================
# 3. RANDOM SEED
# ================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ================================================================
# 4. CONFIGURATION
# ================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ------------------------------------------------
# Image parameters
# ------------------------------------------------
IMG_SIZE = 224

BATCH_SIZE = 8

# ------------------------------------------------
# Training parameters
# ------------------------------------------------
EPOCHS = 100

LEARNING_RATE = 0.001

DROPOUT_RATE = 0.1

# ------------------------------------------------
# Classes
# ------------------------------------------------
CLASS_NAMES = [
    "SPF",
    "SPT",
    "DPT",
    "FT"
]

NUM_CLASSES = len(
    CLASS_NAMES
)


# ================================================================
# 5. PRINT CONFIGURATION
# ================================================================

print("=" * 75)
print("ABDOLAHNEJAD-STYLE CNN")
print("EFFICIENTNET-B7 BURN SEVERITY CLASSIFICATION")
print("=" * 75)

print("Dataset      :", DATASET_DIR)
print("Output       :", OUTPUT_DIR)
print("Image size   :", IMG_SIZE)
print("Batch size   :", BATCH_SIZE)
print("Epochs       :", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Dropout rate :", DROPOUT_RATE)

print()
print("Classes:")

for i, class_name in enumerate(
    CLASS_NAMES
):
    print(
        f"{i}: {class_name}"
    )

print("=" * 75)


# ================================================================
# 6. CHECK DATASET STRUCTURE
# ================================================================

print()
print("CHECKING DATASET...")
print("-" * 75)

missing_classes = []

for class_name in CLASS_NAMES:

    class_path = os.path.join(
        DATASET_DIR,
        class_name
    )

    if os.path.isdir(
        class_path
    ):

        image_count = 0

        for root, dirs, files in os.walk(
            class_path
        ):

            for file in files:

                if file.lower().endswith(
                    (
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".tif",
                        ".tiff"
                    )
                ):

                    image_count += 1

        print(
            f"{class_name:5s} : "
            f"{image_count} images"
        )

    else:

        missing_classes.append(
            class_name
        )

        print(
            f"{class_name:5s} : "
            "NOT FOUND"
        )


use_df = False

if len(missing_classes) > 0:
    print()
    print("Class subfolders (SPF/SPT/DPT/FT) not found.")
    print("Checking for YOLO .txt annotation files in:", DATASET_DIR)
    
    txt_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.txt")))
    df_rows = []
    for tf_p in txt_files:
        try:
            with open(tf_p, "r") as f:
                lines = [l.strip().split() for l in f if l.strip()]
                if lines and len(lines[0]) >= 1:
                    cls_id = lines[0][0]
                    stem = os.path.splitext(tf_p)[0]
                    for ext in [".jpg", ".jpeg", ".png"]:
                        cand = stem + ext
                        if os.path.exists(cand):
                            df_rows.append({"filename": cand, "class": f"Class_{cls_id}"})
                            break
        except Exception:
            pass

    if len(df_rows) > 0:
        df = pd.DataFrame(df_rows)
        CLASS_NAMES = sorted(df["class"].unique().tolist())
        NUM_CLASSES = len(CLASS_NAMES)
        use_df = True
        print(f"Found {len(df)} images with YOLO annotations.")
        print(f"Detected classes: {CLASS_NAMES} (NUM_CLASSES = {NUM_CLASSES})")
    else:
        print("No image annotations found. Cannot proceed without labels.")
        raise SystemExit

# ================================================================
# 8. DATA AUGMENTATION
# ================================================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=(0.85, 1.15),
    validation_split=0.20
)

# ================================================================
# 9. TRAINING & TESTING DATA GENERATORS
# ================================================================

if use_df:
    train_generator = train_datagen.flow_from_dataframe(
        df,
        x_col="filename",
        y_col="class",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
        seed=SEED,
        subset="training"
    )
    test_generator = train_datagen.flow_from_dataframe(
        df,
        x_col="filename",
        y_col="class",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
        seed=SEED,
        subset="validation"
    )
else:
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
        seed=SEED,
        subset="training"
    )
    test_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
        seed=SEED,
        subset="validation"
    )


# ================================================================
# 11. DISPLAY DATASET INFORMATION
# ================================================================

print()
print("=" * 75)
print("DATASET INFORMATION")
print("=" * 75)

print(
    "Training images:",
    train_generator.samples
)

print(
    "Testing images :",
    test_generator.samples
)

print(
    "Number classes :",
    NUM_CLASSES
)

print(
    "Class indices  :",
    train_generator.class_indices
)

print("=" * 75)


# ================================================================
# 12. BUILD EFFICIENTNET-B7
# ================================================================

print()
print("=" * 75)
print("BUILDING EFFICIENTNET-B7 CNN")
print("=" * 75)


# ------------------------------------------------
# Pre-trained EfficientNet-B7
# ------------------------------------------------

base_model = EfficientNetB7(

    include_top=False,

    weights="imagenet",

    input_shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    )
)


# ------------------------------------------------
# Freeze base network initially
# ------------------------------------------------

base_model.trainable = False


# ================================================================
# 13. CLASSIFICATION HEAD
# ================================================================

inputs = layers.Input(
    shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    )
)


# EfficientNet-B7
x = base_model(
    inputs,
    training=False
)


# Global average pooling
x = layers.GlobalAveragePooling2D()(x)


# Dropout
x = layers.Dropout(
    DROPOUT_RATE
)(x)


# Dense layer
x = layers.Dense(
    512,
    activation="relu"
)(x)


# Batch normalization
x = layers.BatchNormalization()(x)


# Dropout
x = layers.Dropout(
    DROPOUT_RATE
)(x)


# Final classification layer
outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


# Create model
model = models.Model(
    inputs,
    outputs
)


# ================================================================
# 14. COMPILE MODEL
# ================================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ================================================================
# 15. DISPLAY MODEL
# ================================================================

model.summary()


# ================================================================
# 16. CALLBACKS
# ================================================================

checkpoint_path = os.path.join(
    OUTPUT_DIR,
    "Abdolahnejad_EfficientNetB7_Best.keras"
)


checkpoint = ModelCheckpoint(

    checkpoint_path,

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=5,

    min_lr=1e-7,

    verbose=1
)


csv_logger = CSVLogger(

    os.path.join(
        OUTPUT_DIR,
        "training_history.csv"
    ),

    append=False
)


# ================================================================
# 17. TRAIN CNN
# ================================================================

print()
print("=" * 75)
print("STARTING CNN TRAINING")
print("=" * 75)

history = model.fit(

    train_generator,

    validation_data=test_generator,

    epochs=EPOCHS,

    callbacks=[
        checkpoint,
        reduce_lr,
        csv_logger
    ],

    verbose=1
)


# ================================================================
# 18. SAVE FINAL MODEL
# ================================================================

final_model_path = os.path.join(
    OUTPUT_DIR,
    "Abdolahnejad_EfficientNetB7_Final.keras"
)

model.save(
    final_model_path
)


# ================================================================
# 19. TRAINING HISTORY
# ================================================================

history_df = pd.DataFrame(
    history.history
)

history_csv_path = os.path.join(
    OUTPUT_DIR,
    "CNN_Training_History.csv"
)

history_df.to_csv(
    history_csv_path,
    index=False
)


# ================================================================
# 20. TRAINING AND TESTING ACCURACY
# ================================================================

train_accuracy = history.history[
    "accuracy"
]

test_accuracy = history.history[
    "val_accuracy"
]


# ================================================================
# 21. TRAINING AND TESTING LOSS
# ================================================================

train_loss = history.history[
    "loss"
]

test_loss = history.history[
    "val_loss"
]


# ================================================================
# 22. FINAL TRAINING VALUES
# ================================================================

print()
print("=" * 75)
print("FINAL TRAINING RESULTS")
print("=" * 75)

print(
    "Final Training Accuracy:",
    f"{train_accuracy[-1] * 100:.2f}%"
)

print(
    "Final Testing Accuracy :",
    f"{test_accuracy[-1] * 100:.2f}%"
)

print(
    "Final Training Loss    :",
    f"{train_loss[-1]:.4f}"
)

print(
    "Final Testing Loss     :",
    f"{test_loss[-1]:.4f}"
)


# ================================================================
# 23. BEST TRAINING ACCURACY
# ================================================================

best_train_accuracy = max(
    train_accuracy
)

best_test_accuracy = max(
    test_accuracy
)

best_train_epoch = (
    np.argmax(
        train_accuracy
    ) + 1
)

best_test_epoch = (
    np.argmax(
        test_accuracy
    ) + 1
)


print()
print(
    "Best Training Accuracy:",
    f"{best_train_accuracy * 100:.2f}%"
)

print(
    "Best Training Epoch:",
    best_train_epoch
)

print(
    "Best Testing Accuracy:",
    f"{best_test_accuracy * 100:.2f}%"
)

print(
    "Best Testing Epoch:",
    best_test_epoch
)


# ================================================================
# 24. ACCURACY GRAPH
# ================================================================

epochs_range = range(
    1,
    len(train_accuracy) + 1
)

plt.figure(
    figsize=(12, 7)
)

plt.plot(
    epochs_range,
    np.array(train_accuracy) * 100,
    linewidth=2,
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    np.array(test_accuracy) * 100,
    linewidth=2,
    label="Testing Accuracy"
)

plt.xlabel(
    "Epoch",
    fontsize=16
)

plt.ylabel(
    "Accuracy (%)",
    fontsize=16
)

plt.title(
    "CNN Training and Testing Accuracy",
    fontsize=18,
    fontweight="bold"
)

plt.legend(
    fontsize=13
)

plt.grid(
    True,
    alpha=0.3
)

plt.xticks(
    fontsize=12
)

plt.yticks(
    fontsize=12
)

plt.tight_layout()


accuracy_graph_path = os.path.join(
    OUTPUT_DIR,
    "CNN_Training_Testing_Accuracy.png"
)

plt.savefig(
    accuracy_graph_path,
    dpi=600,
    bbox_inches="tight"
)

plt.close()

plt.close()


# ================================================================
# 25. LOSS GRAPH
# ================================================================

plt.figure(
    figsize=(12, 7)
)

plt.plot(
    epochs_range,
    train_loss,
    linewidth=2,
    label="Training Loss"
)

plt.plot(
    epochs_range,
    test_loss,
    linewidth=2,
    label="Testing Loss"
)

plt.xlabel(
    "Epoch",
    fontsize=16
)

plt.ylabel(
    "Loss",
    fontsize=16
)

plt.title(
    "CNN Training and Testing Loss",
    fontsize=18,
    fontweight="bold"
)

plt.legend(
    fontsize=13
)

plt.grid(
    True,
    alpha=0.3
)

plt.xticks(
    fontsize=12
)

plt.yticks(
    fontsize=12
)

plt.tight_layout()


loss_graph_path = os.path.join(
    OUTPUT_DIR,
    "CNN_Training_Testing_Loss.png"
)

plt.savefig(
    loss_graph_path,
    dpi=600,
    bbox_inches="tight"
)

plt.close()

plt.close()


# ================================================================
# 26. EVALUATE ON TESTING DATA
# ================================================================

print()
print("=" * 75)
print("TESTING MODEL")
print("=" * 75)

test_loss_value, test_accuracy_value = model.evaluate(
    test_generator,
    verbose=1
)

print()
print(
    "Testing Loss:",
    f"{test_loss_value:.4f}"
)

print(
    "Testing Accuracy:",
    f"{test_accuracy_value * 100:.2f}%"
)


# ================================================================
# 27. TEST PREDICTIONS
# ================================================================

test_generator.reset()

predictions = model.predict(
    test_generator,
    verbose=1
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

true_classes = test_generator.classes


# ================================================================
# 28. CLASSIFICATION REPORT
# ================================================================

report = classification_report(

    true_classes,

    predicted_classes,

    target_names=CLASS_NAMES,

    digits=4
)

print()
print("=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

print(
    report
)


report_path = os.path.join(
    OUTPUT_DIR,
    "Classification_Report.txt"
)

with open(
    report_path,
    "w"
) as f:

    f.write(
        report
    )


# ================================================================
# 29. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(

    true_classes,

    predicted_classes
)


plt.figure(
    figsize=(8, 7)
)

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    xticklabels=CLASS_NAMES,

    yticklabels=CLASS_NAMES,

    cbar=False,

    annot_kws={
        "size": 14
    }
)

plt.xlabel(
    "Predicted Class",
    fontsize=15
)

plt.ylabel(
    "True Class",
    fontsize=15
)

plt.title(
    "CNN Confusion Matrix",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()


cm_path = os.path.join(
    OUTPUT_DIR,
    "CNN_Confusion_Matrix.png"
)

plt.savefig(
    cm_path,
    dpi=600,
    bbox_inches="tight"
)

plt.close()

plt.close()


# ================================================================
# 30. SAVE PREDICTIONS
# ================================================================

prediction_df = pd.DataFrame({

    "Image_Path":
        test_generator.filepaths,

    "True_Class":
        [
            CLASS_NAMES[i]
            for i in true_classes
        ],

    "Predicted_Class":
        [
            CLASS_NAMES[i]
            for i in predicted_classes
        ],

    "Prediction_Confidence":
        np.max(
            predictions,
            axis=1
        )

})


prediction_path = os.path.join(
    OUTPUT_DIR,
    "CNN_Test_Predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# ================================================================
# 31. FINAL OUTPUT SUMMARY
# ================================================================

print()
print("=" * 75)
print("PROCESS COMPLETED")
print("=" * 75)

print()
print("Models:")
print(
    "Best model:",
    checkpoint_path
)

print(
    "Final model:",
    final_model_path
)

print()
print("Graphs:")
print(
    "Accuracy:",
    accuracy_graph_path
)

print(
    "Loss:",
    loss_graph_path
)

print()
print("Reports:")
print(
    "Classification report:",
    report_path
)

print(
    "Confusion matrix:",
    cm_path
)

print(
    "Predictions:",
    prediction_path
)

print(
    "History CSV:",
    history_csv_path
)

print()
print("Final Testing Accuracy:")
print(
    f"{test_accuracy_value * 100:.2f}%"
)

print("=" * 75)