# ================================================================
# ELSARTA et al. [27] STYLE RESNET-50
# BURN DEGREE CLASSIFICATION
#
# OUTPUT:
#   1. Training Accuracy
#   2. Testing Accuracy
#   3. Training Loss
#   4. Testing Loss
#   5. Confusion Matrix
#   6. Classification Report
#   7. Test Predictions
#   8. Best ResNet-50 Model
#   9. Final ResNet-50 Model
#
# MODEL:
#   ResNet-50
#
# CLASSES:
#   First Degree
#   Second Degree
#   Third Degree
#
# EPOCHS:
#   100
#
# NOTE:
# This is an Elsarta-style implementation, not an exact
# reproduction of all paper-specific modifications.
# ================================================================


# ================================================================
# 1. IMPORT LIBRARIES
# ================================================================

import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.applications import ResNet50

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger
)

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import seaborn as sns


# ================================================================
# 2. LOCAL WORKSPACE PATHS (VS CODE)
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "skin")
OUTPUT_DIR = os.path.join(BASE_DIR, "Elsarta_ResNet50_100_Epoch")

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# 3. RANDOM SEED
# ================================================================

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)


# ================================================================
# 6. IMAGE CONFIGURATION
# ================================================================

IMG_SIZE = 224

BATCH_SIZE = 16

EPOCHS = 100

LEARNING_RATE = 0.001

DROPOUT_RATE = 0.1


# ================================================================
# 7. CLASS CONFIGURATION
# ================================================================

CLASS_NAMES = [
    "First_Degree",
    "Second_Degree",
    "Third_Degree"
]

NUM_CLASSES = len(
    CLASS_NAMES
)


# ================================================================
# 8. PRINT CONFIGURATION
# ================================================================

print("=" * 75)
print("ELSARTA et al. [27] STYLE RESNET-50")
print("BURN DEGREE CLASSIFICATION")
print("=" * 75)

print(
    "Dataset       :",
    DATASET_DIR
)

print(
    "Output        :",
    OUTPUT_DIR
)

print(
    "Image size    :",
    IMG_SIZE
)

print(
    "Batch size    :",
    BATCH_SIZE
)

print(
    "Epochs        :",
    EPOCHS
)

print(
    "Learning rate :",
    LEARNING_RATE
)

print(
    "Dropout rate  :",
    DROPOUT_RATE
)

print()

print(
    "Classes:"
)

for i, class_name in enumerate(
    CLASS_NAMES
):

    print(
        f"{i}: {class_name}"
    )

print("=" * 75)


# ================================================================
# 9. CHECK CLASS FOLDERS
# ================================================================

print()
print("=" * 75)
print("CHECKING DATASET")
print("=" * 75)

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
            f"{class_name:15s} : "
            f"{image_count} images"
        )

    else:

        missing_classes.append(
            class_name
        )

        print(
            f"{class_name:15s} : "
            "NOT FOUND"
        )


# ================================================================
# 10. STOP IF LABELS ARE NOT AVAILABLE
# ================================================================

if len(missing_classes) > 0:

    print()
    print("=" * 75)
    print("ERROR: CLASS LABELS NOT FOUND")
    print("=" * 75)

    print()
    print(
        "Required folders:"
    )

    for class_name in CLASS_NAMES:

        print(
            " -",
            class_name
        )

    print()
    print(
        "Expected structure:"
    )

    print(
        "skin/"
    )

    print(
        "├── First_Degree/"
    )

    print(
        "├── Second_Degree/"
    )

    print(
        "└── Third_Degree/"
    )

    print()
    print(
        "Supervised accuracy and loss require "
        "ground-truth labels."
    )

    print("=" * 75)

    raise SystemExit


# ================================================================
# 11. DATA AUGMENTATION
# ================================================================

train_datagen = ImageDataGenerator(

    rescale=1.0 / 255.0,

    rotation_range=15,

    width_shift_range=0.10,

    height_shift_range=0.10,

    zoom_range=0.15,

    shear_range=0.10,

    horizontal_flip=True,

    vertical_flip=False,

    brightness_range=(
        0.85,
        1.15
    ),

    validation_split=0.20
)


# ================================================================
# 12. TRAINING DATA GENERATOR
# ================================================================

train_generator = train_datagen.flow_from_directory(

    DATASET_DIR,

    target_size=(
        IMG_SIZE,
        IMG_SIZE
    ),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    classes=CLASS_NAMES,

    shuffle=True,

    seed=SEED,

    subset="training"
)


# ================================================================
# 13. TESTING DATA GENERATOR
# ================================================================

test_generator = train_datagen.flow_from_directory(

    DATASET_DIR,

    target_size=(
        IMG_SIZE,
        IMG_SIZE
    ),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    classes=CLASS_NAMES,

    shuffle=False,

    seed=SEED,

    subset="validation"
)


# ================================================================
# 14. DATASET INFORMATION
# ================================================================

print()
print("=" * 75)
print("DATASET INFORMATION")
print("=" * 75)

print(
    "Training images :",
    train_generator.samples
)

print(
    "Testing images  :",
    test_generator.samples
)

print(
    "Number classes  :",
    NUM_CLASSES
)

print(
    "Class indices   :",
    train_generator.class_indices
)

print("=" * 75)


# ================================================================
# 15. BUILD RESNET-50
# ================================================================

print()
print("=" * 75)
print("BUILDING RESNET-50")
print("=" * 75)


base_model = ResNet50(

    include_top=False,

    weights="imagenet",

    input_shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    )
)


# ================================================================
# 16. FREEZE PRETRAINED RESNET-50
# ================================================================

base_model.trainable = False


# ================================================================
# 17. RESNET-50 CLASSIFICATION HEAD
# ================================================================

inputs = layers.Input(
    shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    )
)


# ResNet-50 feature extraction
x = base_model(
    inputs,
    training=False
)


# Global Average Pooling
x = layers.GlobalAveragePooling2D()(
    x
)


# Dropout
x = layers.Dropout(
    DROPOUT_RATE
)(
    x
)


# Dense feature layer
x = layers.Dense(
    512,
    activation="relu"
)(
    x
)


# Batch normalization
x = layers.BatchNormalization()(
    x
)


# Dropout
x = layers.Dropout(
    DROPOUT_RATE
)(
    x
)


# Final classification layer
outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(
    x
)


# Create model
model = models.Model(
    inputs=inputs,
    outputs=outputs
)


# ================================================================
# 18. COMPILE MODEL
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
# 19. MODEL SUMMARY
# ================================================================

model.summary()


# ================================================================
# 20. CALLBACKS
# ================================================================

best_model_path = os.path.join(

    OUTPUT_DIR,

    "Elsarta_ResNet50_Best.keras"
)


checkpoint = ModelCheckpoint(

    best_model_path,

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

        "ResNet50_Training_Log.csv"
    ),

    append=False
)


# ================================================================
# 21. TRAIN RESNET-50
# ================================================================

print()
print("=" * 75)
print("STARTING RESNET-50 TRAINING")
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
# 22. SAVE FINAL MODEL
# ================================================================

final_model_path = os.path.join(

    OUTPUT_DIR,

    "Elsarta_ResNet50_Final.keras"
)


model.save(
    final_model_path
)


# ================================================================
# 23. SAVE TRAINING HISTORY
# ================================================================

history_df = pd.DataFrame(
    history.history
)


history_csv_path = os.path.join(

    OUTPUT_DIR,

    "ResNet50_Training_History.csv"
)


history_df.to_csv(

    history_csv_path,

    index=False
)


# ================================================================
# 24. EXTRACT ACCURACY
# ================================================================

train_accuracy = history.history[
    "accuracy"
]

test_accuracy = history.history[
    "val_accuracy"
]


# ================================================================
# 25. EXTRACT LOSS
# ================================================================

train_loss = history.history[
    "loss"
]

test_loss = history.history[
    "val_loss"
]


# ================================================================
# 26. FINAL RESULTS
# ================================================================

print()
print("=" * 75)
print("FINAL TRAINING RESULTS")
print("=" * 75)

print(
    "Final Training Accuracy :",
    f"{train_accuracy[-1] * 100:.2f}%"
)

print(
    "Final Testing Accuracy  :",
    f"{test_accuracy[-1] * 100:.2f}%"
)

print(
    "Final Training Loss     :",
    f"{train_loss[-1]:.4f}"
)

print(
    "Final Testing Loss      :",
    f"{test_loss[-1]:.4f}"
)


# ================================================================
# 27. BEST RESULTS
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
    "Best Training Accuracy :",
    f"{best_train_accuracy * 100:.2f}%"
)

print(
    "Best Training Epoch    :",
    best_train_epoch
)

print(
    "Best Testing Accuracy  :",
    f"{best_test_accuracy * 100:.2f}%"
)

print(
    "Best Testing Epoch     :",
    best_test_epoch
)


# ================================================================
# 28. EPOCH RANGE
# ================================================================

epochs_range = range(

    1,

    len(train_accuracy) + 1
)


# ================================================================
# 29. TRAINING / TESTING ACCURACY GRAPH
# ================================================================

plt.figure(
    figsize=(12, 7)
)


plt.plot(

    epochs_range,

    np.array(
        train_accuracy
    ) * 100,

    linewidth=2,

    label="Training Accuracy"
)


plt.plot(

    epochs_range,

    np.array(
        test_accuracy
    ) * 100,

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

    "ResNet-50 Training and Testing Accuracy",

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

    "ResNet50_Training_Testing_Accuracy.png"
)


plt.savefig(

    accuracy_graph_path,

    dpi=600,

    bbox_inches="tight"
)


plt.close()

plt.close()


# ================================================================
# 30. TRAINING / TESTING LOSS GRAPH
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

    "ResNet-50 Training and Testing Loss",

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

    "ResNet50_Training_Testing_Loss.png"
)


plt.savefig(

    loss_graph_path,

    dpi=600,

    bbox_inches="tight"
)


plt.close()

plt.close()


# ================================================================
# 31. EVALUATE TESTING DATA
# ================================================================

print()
print("=" * 75)
print("FINAL TESTING")
print("=" * 75)


test_generator.reset()


test_loss_value, test_accuracy_value = model.evaluate(

    test_generator,

    verbose=1
)


print()
print(
    "Testing Loss     :",
    f"{test_loss_value:.4f}"
)

print(
    "Testing Accuracy :",
    f"{test_accuracy_value * 100:.2f}%"
)


# ================================================================
# 32. GENERATE TEST PREDICTIONS
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
# 33. CLASSIFICATION REPORT
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

    "ResNet50_Classification_Report.txt"
)


with open(

    report_path,

    "w"

) as f:

    f.write(
        report
    )


# ================================================================
# 34. CONFUSION MATRIX
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

    "ResNet-50 Confusion Matrix",

    fontsize=18,

    fontweight="bold"
)


plt.tight_layout()


confusion_matrix_path = os.path.join(

    OUTPUT_DIR,

    "ResNet50_Confusion_Matrix.png"
)


plt.savefig(

    confusion_matrix_path,

    dpi=600,

    bbox_inches="tight"
)


plt.close()

plt.close()


# ================================================================
# 35. SAVE TEST PREDICTIONS
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

    "ResNet50_Test_Predictions.csv"
)


prediction_df.to_csv(

    prediction_path,

    index=False
)


# ================================================================
# 36. FINAL OUTPUT SUMMARY
# ================================================================

print()
print("=" * 75)
print("RESNET-50 PROCESS COMPLETED")
print("=" * 75)

print()

print("MODEL FILES")
print(
    "Best model  :",
    best_model_path
)

print(
    "Final model :",
    final_model_path
)

print()

print("ACCURACY / LOSS GRAPHS")

print(
    "Accuracy:",
    accuracy_graph_path
)

print(
    "Loss:",
    loss_graph_path
)

print()

print("EVALUATION")

print(
    "Classification report:",
    report_path
)

print(
    "Confusion matrix:",
    confusion_matrix_path
)

print(
    "Predictions:",
    prediction_path
)

print()

print(
    "Final Testing Accuracy:",
    f"{test_accuracy_value * 100:.2f}%"
)

print(
    "Final Testing Loss:",
    f"{test_loss_value:.4f}"
)

print()

print(
    "All results saved in:"
)

print(
    OUTPUT_DIR
)

print("=" * 75)