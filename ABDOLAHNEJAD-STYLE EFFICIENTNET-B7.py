# ================================================================
# ABDOLAHNEJAD-STYLE EFFICIENTNET-B7
# 100-EPOCH UNSUPERVISED / SELF-SUPERVISED FEATURE LEARNING
#
# DATASET:
# skin
#
# TOTAL IMAGES:
# 1238
#
# NO LABEL CSV REQUIRED
#
# PROCESS:
# Image
#   ↓
# EfficientNet-B7
#   ↓
# Deep Feature Extraction
#   ↓
# Reconstruction Decoder
#   ↓
# 100 Epoch Training
#   ↓
# Feature Extraction
#   ↓
# Standardization
#   ↓
# PCA
#   ↓
# K-Means Clustering
#   ↓
# Visualization
#
# ================================================================


# ================================================================
# 1. IMPORT LIBRARIES
# ================================================================

import os
import gc
import random
import warnings

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.applications import EfficientNetB7

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger
)

from tensorflow.keras.utils import Sequence

# ================================================================
# 2. LOCAL WORKSPACE PATHS (VS CODE)
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "skin")

warnings.filterwarnings("ignore")


OUTPUT_DIR = os.path.join(

    BASE_DIR,

    "EfficientNetB7_Unsupervised_100_Epoch"

)


os.makedirs(

    OUTPUT_DIR,

    exist_ok=True

)


# ================================================================
# MODEL SETTINGS
# ================================================================

IMG_SIZE = 224

BATCH_SIZE = 8

EPOCHS = 100

LEARNING_RATE = 0.001

DROPOUT_RATE = 0.1

SEED = 42


# ================================================================
# DATA SPLIT
# ================================================================

VALIDATION_SIZE = 0.20


# ================================================================
# CLUSTER SETTINGS
# ================================================================

N_CLUSTERS = 4

PCA_COMPONENTS = 2


# ================================================================
# IMAGE EXTENSIONS
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
# OUTPUT FOLDERS
# ================================================================

GRAPH_DIR = os.path.join(

    OUTPUT_DIR,

    "Graphs"

)


FEATURE_DIR = os.path.join(

    OUTPUT_DIR,

    "Features"

)


CLUSTER_DIR = os.path.join(

    OUTPUT_DIR,

    "Clustering"

)


os.makedirs(

    GRAPH_DIR,

    exist_ok=True

)


os.makedirs(

    FEATURE_DIR,

    exist_ok=True

)


os.makedirs(

    CLUSTER_DIR,

    exist_ok=True

)


# ================================================================
# GENERATED FOLDERS TO IGNORE
# ================================================================

IGNORE_FOLDERS = {

    "EfficientNetB7_Unsupervised_100_Epoch",

    "Abdolahnejad_EfficientNetB7_100_Epoch",

    "BURN_NET_Training",

    "BURN_NET_Training_Testing",

    "Training_Testing_Graphs",

    "CNN_100_Epoch_Training_Testing",

    "VGG16_Training_Testing",

    "ResNet50_100_Epoch_Training_Testing",

    "ResNet101_Training_Testing",

    "GWO_Separate",

    "GWO_ShuffleNet_Output"

}


# ================================================================
# 4. RANDOM SEED
# ================================================================

random.seed(

    SEED

)

np.random.seed(

    SEED

)

tf.random.set_seed(

    SEED

)


# ================================================================
# 5. DISPLAY CONFIGURATION
# ================================================================

print("\n")

print("=" * 80)

print(
    "EFFICIENTNET-B7"
)

print(
    "100-EPOCH UNSUPERVISED FEATURE LEARNING"
)

print("=" * 80)

print(
    "Dataset Path       :",
    DATASET_PATH
)

print(
    "Output Path        :",
    OUTPUT_DIR
)

print(
    "Image Size         :",
    IMG_SIZE
)

print(
    "Batch Size         :",
    BATCH_SIZE
)

print(
    "Epochs             :",
    EPOCHS
)

print(
    "Learning Rate      :",
    LEARNING_RATE
)

print(
    "Dropout Rate       :",
    DROPOUT_RATE
)

print(
    "Validation         :",
    "20%"
)

print(
    "K-Means Clusters   :",
    N_CLUSTERS
)

print("=" * 80)


# ================================================================
# 6. GPU CHECK
# ================================================================

print("\n")

print("=" * 80)

print(
    "HARDWARE CHECK"
)

print("=" * 80)


gpus = tf.config.list_physical_devices(

    "GPU"

)


if len(gpus) > 0:

    print(

        "GPU detected:",

        gpus

    )

else:

    print(

        "No GPU detected."

    )

    print(

        "Training will use CPU."

    )


# ================================================================
# 7. SEARCH FOR IMAGE FILES
# ================================================================

print("\n")

print("=" * 80)

print(
    "SEARCHING FOR IMAGE FILES"
)

print("=" * 80)


image_paths = []


for root, dirs, files in os.walk(

    DATASET_PATH

):


    dirs[:] = [

        d

        for d in dirs

        if d not in IGNORE_FOLDERS

    ]


    for filename in files:


        if filename.lower().endswith(

            IMAGE_EXTENSIONS

        ):


            full_path = os.path.join(

                root,

                filename

            )


            image_paths.append(

                full_path

            )


# Remove duplicates

image_paths = sorted(

    list(

        set(

            image_paths

        )

    )

)


print(

    "\nTotal image files found:",

    len(image_paths)

)


if len(image_paths) == 0:

    raise ValueError(

        "No image files were found."

    )


# ================================================================
# 8. SAVE IMAGE LIST
# ================================================================

image_list_df = pd.DataFrame({

    "Image_Path":

        image_paths

})


image_list_df["Image_Name"] = [

    os.path.basename(

        p

    )

    for p in image_paths

]


image_list_path = os.path.join(

    OUTPUT_DIR,

    "All_Image_List.csv"

)


image_list_df.to_csv(

    image_list_path,

    index=False

)


# ================================================================
# 9. TRAIN / VALIDATION SPLIT
# ================================================================

print("\n")

print("=" * 80)

print(
    "CREATING TRAINING / VALIDATION SPLIT"
)

print("=" * 80)


train_paths, val_paths = train_test_split(

    image_paths,

    test_size=VALIDATION_SIZE,

    random_state=SEED,

    shuffle=True

)


train_paths = sorted(

    train_paths

)


val_paths = sorted(

    val_paths

)


print(

    "Total images     :",

    len(image_paths)

)

print(

    "Training images  :",

    len(train_paths)

)

print(

    "Validation images:",

    len(val_paths)

)


# ================================================================
# 10. SAVE SPLITS
# ================================================================

train_split_df = pd.DataFrame({

    "Image_Path":

        train_paths

})


val_split_df = pd.DataFrame({

    "Image_Path":

        val_paths

})


train_split_df.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "Training_Set.csv"

    ),

    index=False

)


val_split_df.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "Validation_Set.csv"

    ),

    index=False

)


# ================================================================
# 11. DATA GENERATOR
# ================================================================

class ImageReconstructionGenerator(

    Sequence

):


    def __init__(

        self,

        image_paths,

        batch_size,

        img_size,

        shuffle=True

    ):

        super().__init__()


        self.image_paths = list(

            image_paths

        )


        self.batch_size = (

            batch_size

        )


        self.img_size = (

            img_size

        )


        self.shuffle = (

            shuffle

        )


        self.indices = np.arange(

            len(

                self.image_paths

            )

        )


        self.on_epoch_end()


    def __len__(

        self

    ):

        return int(

            np.ceil(

                len(

                    self.image_paths

                )

                /

                self.batch_size

            )

        )


    def __getitem__(

        self,

        index

    ):


        batch_indices = (

            self.indices[

                index *

                self.batch_size:

                (

                    index + 1

                ) *

                self.batch_size

            ]

        )


        batch_paths = [

            self.image_paths[i]

            for i in batch_indices

        ]


        X = np.zeros(

            (

                len(batch_paths),

                self.img_size,

                self.img_size,

                3

            ),

            dtype=np.float32

        )


        for i, image_path in enumerate(

            batch_paths

        ):


            image = cv2.imread(

                image_path

            )


            if image is None:

                raise ValueError(

                    "Could not read image:\n"

                    + image_path

                )


            image = cv2.cvtColor(

                image,

                cv2.COLOR_BGR2RGB

            )


            image = cv2.resize(

                image,

                (

                    self.img_size,

                    self.img_size

                ),

                interpolation=

                cv2.INTER_AREA

            )


            image = (

                image.astype(

                    np.float32

                )

                /

                255.0

            )


            X[i] = image


        # Reconstruction target = original image

        return X, X


    def on_epoch_end(

        self

    ):


        if self.shuffle:

            np.random.shuffle(

                self.indices

            )


# ================================================================
# 12. CREATE DATA GENERATORS
# ================================================================

train_generator = ImageReconstructionGenerator(

    train_paths,

    BATCH_SIZE,

    IMG_SIZE,

    shuffle=True

)


validation_generator = ImageReconstructionGenerator(

    val_paths,

    BATCH_SIZE,

    IMG_SIZE,

    shuffle=False

)


print("\n")

print(

    "Training batches:",

    len(train_generator)

)


print(

    "Validation batches:",

    len(validation_generator)

)


# ================================================================
# 13. BUILD EFFICIENTNET-B7 BACKBONE
# ================================================================

print("\n")

print("=" * 80)

print(
    "BUILDING EFFICIENTNET-B7"
)

print("=" * 80)


try:

    base_model = EfficientNetB7(

        include_top=False,

        weights="imagenet",

        input_shape=(

            IMG_SIZE,

            IMG_SIZE,

            3

        )

    )


    print(

        "ImageNet pretrained EfficientNet-B7 loaded."

    )


except Exception as e:

    print("\n")

    print(

        "ImageNet weights could not be loaded."

    )

    print(

        "Reason:",

        str(e)

    )

    print("\n")

    print(

        "Using EfficientNet-B7 with random initialization."

    )


    base_model = EfficientNetB7(

        include_top=False,

        weights=None,

        input_shape=(

            IMG_SIZE,

            IMG_SIZE,

            3

        )

    )


# ================================================================
# 14. FREEZE BACKBONE
# ================================================================

base_model.trainable = False


print(

    "\nEfficientNet-B7 backbone frozen."

)


# ================================================================
# 15. BUILD RECONSTRUCTION DECODER
# ================================================================

print("\n")

print("=" * 80)

print(
    "BUILDING RECONSTRUCTION MODEL"
)

print("=" * 80)


inputs = layers.Input(

    shape=(

        IMG_SIZE,

        IMG_SIZE,

        3

    ),

    name="Input_Image"

)


# ------------------------------------------------
# EfficientNet feature extraction
# ------------------------------------------------

features = base_model(

    inputs,

    training=False

)


# ------------------------------------------------
# Bottleneck
# ------------------------------------------------

x = layers.GlobalAveragePooling2D(

    name="Global_Average_Pooling"

)(features)


x = layers.Dense(

    1024,

    activation="relu",

    name="Feature_Bottleneck"

)(x)


x = layers.Dropout(

    DROPOUT_RATE

)(x)


# ------------------------------------------------
# Expand feature vector
# ------------------------------------------------

x = layers.Dense(

    7 * 7 * 256,

    activation="relu"

)(x)


x = layers.Reshape(

    (

        7,

        7,

        256

    )

)(x)


# ------------------------------------------------
# Decoder block 1
# 7 x 7 -> 14 x 14
# ------------------------------------------------

x = layers.UpSampling2D(

    size=(2, 2)

)(x)


x = layers.Conv2D(

    256,

    kernel_size=3,

    padding="same",

    activation="relu"

)(x)


x = layers.BatchNormalization()(x)


# ------------------------------------------------
# Decoder block 2
# 14 x 14 -> 28 x 28
# ------------------------------------------------

x = layers.UpSampling2D(

    size=(2, 2)

)(x)


x = layers.Conv2D(

    128,

    kernel_size=3,

    padding="same",

    activation="relu"

)(x)


x = layers.BatchNormalization()(x)


# ------------------------------------------------
# Decoder block 3
# 28 x 28 -> 56 x 56
# ------------------------------------------------

x = layers.UpSampling2D(

    size=(2, 2)

)(x)


x = layers.Conv2D(

    64,

    kernel_size=3,

    padding="same",

    activation="relu"

)(x)


x = layers.BatchNormalization()(x)


# ------------------------------------------------
# Decoder block 4
# 56 x 56 -> 112 x 112
# ------------------------------------------------

x = layers.UpSampling2D(

    size=(2, 2)

)(x)


x = layers.Conv2D(

    32,

    kernel_size=3,

    padding="same",

    activation="relu"

)(x)


x = layers.BatchNormalization()(x)


# ------------------------------------------------
# Decoder block 5
# 112 x 112 -> 224 x 224
# ------------------------------------------------

x = layers.UpSampling2D(

    size=(2, 2)

)(x)


x = layers.Conv2D(

    16,

    kernel_size=3,

    padding="same",

    activation="relu"

)(x)


# ------------------------------------------------
# Reconstruction output
# ------------------------------------------------

outputs = layers.Conv2D(

    3,

    kernel_size=3,

    padding="same",

    activation="sigmoid",

    name="Reconstructed_Image"

)(x)


# ================================================================
# 16. CREATE MODEL
# ================================================================

autoencoder = models.Model(

    inputs,

    outputs,

    name=

    "EfficientNetB7_Unsupervised_Autoencoder"

)


# ================================================================
# 17. MODEL SUMMARY
# ================================================================

autoencoder.summary()


# ================================================================
# 18. COMPILE
# ================================================================

autoencoder.compile(

    optimizer=Adam(

        learning_rate=

        LEARNING_RATE

    ),

    loss="mse"

)


# ================================================================
# 19. SAVE MODEL SUMMARY
# ================================================================

model_summary_path = os.path.join(

    OUTPUT_DIR,

    "EfficientNetB7_Model_Summary.txt"

)


with open(

    model_summary_path,

    "w"

) as f:


    autoencoder.summary(

        print_fn=f.write

    )


# ================================================================
# 20. CALLBACKS
# ================================================================

best_model_path = os.path.join(

    OUTPUT_DIR,

    "EfficientNetB7_Unsupervised_Best.keras"

)


final_model_path = os.path.join(

    OUTPUT_DIR,

    "EfficientNetB7_Unsupervised_Final.keras"

)


history_csv_path = os.path.join(

    OUTPUT_DIR,

    "Training_History.csv"

)


checkpoint = ModelCheckpoint(

    best_model_path,

    monitor="val_loss",

    mode="min",

    save_best_only=True,

    verbose=1

)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=8,

    min_lr=1e-7,

    verbose=1

)


csv_logger = CSVLogger(

    history_csv_path,

    append=False

)


callbacks = [

    checkpoint,

    reduce_lr,

    csv_logger

]


# ================================================================
# 21. TRAIN FOR 100 EPOCHS
# ================================================================

print("\n")

print("=" * 80)

print(
    "STARTING 100-EPOCH TRAINING"
)

print("=" * 80)


history = autoencoder.fit(

    train_generator,

    validation_data=

    validation_generator,

    epochs=EPOCHS,

    callbacks=callbacks,

    verbose=1

)


# ================================================================
# 22. SAVE FINAL MODEL
# ================================================================

autoencoder.save(

    final_model_path

)


# ================================================================
# 23. SAVE HISTORY
# ================================================================

history_df = pd.DataFrame(

    history.history

)


history_df.insert(

    0,

    "Epoch",

    np.arange(

        1,

        len(history_df) + 1

    )

)


history_df.to_csv(

    history_csv_path,

    index=False

)


# ================================================================
# 24. TRAINING LOSS
# ================================================================

epochs_array = np.arange(

    1,

    len(

        history.history["loss"]

    ) + 1

)


training_loss = np.array(

    history.history["loss"]

)


validation_loss = np.array(

    history.history["val_loss"]

)


# ================================================================
# 25. LOSS GRAPH
# ================================================================

plt.figure(

    figsize=(12, 7)

)


plt.plot(

    epochs_array,

    training_loss,

    linewidth=2.5,

    label="Training Loss"

)


plt.plot(

    epochs_array,

    validation_loss,

    linewidth=2.5,

    label="Validation Loss"

)


plt.xlabel(

    "Epoch",

    fontsize=16,

    fontweight="bold"

)


plt.ylabel(

    "Mean Squared Error (MSE)",

    fontsize=16,

    fontweight="bold"

)


plt.title(

    "EfficientNet-B7 100-Epoch Training and Validation Loss",

    fontsize=17,

    fontweight="bold"

)


plt.xticks(

    fontsize=13

)


plt.yticks(

    fontsize=13

)


plt.xlim(

    1,

    EPOCHS

)


plt.grid(

    True,

    linestyle="--",

    alpha=0.35

)


plt.legend(

    fontsize=13

)


plt.tight_layout()


loss_graph_path = os.path.join(

    GRAPH_DIR,

    "EfficientNetB7_100_Epoch_Training_Validation_Loss.png"

)


plt.savefig(

    loss_graph_path,

    dpi=600,

    bbox_inches="tight"

)


plt.close()

plt.close()


# ================================================================
# 26. LOG LOSS GRAPH
# ================================================================

plt.figure(

    figsize=(12, 7)

)


plt.semilogy(

    epochs_array,

    training_loss,

    linewidth=2.5,

    label="Training Loss"

)


plt.semilogy(

    epochs_array,

    validation_loss,

    linewidth=2.5,

    label="Validation Loss"

)


plt.xlabel(

    "Epoch",

    fontsize=16,

    fontweight="bold"

)


plt.ylabel(

    "Loss - Log Scale",

    fontsize=16,

    fontweight="bold"

)


plt.title(

    "EfficientNet-B7 Training and Validation Loss - Log Scale",

    fontsize=17,

    fontweight="bold"

)


plt.grid(

    True,

    linestyle="--",

    alpha=0.35

)


plt.legend(

    fontsize=13

)


plt.tight_layout()


log_loss_path = os.path.join(

    GRAPH_DIR,

    "EfficientNetB7_Training_Validation_Log_Loss.png"

)


plt.savefig(

    log_loss_path,

    dpi=600,

    bbox_inches="tight"

)


plt.close()

plt.close()


# ================================================================
# 27. BEST EPOCH
# ================================================================

best_epoch = (

    np.argmin(

        validation_loss

    )

    + 1

)


best_training_loss = (

    training_loss[

        best_epoch - 1

    ]

)


best_validation_loss = (

    validation_loss[

        best_epoch - 1

    ]

)


print("\n")

print("=" * 80)

print(
    "BEST VALIDATION RESULT"
)

print("=" * 80)


print(

    "Best Epoch:",

    best_epoch

)


print(

    f"Training Loss at Best Epoch: "
    f"{best_training_loss:.8f}"

)


print(

    f"Validation Loss at Best Epoch: "
    f"{best_validation_loss:.8f}"

)


# ================================================================
# 28. LOAD BEST MODEL
# ================================================================

print("\n")

print(
    "Loading best EfficientNet-B7 model..."
)


best_model = tf.keras.models.load_model(

    best_model_path

)


# ================================================================
# 29. CREATE FEATURE EXTRACTOR
# ================================================================

print("\n")

print("=" * 80)

print(
    "CREATING FEATURE EXTRACTOR"
)

print("=" * 80)


feature_extractor = models.Model(

    inputs=best_model.input,

    outputs=best_model.get_layer(

        "Feature_Bottleneck"

    ).output

)


print(

    "Feature extractor created."

)


# ================================================================
# 30. FEATURE EXTRACTION GENERATOR
# ================================================================

class FeatureImageGenerator(

    Sequence

):


    def __init__(

        self,

        image_paths,

        batch_size,

        img_size

    ):

        super().__init__()


        self.image_paths = list(

            image_paths

        )


        self.batch_size = (

            batch_size

        )


        self.img_size = (

            img_size

        )


    def __len__(

        self

    ):

        return int(

            np.ceil(

                len(

                    self.image_paths

                )

                /

                self.batch_size

            )

        )


    def __getitem__(

        self,

        index

    ):


        batch_paths = (

            self.image_paths[

                index *

                self.batch_size:

                (

                    index + 1

                ) *

                self.batch_size

            ]

        )


        X = np.zeros(

            (

                len(batch_paths),

                self.img_size,

                self.img_size,

                3

            ),

            dtype=np.float32

        )


        for i, image_path in enumerate(

            batch_paths

        ):


            image = cv2.imread(

                image_path

            )


            if image is None:

                raise ValueError(

                    "Could not read image:\n"

                    + image_path

                )


            image = cv2.cvtColor(

                image,

                cv2.COLOR_BGR2RGB

            )


            image = cv2.resize(

                image,

                (

                    self.img_size,

                    self.img_size

                ),

                interpolation=

                cv2.INTER_AREA

            )


            image = (

                image.astype(

                    np.float32

                )

                /

                255.0

            )


            X[i] = image


        return X


# ================================================================
# 31. CREATE FEATURE GENERATOR
# ================================================================

all_feature_generator = FeatureImageGenerator(

    image_paths,

    BATCH_SIZE,

    IMG_SIZE

)


# ================================================================
# 32. EXTRACT FEATURES
# ================================================================

print("\n")

print("=" * 80)

print(
    "EXTRACTING EFFICIENTNET-B7 FEATURES"
)

print("=" * 80)


features = feature_extractor.predict(

    all_feature_generator,

    verbose=1

)


print("\n")

print(

    "Feature matrix shape:",

    features.shape

)


# ================================================================
# 33. SAVE RAW FEATURES
# ================================================================

raw_feature_path = os.path.join(

    FEATURE_DIR,

    "EfficientNetB7_Raw_Features.npy"

)


np.save(

    raw_feature_path,

    features

)


print(

    "Raw features saved:"

)

print(

    raw_feature_path

)


# ================================================================
# 34. FLATTEN FEATURES
# ================================================================

features_2d = features.reshape(

    features.shape[0],

    -1

)


print(

    "\nFlattened feature shape:",

    features_2d.shape

)


# ================================================================
# 35. SAVE FLATTENED FEATURES
# ================================================================

flattened_feature_path = os.path.join(

    FEATURE_DIR,

    "EfficientNetB7_Flattened_Features.npy"

)


np.save(

    flattened_feature_path,

    features_2d

)


# ================================================================
# 36. STANDARDIZE FEATURES
# ================================================================

print("\n")

print("=" * 80)

print(
    "STANDARDIZING FEATURES"
)

print("=" * 80)


scaler = StandardScaler()


scaled_features = scaler.fit_transform(

    features_2d

)


print(

    "Standardized feature shape:",

    scaled_features.shape

)


# ================================================================
# 37. SAVE STANDARDIZED FEATURES
# ================================================================

scaled_feature_path = os.path.join(

    FEATURE_DIR,

    "EfficientNetB7_Standardized_Features.npy"

)


np.save(

    scaled_feature_path,

    scaled_features

)


# ================================================================
# 38. PCA
# ================================================================

print("\n")

print("=" * 80)

print(
    "PCA FEATURE REDUCTION"
)

print("=" * 80)


pca = PCA(

    n_components=PCA_COMPONENTS,

    random_state=SEED

)


pca_features = pca.fit_transform(

    scaled_features

)


print(

    "PCA shape:",

    pca_features.shape

)


print(

    "PC1 explained variance:",

    f"{pca.explained_variance_ratio_[0] * 100:.4f}%"

)


print(

    "PC2 explained variance:",

    f"{pca.explained_variance_ratio_[1] * 100:.4f}%"

)


print(

    "Total explained variance:",

    f"{sum(pca.explained_variance_ratio_) * 100:.4f}%"

)


# ================================================================
# 39. SAVE PCA
# ================================================================

pca_path = os.path.join(

    FEATURE_DIR,

    "EfficientNetB7_PCA_2D.npy"

)


np.save(

    pca_path,

    pca_features

)


# ================================================================
# 40. PCA VISUALIZATION BEFORE CLUSTERING
# ================================================================

plt.figure(

    figsize=(11, 8)

)


plt.scatter(

    pca_features[:, 0],

    pca_features[:, 1],

    s=35,

    alpha=0.75

)


plt.xlabel(

    "Principal Component 1",

    fontsize=15,

    fontweight="bold"

)


plt.ylabel(

    "Principal Component 2",

    fontsize=15,

    fontweight="bold"

)


plt.title(

    "EfficientNet-B7 Feature Distribution Using PCA",

    fontsize=17,

    fontweight="bold"

)


plt.grid(

    True,

    linestyle="--",

    alpha=0.3

)


plt.tight_layout()


pca_plot_path = os.path.join(

    GRAPH_DIR,

    "EfficientNetB7_PCA_Feature_Distribution.png"

)


plt.savefig(

    pca_plot_path,

    dpi=600,

    bbox_inches="tight"

)


plt.close()

plt.close()


# ================================================================
# 41. K-MEANS CLUSTERING
# ================================================================

print("\n")

print("=" * 80)

print(
    "K-MEANS CLUSTERING"
)

print("=" * 80)


print(

    "Number of clusters:",

    N_CLUSTERS

)


kmeans = KMeans(

    n_clusters=N_CLUSTERS,

    random_state=SEED,

    n_init=20,

    max_iter=500

)


cluster_labels = kmeans.fit_predict(

    scaled_features

)


# ================================================================
# 42. CLUSTER COUNTS
# ================================================================

cluster_counts = pd.Series(

    cluster_labels

).value_counts().sort_index()


print("\n")

print(
    "Cluster distribution:"
)

for cluster_id, count in (

    cluster_counts.items()

):

    print(

        f"Cluster {cluster_id}: "
        f"{count} images"

    )


# ================================================================
# 43. SILHOUETTE SCORE
# ================================================================

if len(

    np.unique(

        cluster_labels

    )

) > 1:


    silhouette = silhouette_score(

        scaled_features,

        cluster_labels,

        sample_size=min(

            2000,

            len(scaled_features)

        ),

        random_state=SEED

    )


else:

    silhouette = np.nan


print("\n")

print(

    f"Silhouette Score: "
    f"{silhouette:.6f}"

)


# ================================================================
# 44. PCA + CLUSTER VISUALIZATION
# ================================================================

plt.figure(

    figsize=(12, 9)

)


for cluster_id in range(

    N_CLUSTERS

):


    mask = (

        cluster_labels

        ==

        cluster_id

    )


    plt.scatter(

        pca_features[mask, 0],

        pca_features[mask, 1],

        s=45,

        alpha=0.75,

        label=

        f"Cluster {cluster_id + 1}"

    )


plt.xlabel(

    "Principal Component 1",

    fontsize=15,

    fontweight="bold"

)


plt.ylabel(

    "Principal Component 2",

    fontsize=15,

    fontweight="bold"

)


plt.title(

    "EfficientNet-B7 Feature Clustering",

    fontsize=18,

    fontweight="bold"

)


plt.grid(

    True,

    linestyle="--",

    alpha=0.30

)


plt.legend(

    fontsize=12

)


plt.tight_layout()


cluster_plot_path = os.path.join(

    GRAPH_DIR,

    "EfficientNetB7_KMeans_Cluster_Visualization.png"

)


plt.savefig(

    cluster_plot_path,

    dpi=600,

    bbox_inches="tight"

)


plt.close()

plt.close()


# ================================================================
# 45. CLUSTER CENTERS IN PCA SPACE
# ================================================================

cluster_centers_scaled = (

    kmeans.cluster_centers_

)


cluster_centers_pca = pca.transform(

    cluster_centers_scaled

)


plt.figure(

    figsize=(12, 9)

)


for cluster_id in range(

    N_CLUSTERS

):


    mask = (

        cluster_labels

        ==

        cluster_id

    )


    plt.scatter(

        pca_features[mask, 0],

        pca_features[mask, 1],

        s=40,

        alpha=0.65,

        label=

        f"Cluster {cluster_id + 1}"

    )


    plt.scatter(

        cluster_centers_pca[

            cluster_id,

            0

        ],

        cluster_centers_pca[

            cluster_id,

            1

        ],

        marker="X",

        s=250,

        edgecolors="black",

        linewidths=1.5

    )


plt.xlabel(

    "Principal Component 1",

    fontsize=15,

    fontweight="bold"

)


plt.ylabel(

    "Principal Component 2",

    fontsize=15,

    fontweight="bold"

)


plt.title(

    "EfficientNet-B7 K-Means Clusters with Cluster Centers",

    fontsize=17,

    fontweight="bold"

)


plt.grid(

    True,

    linestyle="--",

    alpha=0.30

)


plt.legend(

    fontsize=12

)


plt.tight_layout()


cluster_center_path = os.path.join(

    GRAPH_DIR,

    "EfficientNetB7_KMeans_Cluster_Centers.png"

)


plt.savefig(

    cluster_center_path,

    dpi=600,

    bbox_inches="tight"

)


plt.close()

plt.close()


# ================================================================
# 46. CREATE CLUSTER DATAFRAME
# ================================================================

cluster_df = pd.DataFrame({

    "Image_Path":

        image_paths,

    "Image_Name": [

        os.path.basename(

            p

        )

        for p in image_paths

    ],

    "Cluster_ID":

        cluster_labels + 1,

    "PCA_1":

        pca_features[:, 0],

    "PCA_2":

        pca_features[:, 1]

})


# ================================================================
# 47. DISTANCE FROM CLUSTER CENTER
# ================================================================

distances = kmeans.transform(

    scaled_features

)


minimum_distances = np.min(

    distances,

    axis=1

)


cluster_df[

    "Distance_From_Cluster_Center"

] = minimum_distances


# ================================================================
# 48. SAVE CLUSTER ASSIGNMENTS
# ================================================================

cluster_csv_path = os.path.join(

    CLUSTER_DIR,

    "EfficientNetB7_KMeans_Cluster_Assignments.csv"

)


cluster_df.to_csv(

    cluster_csv_path,

    index=False

)


print("\n")

print(

    "Cluster assignments saved:"

)

print(

    cluster_csv_path

)


# ================================================================
# 49. SAVE CLUSTER SUMMARY
# ================================================================

cluster_summary = (

    cluster_df

    .groupby(

        "Cluster_ID"

    )

    .agg(

        Number_of_Images=(

            "Image_Name",

            "count"

        ),

        Mean_Distance=(

            "Distance_From_Cluster_Center",

            "mean"

        ),

        Mean_PCA1=(

            "PCA_1",

            "mean"

        ),

        Mean_PCA2=(

            "PCA_2",

            "mean"

        )

    )

    .reset_index()

)


cluster_summary_path = os.path.join(

    CLUSTER_DIR,

    "EfficientNetB7_Cluster_Summary.csv"

)


cluster_summary.to_csv(

    cluster_summary_path,

    index=False

)


print(

    "Cluster summary saved:"

)

print(

    cluster_summary_path

)


# ================================================================
# 50. SAVE CLUSTER CENTER DATA
# ================================================================

center_df = pd.DataFrame({

    "Cluster_ID":

        np.arange(

            1,

            N_CLUSTERS + 1

        ),

    "PCA_1":

        cluster_centers_pca[:, 0],

    "PCA_2":

        cluster_centers_pca[:, 1]

})


center_path = os.path.join(

    CLUSTER_DIR,

    "EfficientNetB7_Cluster_Centers_PCA.csv"

)


center_df.to_csv(

    center_path,

    index=False

)


# ================================================================
# 51. FEATURE MATRIX CSV
# ================================================================

# Save PCA features together with image names

pca_df = pd.DataFrame({

    "Image_Path":

        image_paths,

    "Image_Name": [

        os.path.basename(

            p

        )

        for p in image_paths

    ],

    "PC1":

        pca_features[:, 0],

    "PC2":

        pca_features[:, 1],

    "Cluster_ID":

        cluster_labels + 1

})


pca_csv_path = os.path.join(

    FEATURE_DIR,

    "EfficientNetB7_PCA_Features.csv"

)


pca_df.to_csv(

    pca_csv_path,

    index=False

)


# ================================================================
# 52. CLUSTER DISTRIBUTION GRAPH
# ================================================================

plt.figure(

    figsize=(10, 7)

)


cluster_ids = np.arange(

    1,

    N_CLUSTERS + 1

)


counts = [

    np.sum(

        cluster_labels == i

    )

    for i in range(

        N_CLUSTERS

    )

]


plt.bar(

    cluster_ids,

    counts

)


plt.xlabel(

    "Cluster ID",

    fontsize=15,

    fontweight="bold"

)


plt.ylabel(

    "Number of Images",

    fontsize=15,

    fontweight="bold"

)


plt.title(

    "EfficientNet-B7 Cluster Distribution",

    fontsize=17,

    fontweight="bold"

)


plt.xticks(

    cluster_ids

)


plt.grid(

    axis="y",

    linestyle="--",

    alpha=0.30

)


plt.tight_layout()


cluster_distribution_path = os.path.join(

    GRAPH_DIR,

    "EfficientNetB7_Cluster_Distribution.png"

)


plt.savefig(

    cluster_distribution_path,

    dpi=600,

    bbox_inches="tight"

)


plt.close()

plt.close()


# ================================================================
# 53. SAVE CLUSTER METRICS
# ================================================================

cluster_metrics_df = pd.DataFrame({

    "Number_of_Clusters": [

        N_CLUSTERS

    ],

    "Number_of_Images": [

        len(image_paths)

    ],

    "PCA_Components": [

        PCA_COMPONENTS

    ],

    "PC1_Explained_Variance_Percent": [

        pca.explained_variance_ratio_[0] * 100

    ],

    "PC2_Explained_Variance_Percent": [

        pca.explained_variance_ratio_[1] * 100

    ],

    "Total_PCA_Explained_Variance_Percent": [

        sum(

            pca.explained_variance_ratio_

        ) * 100

    ],

    "Silhouette_Score": [

        silhouette

    ]

})


cluster_metrics_path = os.path.join(

    CLUSTER_DIR,

    "EfficientNetB7_Clustering_Metrics.csv"

)


cluster_metrics_df.to_csv(

    cluster_metrics_path,

    index=False

)


# ================================================================
# 54. SAVE K-MEANS MODEL
# ================================================================

import joblib


kmeans_path = os.path.join(

    CLUSTER_DIR,

    "EfficientNetB7_KMeans_Model.pkl"

)


joblib.dump(

    kmeans,

    kmeans_path

)


# ================================================================
# 55. SAVE PCA MODEL
# ================================================================

pca_model_path = os.path.join(

    FEATURE_DIR,

    "EfficientNetB7_PCA_Model.pkl"

)


joblib.dump(

    pca,

    pca_model_path

)


# ================================================================
# 56. SAVE STANDARD SCALER
# ================================================================

scaler_path = os.path.join(

    FEATURE_DIR,

    "EfficientNetB7_StandardScaler.pkl"

)


joblib.dump(

    scaler,

    scaler_path

)


# ================================================================
# 57. FINAL TRAINING / CLUSTERING SUMMARY
# ================================================================

print("\n")

print("=" * 80)

print(
    "FINAL RESULTS"
)

print("=" * 80)


print(

    "Total Images              :",

    len(image_paths)

)


print(

    "Training Images           :",

    len(train_paths)

)


print(

    "Validation Images         :",

    len(val_paths)

)


print(

    "Training Epochs           :",

    EPOCHS

)


print(

    "Best Epoch                :",

    best_epoch

)


print(

    "Best Training Loss        :",

    f"{best_training_loss:.8f}"

)


print(

    "Best Validation Loss      :",

    f"{best_validation_loss:.8f}"

)


print(

    "Feature Dimension         :",

    features_2d.shape[1]

)


print(

    "PCA Dimension             :",

    PCA_COMPONENTS

)


print(

    "K-Means Clusters          :",

    N_CLUSTERS

)


print(

    "Silhouette Score          :",

    f"{silhouette:.6f}"

)


# ================================================================
# 58. DISPLAY CLUSTER SUMMARY
# ================================================================

print("\n")

print("=" * 80)

print(
    "CLUSTER SUMMARY"
)

print("=" * 80)


print(

    cluster_summary.to_string(

        index=False

    )

)


# ================================================================
# 59. OUTPUT FILE LIST
# ================================================================

print("\n")

print("=" * 80)

print(
    "OUTPUT FILES"
)

print("=" * 80)


print("\nMODEL FILES")

print(

    best_model_path

)

print(

    final_model_path

)


print("\nTRAINING FILES")

print(

    history_csv_path

)


print(

    loss_graph_path

)


print(

    log_loss_path

)


print("\nFEATURE FILES")

print(

    raw_feature_path

)


print(

    flattened_feature_path

)


print(

    scaled_feature_path

)


print(

    pca_path

)


print(

    pca_csv_path

)


print("\nCLUSTER FILES")

print(

    cluster_csv_path

)


print(

    cluster_summary_path

)


print(

    center_path

)


print(

    cluster_metrics_path

)


print(

    kmeans_path

)


print("\nGRAPH FILES")

print(

    pca_plot_path

)


print(

    cluster_plot_path

)


print(

    cluster_center_path

)


print(

    cluster_distribution_path

)


# ================================================================
# 60. MEMORY CLEANUP
# ================================================================

gc.collect()

tf.keras.backend.clear_session()


print("\n")

print("=" * 80)

print(
    "EFFICIENTNET-B7 100-EPOCH PIPELINE COMPLETED"
)

print("=" * 80)

print("\n")

print(
    "The clusters are unsupervised groups."
)

print(
    "They must NOT be interpreted as SPF/SPT/DPT/FT "
    "without genuine burn-severity labels."
)

print("=" * 80)