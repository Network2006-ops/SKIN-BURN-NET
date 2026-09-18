# SKIN BURN-NET: Skin Burn Detection, Segmentation, and Severity Assessment

A comprehensive deep learning framework for skin burn detection, lesion segmentation, feature extraction, and burn degree classification using modern computer vision architectures including custom **BURN-NET**, **EfficientNet-B7**, **VGG16**, and **ResNet-50**.

---

## 📌 Project Overview

This repository provides an end-to-end pipeline designed for clinical and automated skin burn analysis:
1. **Preprocessing & Enhancement**: Adaptive median blurring (AMB), color normalization, and contrast enhancement.
2. **Segmentation**: Lesion boundary delineation using GrabCut and contour extraction algorithms.
3. **Data Augmentation**: Geometric transformations and photometric augmentations to expand limited medical imaging datasets.
4. **Feature Extraction & Selection**: Deep representation learning and statistical feature selection.
5. **Model Architectures**:
   - **BURN-NET**: Custom neural network for skin burn feature reconstruction and classification.
   - **Abdolahnejad-Style EfficientNet-B7 & CNN**: State-of-the-art transfer learning and deep feature representations.
   - **VGG16 & ResNet-50**: Benchmarked architectures for transfer learning and comparative evaluation.
6. **Degree Estimation**: Severity assessment and burn degree categorization (1st, 2nd, and 3rd degree).
7. **Explainability**: Grad-CAM visualization for localization and interpretability.

---

## 📁 Repository Structure

```text
├── burn_net.py                           # BURN-NET model training, reconstruction, and evaluation
├── ABDOLAHNEJAD-STYLE EFFICIENTNET-B7.py  # EfficientNet-B7 deep transfer learning architecture
├── ABDOLAHNEJAD-STYLE CNN.py              # Custom CNN model based on Abdolahnejad methodology
├── CNN with VGG16.py                      # VGG16 transfer learning and evaluation pipeline
├── RESNET-50.py                           # ResNet-50 implementation
├── RESNET.py                              # Residual network baseline implementation
├── pre_processing.py                     # Image preprocessing & AMB filtering
├── segment.py                            # GrabCut skin lesion segmentation
├── augmentation.py                       # Image data augmentation pipeline
├── feature_extraction.py                 # Feature extraction from segmented regions
├── selection.py                          # Feature ranking and selection
├── degree.py                             # Burn degree calculation and categorization
├── grad.py                               # Grad-CAM interpretability and heatmaps
├── skin/                                 # Dataset images and annotation files
└── .gitignore                            # Git exclusion rules for large models and caches
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure Python 3.9+ is installed along with required packages:
```bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn tqdm
```

### 2. Running Preprocessing & Segmentation
```bash
python pre_processing.py
python segment.py
```

### 3. Training & Evaluating Models
To train and test the BURN-NET architecture:
```bash
python burn_net.py
```

To run the EfficientNet-B7 architecture:
```bash
python "ABDOLAHNEJAD-STYLE EFFICIENTNET-B7.py"
```

To evaluate with VGG16 or ResNet:
```bash
python "CNN with VGG16.py"
python "RESNET-50.py"
```

### 4. Severity Assessment & Grad-CAM
```bash
python degree.py
python grad.py
```

---

## 📊 Outputs & Artifacts
The training scripts generate performance metrics, confusion matrices, loss/accuracy curves, and visual heatmaps stored in their respective output directories (`gradcam_output`, `selected_features`, `feature_extraction_output`, etc.).
