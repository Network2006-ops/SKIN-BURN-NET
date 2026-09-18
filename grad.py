# ============================================================
# FINAL FULL CODE: GRAD-CAM (ERROR FIXED + BIG OUTPUT + SAVE)
# ============================================================

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import os

# ============================================================
# IMAGE PATHS
# ============================================================

image_paths = [
    "d:/Skin/skin_AMB_Preprocessed/unnamed (1).png",
    "d:/Skin/skin_AMB_Preprocessed/unnamed (2).png",
    "d:/Skin/skin_AMB_Preprocessed/unnamed (3).png",
    "d:/Skin/skin_AMB_Preprocessed/unnamed (4).png",
    "d:/Skin/skin_AMB_Preprocessed/unnamed (5).png",
    "d:/Skin/skin_AMB_Preprocessed/unnamed (6).png",
    "d:/Skin/skin_AMB_Preprocessed/unnamed (7).png",
    "d:/Skin/skin_AMB_Preprocessed/unnamed (8).png",
    "d:/Skin/skin_AMB_Preprocessed/unnamed (9).png",
]

SAVE_DIR = "d:/Skin/gradcam_output"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# GRAD-CAM CLASS
# ============================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x):
        x = x.float()  # FIX

        out = self.model(x)
        class_idx = out.argmax(dim=1).item()

        self.model.zero_grad()
        out[0, class_idx].backward()

        grads = self.gradients[0]
        acts = self.activations[0]

        weights = grads.mean(dim=(1, 2))

        cam = torch.zeros(acts.shape[1:], dtype=acts.dtype).to(acts.device)

        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = F.relu(cam)
        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy()

# ============================================================
# LOAD MODEL
# ============================================================

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
target_layer = model.layer4[-1]
gradcam = GradCAM(model, target_layer)

# ============================================================
# PREPROCESS
# ============================================================

def preprocess(path):
    img = cv2.imread(path)
    if img is None:
        return None, None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    x = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    x = (x - mean) / std
    x = torch.tensor(x.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)

    return img, x

# ============================================================
# OVERLAY FUNCTION
# ============================================================

def overlay(img, cam):
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    result = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return result

# ============================================================
# RUN
# ============================================================

results = []

for i, path in enumerate(image_paths):

    img, x = preprocess(path)
    if img is None:
        print(f"Error loading {path}")
        continue

    cam = gradcam.generate(x)
    out = overlay(img, cam)

    results.append(out)

    # SAVE EACH IMAGE
    save_path = f"{SAVE_DIR}/gradcam_{i+1}.png"
    cv2.imwrite(save_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

# ============================================================
# BIG GRID DISPLAY
# ============================================================

n = len(results)
cols = 3
rows = math.ceil(n / cols)

plt.figure(figsize=(cols * 6, rows * 6))

for i in range(n):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(results[i])
    plt.title(f"Image {i+1}")
    plt.axis("off")

plt.suptitle("Grad-CAM Visualization", fontsize=20, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.96])

# SAVE BIG IMAGE
plt.savefig(f"{SAVE_DIR}/gradcam_BIG.png", dpi=400, bbox_inches="tight")

plt.close()
print(f"Grad-CAM complete. All outputs saved to: {SAVE_DIR}")