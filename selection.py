import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_DIR  = "d:/Skin/skin_output"
OUTPUT_DIR = "d:/Skin/selected_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)
image_files = sorted([f for f in os.listdir(INPUT_DIR) if "_lesion.png" in f])

print(f"Found {len(image_files)} lesion files in {INPUT_DIR}")

GRID_SIZE = 10
OUT_SIZE  = 300
N_SELECT  = 6

# ================================================================
# FEATURE EXTRACTION (full-res saliency, same as script 1)
# ================================================================
def crop_to_content(gray, thresh=10):
    mask = gray > thresh
    if not mask.any():
        return gray
    ys, xs = np.where(mask)
    return gray[ys.min():ys.max()+1, xs.min():xs.max()+1]

def make_saliency_image(gray, out_size=OUT_SIZE, blob_thresh=0.35):
    gray = crop_to_content(gray)
    gray = cv2.resize(gray, (out_size, out_size)).astype(np.float32)

    blur1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
    blur2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=6)
    dog   = np.abs(blur1 - blur2)
    lap   = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    sal   = dog + 0.5 * lap

    sal -= sal.min()
    if sal.max() > 0:
        sal /= sal.max()

    sal = np.where(sal > blob_thresh, sal, 0.0)
    sal = cv2.GaussianBlur(sal, (0, 0), sigmaX=1.2)
    return sal   # full-res, OUT_SIZE x OUT_SIZE, values 0..1

def saliency_to_cell_scores(saliency, grid_size=GRID_SIZE):
    """Only used internally for TDO scoring, NOT for display."""
    h, w = saliency.shape
    gh, gw = h // grid_size, w // grid_size
    scores = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            patch = saliency[i*gh:(i+1)*gh, j*gw:(j+1)*gw]
            scores[i, j] = patch.mean()
    return scores


# ================================================================
# TASMANIAN DEVIL OPTIMIZATION -> pick exactly N_SELECT cells
# ================================================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def fitness_function(prob_vec, feature_values, k):
    top_k_idx = np.argsort(prob_vec)[-k:]
    return -feature_values[top_k_idx].mean()

def tasmanian_devil_optimization(feature_values, k, n_agents=20, max_iter=60, lb=-4, ub=4, seed=None):
    rng = np.random.default_rng(seed)
    dim = len(feature_values)
    population = rng.uniform(lb, ub, (n_agents, dim))
    probs = lambda pos: sigmoid(pos)

    fitness = np.array([fitness_function(probs(population[i]), feature_values, k)
                         for i in range(n_agents)])
    best_idx = np.argmin(fitness)
    best_pos = population[best_idx].copy()
    best_fit = fitness[best_idx]

    for t in range(max_iter):
        for i in range(n_agents):
            j = rng.integers(n_agents)
            prey = population[j]
            I = rng.choice([1, 2])
            r = rng.random(dim)
            cand = (population[i] + r * (prey - I * population[i])
                    if fitness[j] < fitness[i]
                    else population[i] + r * (population[i] - prey))
            cand = np.clip(cand, lb, ub)
            cand_fit = fitness_function(probs(cand), feature_values, k)
            if cand_fit < fitness[i]:
                population[i], fitness[i] = cand, cand_fit

            r2 = rng.random(dim)
            radius = (1 - t / max_iter)
            cand2 = population[i] + radius * (2 * r2 - 1) * np.abs(population[i])
            cand2 = np.clip(cand2, lb, ub)
            cand2_fit = fitness_function(probs(cand2), feature_values, k)
            if cand2_fit < fitness[i]:
                population[i], fitness[i] = cand2, cand2_fit

        gbest = np.argmin(fitness)
        if fitness[gbest] < best_fit:
            best_fit = fitness[gbest]
            best_pos = population[gbest].copy()

    final_probs = probs(best_pos)
    top_k_idx = np.argsort(final_probs)[-k:]
    binary = np.zeros(dim, dtype=int)
    binary[top_k_idx] = 1
    return binary

def select_cells(cell_scores, k=N_SELECT, n_agents=20, max_iter=60, seed=None):
    grid_size = cell_scores.shape[0]
    flat = cell_scores.flatten()
    norm = (flat - flat.min()) / (np.ptp(flat) + 1e-6)
    best_binary = tasmanian_devil_optimization(norm, k, n_agents=n_agents, max_iter=max_iter, seed=seed)
    return best_binary.reshape(grid_size, grid_size)   # 10x10 binary mask, k ones


# ================================================================
# APPLY MASK BACK ONTO FULL-RES SALIENCY (keeps real blob texture)
# ================================================================
def mask_to_fullres(mask, out_size=OUT_SIZE):
    return cv2.resize(mask.astype(np.float32), (out_size, out_size), interpolation=cv2.INTER_NEAREST)

def draw_grid(ax, size, grid_size, color='white', lw=0.8):
    step = size / grid_size
    for k_ in range(grid_size + 1):
        ax.axhline(k_ * step, color=color, lw=lw)
        ax.axvline(k_ * step, color=color, lw=lw)

def plot_selection_pair(extracted_saliency, selected_saliency, filename,
                         grid_size=GRID_SIZE, out_size=OUT_SIZE):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, img, title in zip(
        axes,
        [extracted_saliency, selected_saliency],
        ["Extracted features", f"Selected features ({N_SELECT}/{grid_size*grid_size})"]
    ):
        ax.imshow(img, cmap='viridis', vmin=0, vmax=1, extent=[0, out_size, out_size, 0])
        draw_grid(ax, out_size, grid_size)
        ax.set_xlim(0, out_size); ax.set_ylim(out_size, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title)
        ax.set_aspect('equal')
    fig.suptitle(filename)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename.replace(".png", "_selection.png"))
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ================================================================
# RUN
# ================================================================
for file in image_files:
    path = os.path.join(INPUT_DIR, file)
    bgr = cv2.imread(path)
    if bgr is None:
        continue
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    saliency = make_saliency_image(gray)                      # full-res, real blobs
    cell_scores = saliency_to_cell_scores(saliency)            # 10x10, for TDO only

    mask = select_cells(cell_scores, k=N_SELECT)                # 10x10, 6 ones
    mask_fullres = mask_to_fullres(mask)                        # upsample mask to 300x300

    selected_saliency = saliency * mask_fullres                 # keep real blobs, only in selected cells

    plot_selection_pair(saliency, selected_saliency, file)
    print(f"{file}: selected exactly {int(mask.sum())}/{mask.size} cells")

    # Save the selected saliency map as a PNG for degree.py
    selected_uint8 = (selected_saliency * 255).astype(np.uint8)
    selected_bgr = cv2.cvtColor(selected_uint8, cv2.COLOR_GRAY2BGR)
    selected_path = os.path.join(OUTPUT_DIR, file.replace("_lesion.png", "_selected.png"))
    cv2.imwrite(selected_path, selected_bgr)
    print(f"Saved selected map: {selected_path}")

print(f"\nAll selection outputs saved to: {OUTPUT_DIR}")