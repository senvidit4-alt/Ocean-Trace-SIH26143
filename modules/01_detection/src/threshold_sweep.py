"""
threshold_sweep.py - Tests inference-time thresholds 0.6, 0.7, 0.8 on 4 test images.
Generates 4-panel visualizations and prints IoU vs ground truth summary.
Usage: python threshold_sweep.py
"""

import os, cv2, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import rasterio, torch
from model import load_spill_model

CHECKPOINT = "unet_spill_best.pth"
IMAGE_SIZE  = 256
THRESHOLDS  = [0.6, 0.7, 0.8]
MAX_DIM     = 2048
OUT_DIR     = "threshold_sweep"

TESTS = [
    {"name": "Gulf_of_Mexico_image129",
     "image_path": "dataset/backup_synthetic_200/images/real_techie1_image129aug.tif",
     "mask_path": None, "input_scale": "linear",
     "label": "Gulf of Mexico (real_techie1_image129aug.tif)"},
    {"name": "Zenodo_00000",
     "image_path": "dataset/real_dataset/images/00000.tif",
     "mask_path": "dataset/real_dataset/masks/00000.tif",
     "input_scale": "db", "label": "Zenodo 00000.tif"},
    {"name": "Zenodo_00002",
     "image_path": "dataset/real_dataset/images/00002.tif",
     "mask_path": "dataset/real_dataset/masks/00002.tif",
     "input_scale": "db", "label": "Zenodo 00002.tif"},
    {"name": "Zenodo_00004",
     "image_path": "dataset/real_dataset/images/00004.tif",
     "mask_path": "dataset/real_dataset/masks/00004.tif",
     "input_scale": "db", "label": "Zenodo 00004.tif"},
]

os.makedirs(OUT_DIR, exist_ok=True)

def load_and_norm(image_path, input_scale):
    with rasterio.open(image_path) as src:
        img = src.read(1).astype(np.float32)
    h, w = img.shape
    scale = min(MAX_DIM/h, MAX_DIM/w, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    V_MIN, V_MAX = -35.0, -5.0
    if input_scale == "db":
        valid = ~np.isnan(img)
        norm = np.zeros_like(img)
        norm[valid] = np.clip((img[valid] - V_MIN) / (V_MAX - V_MIN), 0.0, 1.0)
    else:
        valid = (img > 0) & (~np.isnan(img))
        norm = np.zeros_like(img)
        db = 10.0 * np.log10(np.maximum(img[valid], 1e-6))
        norm[valid] = np.clip((db - V_MIN) / (V_MAX - V_MIN), 0.0, 1.0)
    return norm

def load_gt(mask_path, display_shape):
    with rasterio.open(mask_path) as src:
        gt = src.read(1).astype(np.float32)
    return cv2.resize(gt, (display_shape[1], display_shape[0]), interpolation=cv2.INTER_NEAREST)

def run_inference(img_norm, model, threshold):
    resized = cv2.resize(img_norm, (IMAGE_SIZE, IMAGE_SIZE))
    tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        pred = model(tensor)
        prob_map = torch.sigmoid(pred).squeeze().numpy()
    pred_128 = (prob_map > threshold).astype(np.uint8)
    pred_full = cv2.resize(pred_128, (img_norm.shape[1], img_norm.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    return prob_map, pred_full

def calc_iou(pred, gt):
    p = pred > 0.5; g = gt > 0.5
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter) / float(union + 1e-6)

def render_panel(img_norm, pred_full, gt_full, threshold, label, save_path):
    cmap_red   = ListedColormap(["#ff2222"])
    cmap_green = ListedColormap(["#00cc44"])
    pred_masked = np.ma.masked_where(pred_full < 0.5, pred_full)
    has_gt = gt_full is not None
    n = 4 if has_gt else 3
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6), facecolor="#0e0e0e")
    fig.suptitle(f"{label}  |  threshold = {threshold:.2f}",
                 fontsize=13, fontweight="bold", color="white", y=1.01)

    axes[0].imshow(img_norm, cmap="gray")
    axes[0].set_title("SAR Image", fontsize=11, color="white"); axes[0].axis("off")

    col = 1
    iou = None
    if has_gt:
        iou = calc_iou(pred_full, gt_full)
        gt_masked = np.ma.masked_where(gt_full < 0.5, gt_full)
        axes[col].imshow(np.zeros_like(gt_full), cmap="gray", vmin=0, vmax=1)
        axes[col].imshow(gt_masked, cmap=cmap_green, vmin=0, vmax=1)
        axes[col].set_title("Ground Truth (Verified Spill)", fontsize=11, color="#00cc44")
        axes[col].axis("off"); col += 1

    cov = 100 * pred_full.mean()
    iou_str = f"  IoU={iou:.4f}" if iou is not None else ""
    axes[col].imshow(np.zeros_like(pred_full), cmap="gray", vmin=0, vmax=1)
    axes[col].imshow(pred_masked, cmap=cmap_red, vmin=0, vmax=1)
    axes[col].set_title(f"Prediction\n({cov:.2f}% coverage{iou_str})", fontsize=11, color="#ff5555")
    axes[col].axis("off"); col += 1

    axes[col].imshow(img_norm, cmap="gray")
    axes[col].imshow(pred_masked, cmap=cmap_red, alpha=0.55)
    if has_gt:
        gt_masked2 = np.ma.masked_where(gt_full < 0.5, gt_full)
        axes[col].imshow(gt_masked2, cmap=cmap_green, alpha=0.35)
        axes[col].legend(handles=[
            Patch(facecolor="#00cc44", label="Ground Truth"),
            Patch(facecolor="#ff2222", label=f"Pred (thr={threshold:.2f})")
        ], loc="upper right", framealpha=0.85, fontsize=9)
    axes[col].set_title("Overlay", fontsize=11, color="white")
    axes[col].axis("off")

    for ax in axes: ax.set_facecolor("#0e0e0e")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="#0e0e0e")
    plt.close(fig)
    return iou, cov

print("Loading model...")
model = load_spill_model(CHECKPOINT, device="cpu")
results = {}

for test in TESTS:
    name, label = test["name"], test["label"]
    print(f"\n{'='*60}\nProcessing: {label}\n{'='*60}")
    img_norm = load_and_norm(test["image_path"], test["input_scale"])
    gt_full  = load_gt(test["mask_path"], img_norm.shape) if test["mask_path"] else None
    results[name] = {}
    for thr in THRESHOLDS:
        _, pred_full = run_inference(img_norm, model, thr)
        save_path = os.path.join(OUT_DIR, f"{name}_thr{int(thr*100)}.png")
        iou, cov = render_panel(img_norm, pred_full, gt_full, thr, label, save_path)
        results[name][thr] = {"iou": iou, "coverage_pct": cov, "save_path": save_path}
        iou_str = f"IoU={iou:.4f}" if iou is not None else "no GT"
        print(f"  threshold={thr:.2f}  coverage={cov:.3f}%  {iou_str}  -> {save_path}")

print("\n" + "="*72)
print("THRESHOLD SWEEP SUMMARY")
print("="*72)
for test in TESTS:
    name = test["name"]
    has_gt = test["mask_path"] is not None
    print(f"\n{name}")
    cov_row = "  Coverage %:  " + "  ".join(f"thr={t:.1f}:{results[name][t]['coverage_pct']:>6.3f}%" for t in THRESHOLDS)
    print(cov_row)
    if has_gt:
        iou_row = "  IoU vs GT:  " + "  ".join(f"thr={t:.1f}:{results[name][t]['iou']:.4f}    " for t in THRESHOLDS)
        print(iou_row)

print("\n" + "="*72)
print("BEST THRESHOLD PER ZENODO SCENE (highest IoU vs ground truth):")
print("="*72)
for test in TESTS:
    if not test["mask_path"]: continue
    name = test["name"]
    best_thr = max(THRESHOLDS, key=lambda t: results[name][t]["iou"] or 0)
    best_iou = results[name][best_thr]["iou"]
    print(f"  {name:<30}  best threshold = {best_thr:.2f}  IoU = {best_iou:.4f}")

print(f"\nAll images saved to: {os.path.abspath(OUT_DIR)}/")
