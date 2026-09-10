"""
Visualize model predictions - original image, predicted mask, and overlay.
Useful for demo screenshots and sanity-checking model output.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import rasterio
import cv2
from model import UNet, load_spill_model


from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def visualize_result(image_path, model_path="unet_spill_best.pth", image_size=256, threshold=0.70, save_path="prediction_result.png"):
    model = load_spill_model(model_path, device="cpu")

    with rasterio.open(image_path) as src:
        img = src.read(1).astype(np.float32)
        orig_shape = img.shape

    if img.min() < 0:
        vmin, vmax = -35.0, -5.0
        img_norm = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)
    else:
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img_resized = cv2.resize(img_norm, (image_size, image_size))
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        pred = model(img_tensor)
        pred_mask = (torch.sigmoid(pred) > threshold).float().squeeze().numpy()

    pred_mask_full = cv2.resize(pred_mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)

    cmap_red = ListedColormap(['#ff2222'])
    pred_masked = np.ma.masked_where(pred_mask_full < 0.5, pred_mask_full)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_norm, cmap='gray')
    axes[0].set_title("1. Original SAR Satellite Image", fontsize=13, fontweight='bold')
    axes[0].axis('off')

    black_bg = np.zeros_like(pred_mask_full)
    axes[1].imshow(black_bg, cmap='gray', vmin=0, vmax=1)
    axes[1].imshow(pred_masked, cmap=cmap_red, vmin=0, vmax=1)
    axes[1].set_title("2. Model Predicted Oil Spill", fontsize=13, fontweight='bold', color='red')
    axes[1].axis('off')

    axes[2].imshow(img_norm, cmap='gray')
    axes[2].imshow(pred_masked, cmap=cmap_red, alpha=0.55)
    axes[2].set_title("3. Overlay on SAR Scene", fontsize=13, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")

    area_pixels = np.count_nonzero(pred_mask_full)
    print(f"Detected oil pixels: {area_pixels} ({100 * area_pixels / pred_mask_full.size:.3f}% of image)")


def visualize_ground_truth_comparison(image_path, mask_path, model_path="unet_spill_best.pth",
                                      image_size=256, threshold=0.70, save_path="prediction_vs_groundtruth.png"):
    """
    Side-by-side: original, ground truth mask, predicted mask, and clean transparent overlay.
    """
    model = load_spill_model(model_path, device="cpu")

    with rasterio.open(image_path) as src:
        img = src.read(1).astype(np.float32)
        orig_shape = img.shape

    with rasterio.open(mask_path) as src:
        gt_mask = src.read(1).astype(np.float32)

    if img.min() < 0:
        vmin, vmax = -35.0, -5.0
        img_norm = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)
    else:
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img_resized = cv2.resize(img_norm, (image_size, image_size))
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        pred = model(img_tensor)
        pred_mask = (torch.sigmoid(pred) > threshold).float().squeeze().numpy()

    pred_mask_full = cv2.resize(pred_mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)

    cmap_green = ListedColormap(['#00ff00'])
    cmap_red = ListedColormap(['#ff2222'])

    gt_masked = np.ma.masked_where(gt_mask < 0.5, gt_mask)
    pred_masked = np.ma.masked_where(pred_mask_full < 0.5, pred_mask_full)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Panel 1: Original SAR
    axes[0].imshow(img_norm, cmap='gray')
    axes[0].set_title("1. Original SAR Image", fontsize=13, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Ground Truth on black
    black_bg1 = np.zeros_like(gt_mask)
    axes[1].imshow(black_bg1, cmap='gray', vmin=0, vmax=1)
    axes[1].imshow(gt_masked, cmap=cmap_green, vmin=0, vmax=1)
    axes[1].set_title("2. Ground Truth Mask (Verified Spill)", fontsize=13, fontweight='bold', color='green')
    axes[1].axis('off')

    # Panel 3: Model Prediction on black
    black_bg2 = np.zeros_like(pred_mask_full)
    axes[2].imshow(black_bg2, cmap='gray', vmin=0, vmax=1)
    axes[2].imshow(pred_masked, cmap=cmap_red, vmin=0, vmax=1)
    axes[2].set_title("3. Model Prediction (Detected)", fontsize=13, fontweight='bold', color='red')
    axes[2].axis('off')

    # Panel 4: Clean transparent overlay on grayscale SAR
    axes[3].imshow(img_norm, cmap='gray')
    axes[3].imshow(gt_masked, cmap=cmap_green, alpha=0.55)
    axes[3].imshow(pred_masked, cmap=cmap_red, alpha=0.55)

    legend_elements = [
        Patch(facecolor='#00ff00', edgecolor='black', label='Ground Truth (Actual Spill)'),
        Patch(facecolor='#ff2222', edgecolor='black', label='Model Prediction (Detected)'),
        Patch(facecolor='#ffff00', edgecolor='black', label='Overlap (True Positive)')
    ]
    axes[3].legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=10)
    axes[3].set_title("4. Transparent Overlay (No Haze)", fontsize=13, fontweight='bold')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved clean comparison to {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize oil spill predictions against SAR imagery and ground truth.")
    default_ckpt = "unet_spill_best.pth" if os.path.exists("unet_spill_best.pth") else "unet_spill_checkpoint.pth"
    default_img = "dataset/real_dataset/images/00000.tif" if os.path.exists("dataset/real_dataset/images/00000.tif") else "dataset/backup_synthetic_200/images/00000.tif"
    parser.add_argument("--image", type=str, default=default_img, help="Path to Sentinel-1 SAR image (.tif)")
    parser.add_argument("--mask", type=str, default=None, help="Optional path to ground truth mask (.tif)")
    parser.add_argument("--checkpoint", "--model", dest="checkpoint", type=str, default=default_ckpt, help="Path to trained .pth model checkpoint")
    parser.add_argument("--image-size", type=int, default=128, help="Resize image dimension")
    parser.add_argument("--save", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    if args.mask:
        out_file = args.save if args.save else "prediction_vs_groundtruth.png"
        visualize_ground_truth_comparison(
            image_path=args.image,
            mask_path=args.mask,
            model_path=args.checkpoint,
            image_size=args.image_size,
            save_path=out_file
        )
    else:
        out_file = args.save if args.save else "prediction_result.png"
        visualize_result(
            image_path=args.image,
            model_path=args.checkpoint,
            image_size=args.image_size,
            save_path=out_file
        )