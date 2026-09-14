import os
import shutil
import torch
from torch.utils.data import DataLoader, random_split
from data_loader import SpillDataset
from model import build_model, DiceBCELoss, HybridFocalLoss, calculate_metrics

import argparse

# ---- Config with CLI Arguments ----
parser = argparse.ArgumentParser(description="Train U-Net on Sentinel-1 SAR Oil Spill dataset.")
parser.add_argument("--image-dir", type=str, default="dataset/images", help="Path to SAR images folder")
parser.add_argument("--mask-dir", type=str, default="dataset/masks", help="Path to masks folder")
parser.add_argument("--image-size", type=int, default=128, help="Resize image height and width")
parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--max-samples", type=int, default=0, help="Max samples to load (0 for all samples)")
parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
parser.add_argument("--checkpoint", type=str, default="unet_spill_best.pth", help="Checkpoint save path")
parser.add_argument("--alpha", type=float, default=0.3, help="Focal Tversky alpha (False Positive weight)")
parser.add_argument("--beta", type=float, default=0.7, help="Focal Tversky beta (False Negative weight)")
parser.add_argument("--gamma", type=float, default=0.75, help="Focal Tversky gamma exponent")
parser.add_argument("--blend-weight", type=float, default=0.8, help="Weight for Focal Tversky vs DiceBCE (0.8 = 80%% FTL)")
args = parser.parse_args()

IMAGE_DIR = args.image_dir
MASK_DIR = args.mask_dir
IMAGE_SIZE = args.image_size
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR = args.lr
MAX_SAMPLES = args.max_samples if args.max_samples > 0 else None
PATIENCE = args.patience
CHECKPOINT_PATH = args.checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- Datasets (augmentation ONLY on training set, not validation) ----
full_dataset_no_aug = SpillDataset(IMAGE_DIR, MASK_DIR, image_size=IMAGE_SIZE, max_samples=MAX_SAMPLES, augment=False)
full_dataset_aug = SpillDataset(IMAGE_DIR, MASK_DIR, image_size=IMAGE_SIZE, max_samples=MAX_SAMPLES, augment=True)

train_size = int(0.8 * len(full_dataset_no_aug))
val_size = len(full_dataset_no_aug) - train_size

# Use the same split indices for both augmented/non-augmented versions
generator = torch.Generator().manual_seed(42)
train_indices, val_indices = random_split(range(len(full_dataset_no_aug)), [train_size, val_size], generator=generator)

train_dataset = torch.utils.data.Subset(full_dataset_aug, train_indices.indices)
val_dataset = torch.utils.data.Subset(full_dataset_no_aug, val_indices.indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# ---- Model, Loss, Optimizer ----
model = build_model().to(device)
criterion = HybridFocalLoss(
    alpha=args.alpha,
    beta=args.beta,
    gamma=args.gamma,
    blend_weight=args.blend_weight
)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---- Training Loop with Early Stopping ----
best_iou = 0.0
epochs_without_improvement = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_iou, val_dice = 0, 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            iou, dice = calculate_metrics(preds, masks)
            val_iou += iou
            val_dice += dice

    avg_train_loss = train_loss / len(train_loader)
    avg_val_iou = val_iou / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val IoU: {avg_val_iou:.4f} | Val Dice: {avg_val_dice:.4f}")

    # ---- Early stopping + best model checkpoint ----
    if avg_val_iou > best_iou or (epoch == 0 and avg_val_iou >= best_iou):
        best_iou = avg_val_iou
        epochs_without_improvement = 0
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"  -> New best model saved (IoU: {best_iou:.4f}) to {CHECKPOINT_PATH}")

        # Immediately backup best checkpoint to Google Drive (for Colab session persistence)
        drive_checkpoint = "/content/drive/MyDrive/oilspill_dataset/unet_spill_best_latest.pth"
        try:
            os.makedirs(os.path.dirname(drive_checkpoint), exist_ok=True)
            shutil.copy(CHECKPOINT_PATH, drive_checkpoint)
            print(f"  -> Best model backup copied to Drive: {drive_checkpoint}")
        except Exception as e:
            print(f"  -> Note: Drive backup skipped ({e})")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping triggered - no improvement for {PATIENCE} epochs.")
            break

print(f"\nTraining complete. Best Val IoU: {best_iou:.4f}")
print(f"Best model saved as {CHECKPOINT_PATH}")