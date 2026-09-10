# ==============================================================================
# Colab Script: Real Zenodo Dataset Download + Level 1 Retraining on T4 GPU
# ==============================================================================

import os
import shutil
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from google.colab import files

# Verify GPU
assert torch.cuda.is_available(), "⚠️ Please switch runtime to T4 GPU first! (Runtime -> Change runtime type -> T4 GPU)"
print(f"🚀 Active GPU: {torch.cuda.get_device_name(0)}")

# ------------------------------------------------------------------------------
# 1. Clean Colab Disk & Install Fast Downloader
# ------------------------------------------------------------------------------
print("\n🧹 1. Cleaning Colab disk space...")
get_ipython().system("rm -rf /usr/local/cuda-11* /usr/local/cuda-12.0 /root/.cache")
get_ipython().system("apt-get update -qq && apt-get install -y -qq aria2 p7zip-full > /dev/null")
get_ipython().system("pip install -q rasterio opencv-python")

os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/masks", exist_ok=True)
get_ipython().system("df -h /")

# ------------------------------------------------------------------------------
# 2. Download & Extract Ground Truth Masks (~6.2 MB)
# ------------------------------------------------------------------------------
print("\n📥 2. Downloading Ground Truth Masks (6.2 MB)...")
get_ipython().system('aria2c -x 16 -s 16 -k 1M -c --dir=dataset --out=masks.7z "https://zenodo.org/api/records/8346860/files/01_Train_Val_Oil_Spill_mask.7z/content"')

get_ipython().system("7z x -y -odataset/_temp_masks dataset/masks.7z > /dev/null")
get_ipython().system("rm -f dataset/masks.7z")
get_ipython().system('find dataset/_temp_masks -type f -name "*.tif*" -exec mv {} dataset/masks/ \\;')
get_ipython().system("rm -rf dataset/_temp_masks")
print(f"✅ Masks ready: {len(os.listdir('dataset/masks'))} files")

# ------------------------------------------------------------------------------
# 3. Download & Extract Images (40.7 GB Archive -> Immediate Cleanup)
# ------------------------------------------------------------------------------
print("\n📥 3. Downloading Images Archive (40.7 GB via 16 threads, ~10-12 mins)...")
get_ipython().system('aria2c -x 16 -s 16 -k 1M -c --dir=dataset --out=images.7z "https://zenodo.org/api/records/8346860/files/01_Train_Val_Oil_Spill_images.7z/content"')

print("\n📦 Extracting real Zenodo scenes...")
get_ipython().system("7z x -y -odataset/_temp_img dataset/images.7z > /dev/null")

print("🗑️ Deleting 40.7 GB archive immediately to reclaim disk space...")
get_ipython().system("rm -f dataset/images.7z")

print("📂 Organizing images into dataset/images/...")
get_ipython().system('find dataset/_temp_img -type f -name "*.tif*" -exec mv {} dataset/images/ \\;')
get_ipython().system("rm -rf dataset/_temp_img")

# ------------------------------------------------------------------------------
# 4. Step 1 Audit: Integrity & Real Zenodo Confirmation
# ------------------------------------------------------------------------------
import rasterio
import cv2

img_files = sorted([f for f in os.listdir("dataset/images") if f.lower().endswith(('.tif', '.tiff'))])
mask_files = sorted([f for f in os.listdir("dataset/masks") if f.lower().endswith(('.tif', '.tiff'))])
matched_files = sorted(list(set(img_files).intersection(set(mask_files))))

print("\n" + "=" * 60)
print("🔍 DATASET INTEGRITY & AUTHENTICITY AUDIT")
print("=" * 60)
print(f"Total Images in dataset/images/: {len(img_files)}")
print(f"Total Masks in dataset/masks/:   {len(mask_files)}")
print(f"Matched Image-Mask Pairs:       {len(matched_files)}")

sample_path = os.path.join("dataset/images", matched_files[0])
with rasterio.open(sample_path) as src:
    sample_arr = src.read(1).astype(np.float32)
    sample_shape = src.shape
    sample_bands = src.count

print(f"\nSample Scene: {matched_files[0]}")
print(f"  Shape: {sample_shape} | Bands: {sample_bands}")
print(f"  Value Range: Min {sample_arr.min():.2f} dB, Max {sample_arr.max():.2f} dB")
is_real = sample_arr.min() < -35.0
print(f"  Real Zenodo Confirmation: {'✅ GENUINE REAL SAR DATA' if is_real else '⚠️ UNEXPECTED RANGE'}")
print("=" * 60 + "\n")

# ------------------------------------------------------------------------------
# 5. In-Memory Preprocessing with Fixed dB Clipping [-35 dB, -5 dB]
# ------------------------------------------------------------------------------
IMAGE_SIZE = 256
V_MIN, V_MAX = -35.0, -5.0

print(f"📥 Caching {len(matched_files)} scenes into RAM at {IMAGE_SIZE}x{IMAGE_SIZE}...")
cached_images = []
cached_masks = []

for idx, fname in enumerate(matched_files):
    with rasterio.open(os.path.join("dataset/images", fname)) as s_img:
        arr_img = s_img.read(1).astype(np.float32)
    with rasterio.open(os.path.join("dataset/masks", fname)) as s_mask:
        arr_mask = s_mask.read(1).astype(np.float32)

    # Level 1 Fixed SAR dB clipping
    if arr_img.min() < 0:
        arr_img = np.clip(arr_img, V_MIN, V_MAX)
        arr_img = (arr_img - V_MIN) / (V_MAX - V_MIN)
    elif arr_img.max() > 1.0:
        arr_img = (arr_img - arr_img.min()) / (arr_img.max() - arr_img.min() + 1e-6)

    arr_img = cv2.resize(arr_img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    arr_mask = cv2.resize(arr_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

    cached_images.append(arr_img)
    cached_masks.append(arr_mask)

cached_images = np.array(cached_images, dtype=np.float32)
cached_masks = np.array(cached_masks, dtype=np.float32)

# ------------------------------------------------------------------------------
# 6. 80/20 Train/Val Split
# ------------------------------------------------------------------------------
class FastSpillDataset(Dataset):
    def __init__(self, images, masks, augment=False):
        self.images, self.masks, self.augment = images, masks, augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, mask = self.images[idx], self.masks[idx]
        if self.augment:
            if random.random() > 0.5: img, mask = np.fliplr(img).copy(), np.fliplr(mask).copy()
            if random.random() > 0.5: img, mask = np.flipud(img).copy(), np.flipud(mask).copy()
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                img, mask = np.rot90(img, k).copy(), np.rot90(mask, k).copy()
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

random.seed(42)
indices = list(range(len(matched_files)))
random.shuffle(indices)
train_count = int(0.8 * len(matched_files))
train_idx, val_idx = indices[:train_count], indices[train_count:]

train_loader = DataLoader(FastSpillDataset(cached_images[train_idx], cached_masks[train_idx], augment=True), batch_size=16, shuffle=True)
val_loader = DataLoader(FastSpillDataset(cached_images[val_idx], cached_masks[val_idx], augment=False), batch_size=16, shuffle=False)
print(f"📊 Split: {len(train_idx)} Train scenes | {len(val_idx)} Validation scenes")

# ------------------------------------------------------------------------------
# 7. Model & Hybrid Focal Tversky Loss (Level 1 Fix)
# ------------------------------------------------------------------------------
class DConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class UNetDirect(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        self.inc = DConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DConv(512, 512))
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = DConv(512 + 256, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DConv(256 + 128, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = DConv(128 + 64, 64)
        self.up4 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv4 = DConv(64 + 64, 64)
        self.outc = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.conv1(torch.cat([x4, self.up1(x5)], dim=1))
        x = self.conv2(torch.cat([x3, self.up2(x)], dim=1))
        x = self.conv3(torch.cat([x2, self.up3(x)], dim=1))
        x = self.conv4(torch.cat([x1, self.up4(x)], dim=1))
        return self.outc(x)

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        tp = (probs * targets).sum()
        fp = (probs * (1.0 - targets)).sum()
        fn = ((1.0 - probs) * targets).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.pow(1.0 - tversky, self.gamma)

class HybridFocalLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, blend_weight=0.8):
        super().__init__()
        self.ftl = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.bce = nn.BCEWithLogitsLoss()
        self.blend_weight = blend_weight

    def forward(self, inputs, targets):
        ftl_loss = self.ftl(inputs, targets)
        bce_loss = self.bce(inputs, targets)
        probs = torch.sigmoid(inputs).view(-1)
        t_flat = targets.view(-1)
        intersection = (probs * t_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (probs.sum() + t_flat.sum() + 1e-6)
        dice_bce = 0.5 * bce_loss + 0.5 * dice_loss
        return self.blend_weight * ftl_loss + (1.0 - self.blend_weight) * dice_bce

def calculate_metrics(preds, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(preds)
    bin_preds = (probs > threshold).float().view(-1)
    t = targets.view(-1)
    intersection = (bin_preds * t).sum().item()
    total_union = (bin_preds.sum() + t.sum()).item() - intersection
    iou = (intersection + smooth) / (total_union + smooth)
    dice = (2.0 * intersection + smooth) / (bin_preds.sum().item() + t.sum().item() + smooth)
    return iou, dice

# ------------------------------------------------------------------------------
# 8. Train 30 Epochs on GPU (~90 seconds total)
# ------------------------------------------------------------------------------
model = UNetDirect(in_channels=1, num_classes=1).to("cuda")
criterion = HybridFocalLoss(alpha=0.3, beta=0.7, gamma=0.75, blend_weight=0.8)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 30
best_iou, best_epoch = 0.0, 0

print("\n" + "=" * 60)
print("🚀 TRAINING 30 EPOCHS ON T4 GPU (Hybrid Focal Tversky)")
print("=" * 60)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to("cuda"), masks.to("cuda")
        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_iou, val_dice = 0.0, 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to("cuda"), masks.to("cuda")
            preds = model(images)
            iou, dice = calculate_metrics(preds, masks)
            val_iou += iou
            val_dice += dice

    avg_train_loss = train_loss / len(train_loader)
    avg_val_iou = val_iou / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)

    improved = avg_val_iou > best_iou
    if improved:
        best_iou = avg_val_iou
        best_epoch = epoch + 1
        torch.save(model.state_dict(), "unet_spill_best.pth")

    marker = "⭐ [BEST]" if improved else ""
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_train_loss:.4f} | Val IoU: {avg_val_iou:.4f} | Val Dice: {avg_val_dice:.4f} {marker}")

print("=" * 60)
print(f"🎉 Training Complete! Best Val IoU: {best_iou:.4f} at Epoch {best_epoch}")
print("=" * 60)

# ------------------------------------------------------------------------------
# 9. Download Checkpoint
# ------------------------------------------------------------------------------
files.download("unet_spill_best.pth")
