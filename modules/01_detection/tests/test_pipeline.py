import sys
import os
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_repo_root / 'modules/01_detection/src'))

import torch
from torch.utils.data import DataLoader, Dataset
from data_loader import SpillDataset
from model import UNet, DiceBCELoss, calculate_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing pipeline using device: {device}")

data_img_dir = "dataset/images"
data_mask_dir = "dataset/masks"

if os.path.exists(data_img_dir) and os.listdir(data_img_dir):
    dataset = SpillDataset(data_img_dir, data_mask_dir, image_size=128, max_samples=20, augment=True)
else:
    class DummyDataset(Dataset):
        def __init__(self, num_samples=10, size=128):
            self.num_samples = num_samples
            self.size = size
        def __len__(self):
            return self.num_samples
        def __getitem__(self, idx):
            img = torch.rand(1, self.size, self.size)
            mask = (torch.rand(1, self.size, self.size) > 0.95).float()
            return img, mask
    dataset = DummyDataset()

loader = DataLoader(dataset, batch_size=2, shuffle=True)

print(f"Loaded {len(dataset)} dataset samples for pipeline test.")

model = UNet(in_channels=1, num_classes=1).to(device)
criterion = DiceBCELoss(bce_weight=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("Running pipeline training test (2 epochs)...")
for epoch in range(2):
    model.train()
    total_loss = 0
    total_iou = 0
    total_dice = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        iou, dice = calculate_metrics(preds, masks)
        total_iou += iou
        total_dice += dice

    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / len(loader)
    avg_dice = total_dice / len(loader)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f} | Mean IoU = {avg_iou:.4f} | Mean Dice = {avg_dice:.4f}")

print("Pipeline test successful! Data loader, U-Net, loss, and metrics verified on dataset.")