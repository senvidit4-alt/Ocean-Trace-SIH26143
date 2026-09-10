"""
U-Net Architecture and Segmentation Loss Functions for Oil Spill Detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 Convolution to map features to classes"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net segmentation model designed for Oil Spill Detection in satellite and aerial imagery.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1, bilinear: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# ---------------------------------------------------------------------------
# Loss Functions & Evaluation Metrics
# ---------------------------------------------------------------------------

class DiceBCELoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss for robust segmentation
    especially when dealing with class imbalance (e.g. small oil spills).
    """

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(inputs, targets)

        probs = torch.sigmoid(inputs)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        dice_loss = 1.0 - dice
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for class-imbalanced segmentation.
    Penalizes False Negatives (missing oil) with beta, False Positives (lookalikes) with alpha.
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)

        tp = (probs * targets).sum()
        fp = (probs * (1.0 - targets)).sum()
        fn = ((1.0 - probs) * targets).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = torch.pow((1.0 - tversky), self.gamma)
        return focal_tversky


class HybridFocalLoss(nn.Module):
    """
    Blends Focal Tversky Loss (default 80%) with unweighted DiceBCELoss (default 20%)
    for optimal lookalike suppression and training stability.
    """
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        bce_weight: float = 0.5,
        blend_weight: float = 0.8,
        smooth: float = 1e-6
    ):
        super().__init__()
        self.ftl = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma, smooth=smooth)
        self.dice_bce = DiceBCELoss(bce_weight=bce_weight, smooth=smooth)
        self.blend_weight = blend_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ftl_loss = self.ftl(inputs, targets)
        dice_bce_loss = self.dice_bce(inputs, targets)
        return self.blend_weight * ftl_loss + (1.0 - self.blend_weight) * dice_bce_loss


def calculate_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6):
    """
    Compute IoU (Intersection over Union) and Dice score for binary segmentation.
    """
    probs = torch.sigmoid(preds)
    binary_preds = (probs > threshold).float()

    binary_preds = binary_preds.view(-1)
    targets = targets.view(-1)

    intersection = (binary_preds * targets).sum().item()
    total_union = (binary_preds.sum() + targets.sum()).item() - intersection

    iou = (intersection + smooth) / (total_union + smooth)
    dice = (2.0 * intersection + smooth) / (binary_preds.sum().item() + targets.sum().item() + smooth)

    return iou, dice


class UNetDirect(nn.Module):
    """
    Direct-mapped U-Net architecture matching Colab in-memory fast training weights.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 1):
        super().__init__()
        class DConv(nn.Module):
            def __init__(self, c_in, c_out):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True),
                )
            def forward(self, x): return self.conv(x)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.conv1(torch.cat([x4, x], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x3, x], dim=1))
        x = self.up3(x)
        x = self.conv3(torch.cat([x2, x], dim=1))
        x = self.up4(x)
        x = self.conv4(torch.cat([x1, x], dim=1))
        return self.outc(x)


def load_spill_model(model_path: str, device: str = "cpu") -> nn.Module:
    """
    Universally loads trained spill checkpoint (supports both UNet and UNetDirect architectures).
    """
    dev = torch.device(device)
    state = torch.load(model_path, map_location=dev)
    if "inc.conv.0.weight" in state:
        model = UNetDirect(in_channels=1, num_classes=1).to(dev)
    else:
        model = UNet(in_channels=1, num_classes=1).to(dev)
    model.load_state_dict(state)
    model.eval()
    return model


if __name__ == "__main__":
    # Quick sanity check — SAR images are 1-channel (grayscale), not 3-channel RGB
    model = UNet(in_channels=1, num_classes=1)
    dummy_input = torch.randn(2, 1, 256, 256)
    output = model(dummy_input)
    print("Model created successfully!")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 1, 256, 256), "Output shape mismatch!"