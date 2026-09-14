"""
Pretrained ResNet34 U-Net Architecture and Segmentation Loss Functions for Oil Spill Detection.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_model():
    """
    Builds a U-Net model with a pretrained ResNet34 encoder from segmentation_models_pytorch.
    Adapts ImageNet weights (3-channel) automatically to 1-channel grayscale SAR input.
    """
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    )


# Backwards compatibility alias for imports expecting UNet
UNet = build_model


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


def load_spill_model(model_path: str = "unet_spill_best.pth", device: str = "cpu") -> nn.Module:
    """
    Loads trained spill checkpoint into the ResNet34 U-Net architecture.
    Searches multiple standard locations to guarantee out-of-the-box loading.
    """
    from pathlib import Path
    dev = torch.device(device)
    model = build_model().to(dev)

    candidates = [
        Path(model_path),
        Path(__file__).resolve().parent / model_path,
        Path(__file__).resolve().parent / "unet_spill_best.pth",
        Path(__file__).resolve().parent.parent.parent.parent / model_path,
        Path(__file__).resolve().parent.parent.parent.parent / "unet_spill_best.pth",
    ]
    resolved_path = None
    for cand in candidates:
        if cand.exists() and cand.is_file():
            resolved_path = cand
            break

    if resolved_path is not None:
        state = torch.load(str(resolved_path), map_location=dev)
        model.load_state_dict(state)
    model.eval()
    return model


if __name__ == "__main__":
    # Quick sanity check — SAR images are 1-channel (grayscale)
    model = build_model()
    dummy_input = torch.randn(1, 1, 128, 128)
    output = model(dummy_input)
    print("Model built successfully!")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (1, 1, 128, 128), "Output shape mismatch!"
    print("Sanity check passed.")