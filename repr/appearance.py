"""
Appearance stream representation extraction.

Per DESIGN.md:
- 64x64 RGB crop of mouth region (landmark-derived bbox + 2.2x margin)
- Translation/rotation/scale normalized to canonical frontal crop
- Compact ResNet-style stem: channels [32, 64, 128], LayerNorm, GELU, global average-pool
- Frame embeddings v_t ∈ ℝ^96
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AppearanceConfig:
    """Configuration for appearance stream extraction."""
    crop_size: Tuple[int, int] = (64, 64)
    channels: List[int] = None  # [32, 64, 128]
    output_dim: int = 96
    normalization_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalization_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    color_space: str = "RGB"

    def __post_init__(self):
        if self.channels is None:
            self.channels = [32, 64, 128]


class ResNetStem(nn.Module):
    """
    Compact ResNet-style stem for appearance stream.

    Input: (B, 3, 64, 64) - mouth crops
    Output: (B, 96) - frame embeddings
    """

    def __init__(self, config: AppearanceConfig):
        super().__init__()
        self.config = config

        # Initial convolution: (B,3,64,64) -> (B,32,32,32)
        self.conv1 = nn.Conv2d(
            3, config.channels[0], kernel_size=7, stride=2, padding=3, bias=False
        )
        self.ln1 = nn.LayerNorm([config.channels[0], 32, 32])
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # -> (B,32,16,16)

        # Residual blocks (each halves spatial size and changes channels)
        # layer1: (B,32,16,16) -> (B,64,8,8)
        self.layer1 = self._make_layer(config.channels[0], config.channels[1], half=True)
        # layer2: (B,64,8,8) -> (B,128,4,4)
        self.layer2 = self._make_layer(config.channels[1], config.channels[2], half=True)

        # 1x1 shortcut projections to match channel count and spatial size
        self.shortcut1 = nn.Conv2d(
            config.channels[0], config.channels[1], kernel_size=1, stride=2, bias=False
        )
        self.shortcut2 = nn.Conv2d(
            config.channels[1], config.channels[2], kernel_size=1, stride=2, bias=False
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final projection to output_dim
        self.fc = nn.Linear(config.channels[2], config.output_dim)

        # LayerNorm after projection
        self.ln_out = nn.LayerNorm(config.output_dim)

    def _make_layer(
        self, in_channels: int, out_channels: int, half: bool
    ) -> nn.Sequential:
        """Create a ResNet residual block (2 convs, LayerNorm + GELU between)."""
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=2 if half else 1, padding=1, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, stride=1, padding=1, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, 3, 64, 64) input crops

        Returns:
            (B, 96) frame embeddings
        """
        # Initial conv + pool
        x = self.conv1(x)  # (B, 32, 32, 32)
        x = self.ln1(x)
        x = F.gelu(x)
        x = self.pool(x)   # (B, 32, 16, 16)

        # Layer 1 (channels 32 -> 64, spatial 16 -> 8)
        identity = x
        x = self.layer1(x)          # (B, 64, 8, 8)
        x = x + self.shortcut1(identity)  # (B, 64, 8, 8)
        x = F.gelu(x)

        # Layer 2 (channels 64 -> 128, spatial 8 -> 4)
        identity = x
        x = self.layer2(x)          # (B, 128, 4, 4)
        x = x + self.shortcut2(identity)  # (B, 128, 4, 4)
        x = F.gelu(x)

        # Global pool and project
        x = self.global_pool(x)  # (B, 128, 1, 1)
        x = x.flatten(1)         # (B, 128)
        x = self.fc(x)           # (B, 96)
        x = self.ln_out(x)

        return x


class AppearanceExtractor:
    """
    Extract appearance stream features from mouth crops.

    Handles:
    - Crop normalization (mean/std)
    - ResNet stem forward pass
    - Batch processing
    """

    def __init__(self, config: AppearanceConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.model = ResNetStem(config).to(self.device)
        self.model.eval()

    def normalize_crop(self, crop: np.ndarray) -> torch.Tensor:
        """
        Normalize a single crop to [0, 1] then apply ImageNet normalization.

        Args:
            crop: [H, W, 3] RGB image, uint8 or float

        Returns:
            [3, H, W] normalized tensor
        """
        if crop.dtype == np.uint8:
            crop = crop.astype(np.float32) / 255.0

        # Ensure RGB
        if self.config.color_space == "BGR":
            crop = crop[:, :, ::-1]

        # Normalize
        mean = np.array(self.config.normalization_mean, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(self.config.normalization_std, dtype=np.float32).reshape(3, 1, 1)
        crop = (crop.transpose(2, 0, 1) - mean) / std

        return torch.from_numpy(crop).float()

    def extract_batch(self, crops: List[np.ndarray]) -> torch.Tensor:
        """
        Extract appearance embeddings for a batch of crops.

        Args:
            crops: List of [H, W, 3] RGB crops

        Returns:
            [B, 96] embeddings
        """
        # Normalize all crops
        batch = torch.stack([self.normalize_crop(c) for c in crops]).to(self.device)

        with torch.no_grad():
            embeddings = self.model(batch)

        return embeddings.cpu()

    def extract_sequence(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings for a full sequence.

        Args:
            crops: List of [H, W, 3] RGB crops (length T)

        Returns:
            [T, 96] numpy array
        """
        if not crops:
            return np.zeros((0, self.config.output_dim), dtype=np.float32)

        # Process in batches
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i+batch_size]
            embeddings = self.extract_batch(batch_crops)
            all_embeddings.append(embeddings.numpy())

        return np.vstack(all_embeddings)

    def save_embeddings(self, embeddings: np.ndarray, output_path: Path) -> None:
        """Save embeddings to .npz file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, embeddings=embeddings)

    def load_embeddings(self, input_path: Path) -> np.ndarray:
        """Load embeddings from .npz file."""
        data = np.load(input_path)
        return data["embeddings"]


def compute_appearance_from_crops(
    crops_dir: Path,
    output_path: Path,
    config: AppearanceConfig,
    device: str = "cpu"
) -> np.ndarray:
    """
    Convenience function to extract appearance embeddings from crop images.

    Args:
        crops_dir: Directory containing frame_XXXXXX.png crops
        output_path: Path to save embeddings.npz
        config: Appearance configuration
        device: Torch device

    Returns:
        [T, 96] embeddings array
    """
    # Load all crops in order
    crop_files = sorted(crops_dir.glob("frame_*.png"))
    if not crop_files:
        crop_files = sorted(crops_dir.glob("*.png"))

    crops = []
    for cf in crop_files:
        crop = cv2.imread(str(cf))
        if crop is not None:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(crop)

    if not crops:
        logger.warning(f"No crops found in {crops_dir}")
        return np.zeros((0, config.output_dim), dtype=np.float32)

    extractor = AppearanceExtractor(config, device)
    embeddings = extractor.extract_sequence(crops)
    extractor.save_embeddings(embeddings, output_path)

    return embeddings


import cv2  # Import at bottom to avoid circular import issues if needed