"""Shared temporal interpolation, coherent clip augmentation, and LR scheduling."""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F


def resample_temporal(frames, target: int) -> np.ndarray:
    """Linearly interpolate the full sequence, preserving both endpoints."""
    array = np.asarray(frames)
    if array.ndim < 2 or len(array) == 0 or target < 1:
        raise ValueError("A nonempty frame sequence and positive target length are required.")
    array = array.astype(np.float32)
    positions = np.linspace(0, len(array) - 1, target)
    left = np.floor(positions).astype(int)
    right = np.minimum(left + 1, len(array) - 1)
    weight = (positions - left).reshape((target,) + (1,) * (array.ndim - 1))
    return np.ascontiguousarray(array[left] * (1 - weight) + array[right] * weight, dtype=np.float32)


def augment_video(clip: torch.Tensor, *, spatial_jitter=3, mask_probability=.4,
                  photometric_probability=.8) -> torch.Tensor:
    """Augment [T,C,H,W] with ONE spatial/photometric transform for the clip.

    Replicate padding avoids wrapping pixels across the image boundary. Input
    tensors are never mutated; validation/test datasets must not call this.
    """
    if clip.ndim != 4 or clip.shape[0] == 0 or not clip.is_floating_point():
        raise ValueError("Expected a nonempty floating [T,C,H,W] clip in [0,1].")
    result = clip.clone()
    if spatial_jitter > 0:
        height, width = result.shape[-2:]
        dx, dy = torch.randint(-spatial_jitter, spatial_jitter + 1, (2,)).tolist()
        result = F.pad(result, (spatial_jitter,) * 4, mode="replicate")
        y, x = spatial_jitter + dy, spatial_jitter + dx
        result = result[:, :, y:y + height, x:x + width]
    if torch.rand(()).item() < photometric_probability:
        contrast = .8 + .4 * torch.rand(()).item()
        brightness = -.1 + .2 * torch.rand(()).item()
        result = (result * contrast + brightness).clamp(0, 1)
    if result.shape[0] > 2 and torch.rand(()).item() < mask_probability:
        length = min(int(torch.randint(2, 4, ()).item()), result.shape[0] - 1)
        start = int(torch.randint(0, result.shape[0] - length + 1, ()).item())
        result[start:start + length] = 0
    return result.contiguous()


def warmup_cosine_scheduler(optimizer, epochs: int, warmup_epochs=5, min_lr=1e-6):
    if epochs < 1 or warmup_epochs < 0 or min_lr < 0:
        raise ValueError("Invalid learning-rate schedule.")
    warmup = min(warmup_epochs, max(0, epochs - 1))
    functions = []
    for group in optimizer.param_groups:
        base = group["lr"]
        if base <= 0 or min_lr > base:
            raise ValueError("min_lr must be between zero and the base learning rate.")
        floor = min_lr / base
        def multiplier(epoch, floor=floor):
            if epochs == 1:
                return 1.
            if warmup and epoch < warmup:
                return (epoch + 1) / warmup
            progress = min(1., max(0., (epoch - warmup + 1) / max(1, epochs - warmup)))
            return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * progress))
        functions.append(multiplier)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, functions)
