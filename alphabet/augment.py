"""Temporal-sequence augmentation (train-time only).

Operates on the Phase 3 fused sequence [T, 192] as a WHOLE — never on
independent frames. Preserves:
- frame order (crop trims ends; dropout removes frames, never reorders)
- temporal consistency (features move together with their timestamps)
Mild feature-space noise substitutes for appearance-level brightness /
contrast jitter, which is impossible post-fusion without re-running the
appearance encoder; noise scale is deliberately small (default std 0.01).
Disabled for validation/test. NOT verified to improve accuracy — it is a
configurable, measurable experiment knob (§19).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class SequenceAugmenter:
    """Mild, order-preserving temporal augmentation of fused sequences."""

    def __init__(
        self,
        noise_std: float = 0.01,
        crop_fraction: float = 0.1,
        dropout_prob: float = 0.02,
        max_dropout_frames: int = 3,
        min_frames: int = 8,
    ):
        if noise_std < 0:
            raise ValueError("noise_std must be >= 0")
        if not (0.0 <= crop_fraction <= 0.5):
            raise ValueError("crop_fraction must be in [0, 0.5]")
        if not (0.0 <= dropout_prob <= 1.0):
            raise ValueError("dropout_prob must be in [0, 1]")
        if min_frames < 2:
            raise ValueError("min_frames must be >= 2")
        self.noise_std = float(noise_std)
        self.crop_fraction = float(crop_fraction)
        self.dropout_prob = float(dropout_prob)
        self.max_dropout_frames = int(max_dropout_frames)
        self.min_frames = int(min_frames)

    def __call__(self, fused: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        t, d = fused.shape
        # 1. temporal crop: trim up to crop_fraction/2 from each end
        max_side = int(t * self.crop_fraction / 2.0)
        lo = int(rng.integers(0, max_side + 1)) if max_side > 0 else 0
        hi = int(rng.integers(0, max_side + 1)) if max_side > 0 else 0
        out = fused[lo: t - hi] if t - hi > lo else fused
        # 2. mild temporal dropout (order preserved; bounded count)
        if self.dropout_prob > 0 and out.shape[0] > self.min_frames:
            budget = min(self.max_dropout_frames, out.shape[0] - self.min_frames)
            if budget > 0:
                drop = rng.random(out.shape[0]) < self.dropout_prob
                idx = np.flatnonzero(drop)[:budget]
                if idx.size:
                    keep = np.ones(out.shape[0], dtype=bool)
                    keep[idx] = False
                    out = out[keep]
        # 3. mild feature noise
        if self.noise_std > 0:
            out = out + rng.normal(0.0, self.noise_std, size=out.shape).astype(out.dtype)
        return np.ascontiguousarray(out, dtype=np.float32)
