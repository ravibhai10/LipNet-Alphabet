"""AV-HuBERT feature adapter for the existing Phase 3/4/5 pipeline.

The existing pipeline expects a Phase-3 fused sequence of shape ``[T, 192]``
(GatedFusion output dim from `configs/alphabet_dataset.yaml`). AV-HuBERT
outputs ``[T, D]`` with ``D = 768`` (base) or 1024 (large) at 25 Hz. This
module defines the contract:

    AVHubertFeatureAdapter.encode(video_path) -> np.ndarray  [T, D]

and a small linear projection maps D -> 192 before the Conformer, keeping
the AV-HuBERT and Phase-2/3 streams **separated** (never mixed in the
same feature space; a configurable projection layer adapts AV-HuBERT for
the downstream CTC decoder).

Preprocessing contract (implemented in loader.py):
  * AV-HuBERT features are extracted at label_rate = 25 Hz. A 2.5 s clip
    yields ~63 feature frames. AV-HuBERT consumes video via its fairseq
    task (decoded frames fed to the feature encoder), so a *video-only*
    variant is used for lip-reading.
  * Frame sampling: use the full clip, optionally sample ``N`` frames.
  * Raw videos require measured face landmarks and a rotation-aligned mouth ROI.
    Already aligned mouth videos must be identified explicitly.
  * Input is grayscale, center-cropped to task.cfg.image_crop_size (normally
    88), scaled to [0,1], and normalized using the checkpoint's image mean/std.
  * The official video-only source is [B,1,T,H,W] with audio=None. There is no
    random projection or trained classifier supplied by the encoder itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from pipeline.avhubert.loader import AVHubertLoader, ModelUnavailableError

logger = logging.getLogger(__name__)

# Default official ROI size; runtime values come from the checkpoint task.
AVHUBERT_INPUT = (88, 88)

@dataclass
class ProjectConfig:
    """Configuration for the AV-HuBERT → pipeline projection layer."""

    d_out: int = 768      # AV-HuBERT output dimension (768 base / 1024 large)
    target_dim: int = 192  # Pipeline Conformer input dimension


class AVHubertFeatureAdapter:
    """Adapts AV-HuBERT features for the existing Phase 4 Conformer."""

    def __init__(
        self,
        loader: AVHubertLoader | None = None,
        projection: np.ndarray | None = None,
    ):
        self._loader = loader or AVHubertLoader()
        self._proj = projection

    @property
    def loader(self) -> AVHubertLoader:
        return self._loader

    @property
    def available(self) -> bool:
        return self._loader.available

    def encode(self, video_path: str | Path) -> np.ndarray:
        """Return an AV-HuBERT feature sequence ``[T, d_out]`` for a video."""
        if not self.available:
            raise ModelUnavailableError(
                "AV-HuBERT model is not available. "
                "Load a checkpoint via pipeline.avhubert.loader.AVHubertLoader, "
                "or provide a video file + pending real weights."
            )
        return self._loader.encode(video_path)

    def project(self, feats: np.ndarray, to_dim: int = 192) -> np.ndarray:
        """Map ``[T, D]`` using a caller-supplied trained projection."""
        if feats.ndim != 2:
            raise ValueError(f"expected [T, D], got {feats.shape}")
        if self._proj is None:
            raise ModelUnavailableError(
                "No trained AV-HuBERT projection is configured; refusing "
                "to use an untrained fallback projection."
            )
        if self._proj.shape != (to_dim, feats.shape[1]):
            raise ValueError(
                f"projection shape {self._proj.shape} does not match "
                f"expected {(to_dim, feats.shape[1])}"
            )
        return feats @ self._proj.T

    def encode_token(self, video_path: str | Path) -> Tuple[np.ndarray, float]:
        """end-to-end: load -> encode -> [T,192]."""
        start = __import__("time").perf_counter()
        feats = self.encode(video_path)
        out = self.project(feats, 192)
        return out, (__import__("time").perf_counter() - start) * 1000.0
