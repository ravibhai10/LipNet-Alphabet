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

Preprocessing notes (documented; to be reflected in code for production):
  * AV-HuBERT features are extracted at label_rate = 25 Hz. A 2.5 s clip
    yields ~63 feature frames. AV-HuBERT consumes video via its fairseq
    task (decoded frames fed to the feature encoder), so a *video-only*
    variant is used for lip-reading.
  * Frame sampling: use the full clip, optionally sample ``N`` frames.
  * Face/mouth crop requirements: AV-HuBERT has no mouth-crop requirement
    (it learns from raw frames), but a face-centric crop (e.g. face-landmarker
    bounded ``460x460`` then resized to the official ``224x224`` input) is
    the recommended pre-processing for the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from pipeline.avhubert.loader import AVHubertLoader, ModelUnavailableError

logger = logging.getLogger(__name__)

# Official AV-HuBERT input image size used by the feature encoder (224x224).
AVHUBERT_INPUT = (224, 224)

# Dimension mapped from AV-HuBERT D to the pipeline's 192-dim Conformer input.
AVHUBERT_PROJECTION_TO_192 = np.linalg.pinv(
    np.random.default_rng(0).normal(size=(192, 768)).astype(np.float32)
)


class AVHubertFeatureAdapter:
    """Adapts AV-HuBERT features for the existing Phase 4 Conformer."""

    def __init__(self, loader: AVHubertLoader | None = None):
        self._loader = loader or AVHubertLoader()
        self._proj = None

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
        """Map ``[T, D]`` -> ``[T, to_dim]`` (production: learned linear layer)."""
        if feats.ndim != 2:
            raise ValueError(f"expected [T, D], got {feats.shape}")
        if self._proj is None or self._proj.shape[0] != to_dim:
            self._proj = np.random.default_rng(0).normal(
                size=(to_dim, feats.shape[1])
            ).astype(np.float32)
        return feats @ self._proj.T

    def encode_token(self, video_path: str | Path) -> Tuple[np.ndarray, float]:
        """end-to-end: load -> encode -> [T,192]."""
        start = __import__("time").perf_counter()
        feats = self.encode(video_path)
        out = self.project(feats, 192)
        return out, (__import__("time").perf_counter() - start) * 1000.0
