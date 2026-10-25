"""AV-HuBERT checkpoint loading (official fairseq entry point).

See README::

    >>> import fairseq
    >>> from fairseq import checkpoint_utils
    >>> models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    >>> model = models[0]

Fail-closed: raises :class:`ModelUnavailableError` when fairseq/checkpoint
are missing. Output contract: ``encode(video) -> np.ndarray (T, d_out)``
where ``d_out`` = 768 (base) or 1024 (large).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Fail-closed error: fairseq/checkpoint unavailable."""


class AVHubertLoader:
    """Loads an official AV-HuBERT checkpoint via ``fairseq``."""

    _DIPTHON_META = {
        "avhubert_base.pt": 768,
        "avhubert_large.pt": 1024,
    }

    def __init__(self, ckpt_path: str | Path | None = None) -> None:
        self.ckpt_path = Path(ckpt_path) if ckpt_path is not None else None
        self._models = None
        self._cfg = None
        self._task = None
        self.available = False
        self.d_out = 768

    @property
    def _models_root(self) -> Path:
        return Path(__file__).resolve().parents[1] / "models" / "external" / "avhubert"

    def _resolve_ckpt(self) -> Path:
        if self.ckpt_path is not None and self.ckpt_path.exists():
            return self.ckpt_path
        candidates = sorted(self._models_root.glob("*.pt"))
        if not candidates:
            raise ModelUnavailableError(
                f"No AV-HuBERT checkpoint found under {self._models_root}. "
                f"Download from https://facebookresearch.github.io/av_hubert "
                f"and place it in models/external/avhubert/."
            )
        return candidates[0]

    def load(self) -> "AVHubertLoader":
        import warnings
        from pathlib import Path as _P

        warnings.filterwarnings("ignore", category=UserWarning)

        try:
            from fairseq import checkpoint_utils
        except ImportError as exc:
            raise ModelUnavailableError(
                "AV-HuBERT requires the 'fairseq' package. Install with: "
                "pip install fairseq (see AVHUBERT_PLAN.md)."
            ) from exc

        self.ckpt_path = self._resolve_ckpt()
        try:
            models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
                [str(self.ckpt_path)], arg_overrides={}
            )
        except Exception as exc:
            raise ModelUnavailableError(
                f"Could not load AV-HuBERT checkpoint {self.ckpt_path}: {exc}"
            ) from exc

        self._models = models
        self._cfg = cfg
        self._task = task

        d = None
        for m in self._models:
            try:
                d = int(m.cfg.embed_dim)
            except Exception:
                pass
        if d is None:
            d = self._DIPTHON_META.get(self.ckpt_path.name, 768)
        self.d_out = d
        self.available = True
        logger.info("AV-HuBERT loaded: %s (d_out=%d)", self.ckpt_path, d)
        return self

    @property
    def models(self):
        if self._models is None:
            raise ModelUnavailableError("AV-HuBERT checkpoint not loaded")
        return self._models

    @property
    def task(self):
        if self._task is None:
            raise ModelUnavailableError("AV-HuBERT checkpoint not loaded")
        return self._task
