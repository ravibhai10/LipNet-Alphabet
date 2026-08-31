"""
Phase 3 — Gated multistream visual speech representation pipeline.

Consumes verified Phase 2 outputs for one recording:
    appearance crops   -> ResNetStem   -> [T, 96]
    mouth landmarks    -> Geometry     -> [T, 64]
    landmark/scalar d  -> Motion       -> [T, 64]

then:
    per-stream deterministic normalization
    -> projection into R^192 (inside fusion)
    -> gated fusion  alpha_t = softmax(FPN([A_t, G_t, M_t]))
    -> residual      f_t = base(A'_t) + sum_i alpha_i * proj(stream_i)

Output preserves temporal structure: [B, T, 192] (+ gates [B, T, S]).
This module deliberately contains NO Conformer/temporal encoder — that is a
later phase. The interface below ([T, D_fused]) is what the future
TemporalEncoder will consume.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from config.loader import Phase2Config
from repr.appearance import AppearanceConfig
from repr.geometry import GeometryConfig
from repr.motion import MotionConfig
from repr.fusion import FusionConfig

logger = logging.getLogger(__name__)

#: Version recorded in every representation-cache metadata.json.
REPRESENTATION_VERSION = "1.0"


class RepresentationError(RuntimeError):
    """Base class for representation-pipeline failures."""


class AlignmentError(RepresentationError):
    """Streams are not temporally aligned (contract violation)."""


class NumericalValidationError(RepresentationError):
    """NaN/Inf/degenerate values detected in streams or gates."""


class CacheProvenanceError(RepresentationError):
    """Cached representation does not match the current configuration."""


@dataclass
class StreamAblation:
    """Programmatic switches for research ablations.

    The primary pipeline uses all three streams (True, True, True).
    """

    use_appearance: bool = True
    use_geometry: bool = True
    use_motion: bool = True

    def as_list(self) -> List[bool]:
        return [self.use_appearance, self.use_geometry, self.use_motion]


@dataclass
class RepresentationPipelineConfig:
    """
    Single source of truth for Phase 3 runtime parameters.

    Built exclusively from the loaded Phase2Config (YAML-driven); the
    ``from_phase2_config`` factory is the canonical constructor.
    """

    appearance_dim: int = 96
    geometry_dim: int = 64
    motion_dim: int = 64
    fused_dim: int = 192
    # Per-stream normalization method: "standard" (z-score) or "none".
    normalization: str = "standard"
    # Epsilon guarding division by zero in normalization / renormalization.
    eps: float = 1e-6
    # Ablation switches (primary pipeline: all streams on).
    ablation: StreamAblation = field(default_factory=StreamAblation)
    representation_version: str = REPRESENTATION_VERSION
    # Config provenance: the config_hash of the YAML that produced the
    # streams (recorded in the representation cache; stale cache is refused).
    config_hash: str = ""

    @classmethod
    def from_phase2_config(cls, cfg: Phase2Config, **overrides) -> "RepresentationPipelineConfig":
        """Derive the pipeline config from the authoritative YAML-backed config."""
        values = dict(
            appearance_dim=cfg.appearance.output_dim,
            geometry_dim=cfg.geometry.mlp_output,
            motion_dim=cfg.motion.mlp_output,
            fused_dim=cfg.fusion.fused_dim,
            config_hash=getattr(cfg, "config_hash", ""),
        )
        values.update(overrides)
        return cls(**values)


def _as_float_tensor(x: Any, name: str) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(x)).float()
    elif isinstance(x, torch.Tensor):
        t = x.float()
    else:
        raise AlignmentError(f"{name} must be numpy array or torch tensor, got {type(x)}")
    return t


def validate_stream_inputs(
    appearance: torch.Tensor,
    geometry: torch.Tensor,
    motion: torch.Tensor,
    timestamps: Optional[Any] = None,
    require_positive_length: bool = True,
) -> Tuple[int, ...]:
    """Enforce the Phase 3 input contract; fail loudly on any violation.

    Returns the validated common shape ``(T,)`` for unbatched inputs or
    ``(B, T)`` for batched inputs.
    """
    names = ["appearance", "geometry", "motion"]
    tensors = [_as_float_tensor(t, n) for t, n in zip((appearance, geometry, motion), names)]
    was_unbatched = [t.ndim == 2 for t in tensors]
    tensors = [t.unsqueeze(0) if t.ndim == 2 else t for t in tensors]

    for t, n in zip(tensors, names):
        if t.ndim not in (2, 3):
            raise AlignmentError(
                f"{n} must have shape [T, D] or [B, T, D]; got {tuple(t.shape)}"
            )
        if t.shape[-1] == 0:
            raise AlignmentError(f"{n} has zero embedding dimension")
        if require_positive_length and t.shape[-2] == 0:
            raise AlignmentError(f"{n} has zero-length sequence")
        if not torch.isfinite(t).all():
            raise NumericalValidationError(f"{n} contains NaN or Inf values")

    ref = tensors[0].shape[:-1]
    for t, n in zip(tensors[1:], names[1:]):
        if t.shape[:-1] != ref:
            raise AlignmentError(
                f"Frame-count mismatch across streams: appearance {tuple(ref)} vs "
                f"{n} {tuple(t.shape[:-1])}"
            )

    if timestamps is not None:
        ts = np.asarray(timestamps, dtype=np.float64)
        expected = ref[-1]
        if ts.shape[-1] != expected:
            raise AlignmentError(
                f"Timestamp count {ts.shape[-1]} != frame count {expected}"
            )
        diffs = np.diff(ts, axis=-1)
        if np.any(diffs <= 0):
            raise AlignmentError("Timestamps must be strictly increasing")

    # Unbatched inputs report the per-sequence temporal shape [T];
    # batched inputs report [B, T].
    if all(was_unbatched):
        return tuple(tensors[0].shape[1:-1])
    return tuple(tensors[0].shape[:-1])

class StreamNormalizer:
    """Per-stream deterministic normalization ("standard" z-score or "none").

    Statistics are computed per sequence (per call to ``fit``) and stored so
    that ``transform`` is deterministic. Never normalizes streams jointly.
    """

    METHODS = ("standard", "none")

    def __init__(self, method: str = "standard", eps: float = 1e-6):
        if method not in self.METHODS:
            raise RepresentationError(
                f"Unknown normalization method '{method}'; expected one of {self.METHODS}"
            )
        self.method = method
        self.eps = eps
        # Fitted statistics: dict name -> (mean [D], std [D]) or None for "none"
        self.stats: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}

    def fit(self, name: str, values: np.ndarray) -> "StreamNormalizer":
        """Fit on [T, D] (or [B, T, D]) values; stores per-feature stats."""
        arr = np.asarray(values, dtype=np.float64)
        mean = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
        std = arr.reshape(-1, arr.shape[-1]).std(axis=0)
        self.stats[name] = (mean.astype(np.float32), std.astype(np.float32))
        return self

    def transform(self, name: str, values: np.ndarray) -> np.ndarray:
        if self.method == "none":
            return np.asarray(values, dtype=np.float32)
        if name not in self.stats:
            raise RepresentationError(f"StreamNormalizer not fitted for stream '{name}'")
        mean, std = self.stats[name]
        return ((np.asarray(values, dtype=np.float32) - mean) / (std + self.eps)).astype(np.float32)

    def fit_transform(self, name: str, values: np.ndarray) -> np.ndarray:
        return self.fit(name, values).transform(name, values)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "stats": {
                k: None if v is None else {"mean": v[0].tolist(), "std": v[1].tolist()}
                for k, v in self.stats.items()
            },
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state["method"] not in self.METHODS:
            raise RepresentationError(f"Unknown normalization method in state: {state['method']}")
        self.method = state["method"]
        self.stats = {
            k: None if v is None else (
                np.asarray(v["mean"], dtype=np.float32),
                np.asarray(v["std"], dtype=np.float32),
            )
            for k, v in state["stats"].items()
        }


def apply_ablation(
    appearance: torch.Tensor,
    geometry: torch.Tensor,
    motion: torch.Tensor,
    ablation: StreamAblation,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero out disabled streams while keeping tensor shapes fixed.

    Shapes stay constant so the fusion module and cache layout are identical
    across ablation settings; disabled streams contribute zero information.
    """
    a = appearance if ablation.use_appearance else torch.zeros_like(appearance)
    g = geometry if ablation.use_geometry else torch.zeros_like(geometry)
    m = motion if ablation.use_motion else torch.zeros_like(motion)
    return a, g, m


class GatedFusionWithAblation(nn.Module):
    """
    Phase 3 fusion wrapper around the verified Phase 2 GatedFusion module.

    Implements (per timestep t):
        alpha_t = softmax(FPN([A_t, G_t, M_t]))                  # gates sum to 1
        f_t     = base(A'_t) + sum_i alpha_i * proj_i(stream_i)  # residual

    Exposes per-timestep gate values for research analysis.
    """

    def __init__(self, config: RepresentationPipelineConfig):
        super().__init__()
        from repr.fusion import GatedFusionWithResidual

        if not any(config.ablation.as_list()):
            raise RepresentationError(
                "Invalid ablation configuration: all three streams are "
                "disabled; at least one stream must remain active."
            )
        fc = FusionConfig(
            appearance_dim=config.appearance_dim,
            geometry_dim=config.geometry_dim,
            motion_dim=config.motion_dim,
            fused_dim=config.fused_dim,
            use_context=False,
        )
        self.config = config
        # Residual variant: f_t = base(A'_t) + Σ_i alpha_i * proj_i(stream_i).
        # Input and fused representations share R^192 via base_proj.
        self.inner = GatedFusionWithResidual(fc)

    @property
    def fpn(self):
        return self.inner.fpn

    def forward(
        self,
        appearance: torch.Tensor,   # [B, T, Da]
        geometry: torch.Tensor,     # [B, T, Dg]
        motion: torch.Tensor,       # [B, T, Dm]
        attention_mask: Optional[torch.Tensor] = None,  # [B, T], True = valid
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fused, weights = self.inner(appearance, geometry, motion, context=None)
        weights = weights[:, :, :3]  # drop context column when present

        if attention_mask is not None:
            mask = attention_mask.to(weights.dtype).unsqueeze(-1)      # [B, T, 1]
            # Renormalize over valid positions only. Padded positions receive
            # uniform weights so gate sums remain 1 everywhere; their fused
            # outputs are excluded downstream via the returned mask contract.
            wsum = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            weights = weights * mask / wsum + (1.0 - mask) / weights.shape[-1]

        return fused, weights


# ---------------------------------------------------------------------------
# Result container + full pipeline
# ---------------------------------------------------------------------------

@dataclass
class RepresentationResult:
    """Output of the representation pipeline for one recording."""

    recording_id: str
    fused: np.ndarray            # [T, D_fused]
    appearance: np.ndarray       # [T, Da] (normalized)
    geometry: np.ndarray         # [T, Dg] (normalized)
    motion: np.ndarray           # [T, Dm] (normalized)
    gates: np.ndarray            # [T, 3]  (appearance, geometry, motion weights)
    frame_indices: np.ndarray    # [T] int
    timestamps_ms: np.ndarray    # [T] float
    valid_length: int
    representation_version: str
    config_hash: str

    def to_temporal_encoder_input(self) -> Tuple[np.ndarray, int]:
        """Clean [T, D_fused] interface for the future TemporalEncoder."""
        return self.fused[: self.valid_length], self.valid_length


class RepresentationPipeline:
    """Extracts, normalizes and fuses the three streams for one recording.

    Consumes verified Phase 2 outputs only:
      - appearance crops (list of [H, W, 3])
      - mouth landmarks  ([T, K, 2], normalized)
      - geometry scalars ([T, S]) as produced by GeometryScalars
      - frame indices / timestamps from frame_data.json
    """

    STREAM_NAMES = ("appearance", "geometry", "motion")

    def __init__(
        self,
        config: RepresentationPipelineConfig,
        appearance_extractor=None,
        geometry_extractor=None,
        motion_extractor=None,
        device: str = "cpu",
    ):
        self.config = config
        self.normalizer = StreamNormalizer(
            method=config.normalization, eps=config.eps)
        self.ablation = config.ablation
        self.device = device

        if appearance_extractor is not None:
            self.appearance_extractor = appearance_extractor
        else:
            from repr.appearance import AppearanceConfig, AppearanceExtractor
            ac = AppearanceConfig(output_dim=config.appearance_dim)
            self.appearance_extractor = AppearanceExtractor(ac, device=device)

        if geometry_extractor is not None:
            self.geometry_extractor = geometry_extractor
        else:
            from repr.geometry import GeometryConfig, GeometryExtractor
            gc = GeometryConfig(mlp_hidden=96, mlp_output=config.geometry_dim)
            self.geometry_extractor = GeometryExtractor(gc, device=device)

        if motion_extractor is not None:
            self.motion_extractor = motion_extractor
        else:
            from repr.motion import MotionConfig, MotionExtractor
            mc = MotionConfig(mlp_output=config.motion_dim)
            self.motion_extractor = MotionExtractor(mc, device=device)

        self.fusion = GatedFusionWithAblation(config).to(device)
        self.fusion.eval()

    # ------------------------------------------------------------------
    # Input contract validation across Phase 2 artifacts
    # ------------------------------------------------------------------

    def _check_alignment(
        self,
        recording_id: str,
        n_appearance: int,
        n_landmarks: int,
        n_scalars: int,
        frame_indices: np.ndarray,
        timestamps_ms: np.ndarray,
    ) -> None:
        counts = {
            "appearance crops": n_appearance,
            "mouth landmark rows": n_landmarks,
            "geometry scalar rows": n_scalars,
            "frame indices": len(frame_indices),
            "timestamps": len(timestamps_ms),
        }
        unique = set(counts.values())
        if len(unique) != 1:
            raise AlignmentError(
                f"Recording '{recording_id}' streams are not temporally aligned: {counts}"
            )
        diffs = np.diff(np.asarray(timestamps_ms, dtype=np.float64))
        if np.any(diffs <= 0):
            raise AlignmentError(
                f"Recording '{recording_id}' timestamps are not strictly increasing"
            )

    # ------------------------------------------------------------------
    # Full forward pass
    # ------------------------------------------------------------------

    def process_recording(
        self,
        recording_id: str,
        appearance_crops: List[np.ndarray],
        mouth_landmarks: np.ndarray,   # [T, K, 2]
        geometry_scalars: np.ndarray,  # [T, S]
        frame_indices: np.ndarray,
        timestamps_ms: np.ndarray,
    ):
        T = len(appearance_crops)
        self._check_alignment(
            recording_id, T, mouth_landmarks.shape[0],
            geometry_scalars.shape[0], frame_indices, timestamps_ms,
        )
        if T < 2:
            raise AlignmentError(
                f"Recording '{recording_id}' has {T} frames; at least 2 required "
                "for temporal motion features"
            )

        timings: Dict[str, float] = {}

        t0 = time.perf_counter()
        app = self.appearance_extractor.extract_sequence(appearance_crops)
        timings["appearance"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        geo = self.geometry_extractor.extract_sequence(list(mouth_landmarks))
        timings["geometry"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        mot = self.motion_extractor.extract_sequence(mouth_landmarks, geometry_scalars)
        timings["motion"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        fused, gates, norm_streams = self._fuse(recording_id, app, geo, mot)
        timings["projection+fusion"] = time.perf_counter() - t0

        result = RepresentationResult(
            recording_id=recording_id,
            fused=fused,
            appearance=norm_streams["appearance"],
            geometry=norm_streams["geometry"],
            motion=norm_streams["motion"],
            gates=gates,
            frame_indices=np.asarray(frame_indices, dtype=np.int64),
            timestamps_ms=np.asarray(timestamps_ms, dtype=np.float64),
            valid_length=T,
            representation_version=self.config.representation_version,
            config_hash=self.config.config_hash,
        )
        return result, timings

    def _fuse(
        self,
        recording_id: str,
        app: np.ndarray,
        geo: np.ndarray,
        mot: np.ndarray,
    ):
        validate_stream_inputs(app, geo, mot)

        # Deterministic per-stream normalization
        norm_app = self.normalizer.fit_transform("appearance", app)
        norm_geo = self.normalizer.fit_transform("geometry", geo)
        norm_mot = self.normalizer.fit_transform("motion", mot)

        a, g, m = apply_ablation(
            torch.from_numpy(norm_app),
            torch.from_numpy(norm_geo),
            torch.from_numpy(norm_mot),
            self.ablation,
        )
        a = a.unsqueeze(0)  # [1, T, D]
        g = g.unsqueeze(0)
        m = m.unsqueeze(0)

        with torch.no_grad():
            fused, gates = self.fusion(a, g, m, attention_mask=None)

        fused_np = fused.squeeze(0).numpy().astype(np.float32)
        gates_np = gates.squeeze(0).numpy().astype(np.float32)

        if not np.isfinite(fused_np).all() or not np.isfinite(gates_np).all():
            raise NumericalValidationError(
                f"Recording '{recording_id}': NaN/Inf produced by fusion"
            )
        sums = gates_np.sum(axis=-1)
        if not np.allclose(sums, 1.0, atol=1e-4):
            raise NumericalValidationError(
                f"Gate sums deviate from 1.0: min={sums.min()}, max={sums.max()}"
            )
        if (gates_np < 0).any():
            raise NumericalValidationError("Negative gate values produced")

        return fused_np, gates_np, {
            "appearance": norm_app, "geometry": norm_geo, "motion": norm_mot,
        }

    # ------------------------------------------------------------------
    # Cache with provenance
    # ------------------------------------------------------------------

    def save_to_cache(self, result: RepresentationResult, cache_dir: Path) -> Path:
        rec_dir = Path(cache_dir) / result.recording_id
        rec_dir.mkdir(parents=True, exist_ok=True)
        np.save(rec_dir / "appearance.npy", result.appearance)
        np.save(rec_dir / "geometry.npy", result.geometry)
        np.save(rec_dir / "motion.npy", result.motion)
        np.save(rec_dir / "fused.npy", result.fused)
        np.save(rec_dir / "gates.npy", result.gates)
        metadata = {
            "recording_id": result.recording_id,
            "config_hash": result.config_hash,
            "representation_version": result.representation_version,
            "frame_count": int(result.valid_length),
            "dims": {
                "appearance": int(result.appearance.shape[-1]),
                "geometry": int(result.geometry.shape[-1]),
                "motion": int(result.motion.shape[-1]),
                "fused": int(result.fused.shape[-1]),
            },
        }
        (rec_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        return rec_dir

    def load_from_cache(
        self, recording_id: str, cache_dir: Path, config_hash: str,
        representation_version: str = REPRESENTATION_VERSION,
    ) -> RepresentationResult:
        rec_dir = Path(cache_dir) / recording_id
        meta_path = rec_dir / "metadata.json"
        if not meta_path.is_file():
            raise CacheProvenanceError(f"No cached representation for '{recording_id}'")
        meta = json.loads(meta_path.read_text())

        if meta.get("config_hash") != config_hash:
            raise CacheProvenanceError(
                f"Cached representation for '{recording_id}' was built with "
                f"config_hash={meta.get('config_hash')!r}; current config hash is "
                f"{config_hash!r}. Refusing stale reuse."
            )
        if meta.get("representation_version") != representation_version:
            raise CacheProvenanceError(
                f"Cached representation version {meta.get('representation_version')!r} "
                f"!= current {representation_version!r}"
            )

        appearance = np.load(rec_dir / "appearance.npy")
        geometry = np.load(rec_dir / "geometry.npy")
        motion = np.load(rec_dir / "motion.npy")
        fused = np.load(rec_dir / "fused.npy")
        gates = np.load(rec_dir / "gates.npy")
        T = int(meta["frame_count"])

        return RepresentationResult(
            recording_id=recording_id,
            fused=fused, appearance=appearance, geometry=geometry,
            motion=motion, gates=gates,
            frame_indices=np.arange(T, dtype=np.int64),
            timestamps_ms=np.arange(T, dtype=np.float64),
            valid_length=T,
            representation_version=meta["representation_version"],
            config_hash=meta["config_hash"],
        )


# ---------------------------------------------------------------------------
# Batching helper: pad variable-length sequences into [B, Tmax, D]
# ---------------------------------------------------------------------------

def batch_sequences(
    sequences: List[np.ndarray],
    attention_mask: Optional[np.ndarray] = None,
):
    """Pad [T_i, D] sequences into [B, Tmax, D] plus a boolean mask [B, Tmax].

    Padding value is 0; mask True marks real frames. Fails loudly on empty
    input or inconsistent embedding dims.
    """
    if not sequences:
        raise AlignmentError("batch_sequences called with zero sequences")

    dims = {s.shape[-1] for s in sequences}
    if len(dims) != 1:
        raise AlignmentError(f"Inconsistent embedding dims across batch: {dims}")

    lengths = np.array([s.shape[0] for s in sequences], dtype=np.int64)
    if (lengths == 0).any():
        raise AlignmentError("Zero-length sequence inside batch")

    max_len = int(lengths.max())
    B, D = len(sequences), sequences[0].shape[-1]

    if attention_mask is not None:
        am = np.asarray(attention_mask)
        if am.shape != (B, max_len):
            raise AlignmentError(
                f"attention_mask shape {am.shape} != ({B}, {max_len})"
            )
    else:
        am = np.zeros((B, max_len), dtype=bool)
        for i, L in enumerate(lengths):
            am[i, :L] = True

    batch = np.zeros((B, max_len, D), dtype=np.float32)
    for i, s in enumerate(sequences):
        L = s.shape[0]
        if not am[i, :L].all():
            raise AlignmentError(
                f"Mask marks padding within declared length {L} of row {i}"
            )
        batch[i, :L] = s

    return batch.astype(np.float32), am.astype(bool)


if __name__ == "__main__":
    print("repr/pipeline.py — import-only module; see tests/test_pipeline.py")

