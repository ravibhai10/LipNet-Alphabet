"""Phase 3 representation cache builder for Phase 5 training.

Runs the CANONICAL Phase 3 pipeline (repr/pipeline.RepresentationPipeline —
the exact frozen Phase 2->3 code path, including per-recording normalization
and gated fusion) over every manifest entry and stores the result with
``save_to_cache`` provenance (config_hash + representation_version).

Ablation support (§22): the Phase 3 stream switches are reused verbatim —
an ablation mode zeroes streams inside the SAME pipeline, and each ablation
gets its own cache subdirectory + provenance hash, so no experiment can read
another's representations by accident.

Requires preprocessed recordings with appearance crops on disk
(``preprocessing.save_crops: true`` — the appearance stream cannot be
recomputed from embeddings alone).
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from alphabet.config import ablation_config_hash, stream_ablation
from alphabet.dataset import cache_key, load_manifest_entries

logger = logging.getLogger(__name__)


def _load_frames_npz(path: Path) -> np.ndarray:
    data = np.load(path)
    keys = sorted(k for k in data.files if k.startswith("frame_"))
    if not keys:
        raise ValueError(f"no frame_* entries in {path}")
    return np.stack([data[k] for k in keys])


def _load_recording_inputs(rec_dir: Path):
    """Load exactly what RepresentationPipeline.process_recording consumes."""
    mouth_lm = _load_frames_npz(rec_dir / "landmarks" / "mouth_landmarks.npz")
    frame_data = json.loads((rec_dir / "frames" / "frame_data.json").read_text())
    fi = np.array([fd["frame_index"] for fd in frame_data], dtype=np.int64)
    ts = np.array([fd["timestamp_ms"] for fd in frame_data], dtype=np.float64)
    if len(fi) != mouth_lm.shape[0]:
        raise ValueError(
            f"{rec_dir}: frame_data count {len(fi)} != landmark frames "
            f"{mouth_lm.shape[0]}")
    scalars = np.asarray(
        np.load(rec_dir / "geometry_scalars.npz")["scalars"], dtype=np.float32)
    crop_files = sorted((rec_dir / "crops" / "appearance").glob("frame_*.png"))
    if not crop_files:
        raise FileNotFoundError(
            f"{rec_dir}: no appearance crops on disk. Re-run preprocessing "
            "with preprocessing.save_crops: true (required to rebuild the "
            "appearance stream).")
    if len(crop_files) != mouth_lm.shape[0]:
        raise ValueError(
            f"{rec_dir}: crops {len(crop_files)} != T={mouth_lm.shape[0]}")
    crops = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
             for f in crop_files]
    return crops, mouth_lm.astype(np.float64), scalars, fi, ts


def build_representation_cache(
    dataset_root: Path,
    yaml_config_path: Path,
    ablation: str = "all",
    cache_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    splits: Optional[List[str]] = None,
) -> Dict:
    """Build (or refresh) the Phase 3 fused-representation cache.

    Returns a summary dict {n_ok, n_failed, failures:[...], cache_dir, hash}.
    """
    from config.loader import load_phase2_config
    from repr.pipeline import RepresentationPipeline, RepresentationPipelineConfig

    phase2_cfg = load_phase2_config(yaml_config_path)
    root = Path(dataset_root)
    cache_root = (Path(cache_dir) if cache_dir
                  else Path("runs/repr_cache") / ablation)
    cache_root.mkdir(parents=True, exist_ok=True)

    base_cfg = RepresentationPipelineConfig.from_phase2_config(phase2_cfg)
    rep_cfg = replace(
        base_cfg,
        ablation=stream_ablation(ablation),
        config_hash=ablation_config_hash(phase2_cfg.config_hash, ablation),
    )
    pipeline = RepresentationPipeline(rep_cfg)

    entries = load_manifest_entries(root, split=None)
    if splits:
        entries = [e for e in entries if e.split in splits]
    if limit is not None:
        entries = entries[:limit]

    n_ok, failures = 0, []
    for e in entries:
        rec_dir = root / e.path
        try:
            crops, mouth_lm, scalars, fi, ts = _load_recording_inputs(rec_dir)
            result, _ = pipeline.process_recording(
                cache_key(e), crops, mouth_lm, scalars, fi, ts)
            pipeline.save_to_cache(result, cache_root)
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 — collect and continue
            failures.append({"recording": e.path, "error": f"{type(exc).__name__}: {exc}"})
            logger.warning("cache build failed for %s: %s", e.path, exc)
    summary = {
        "n_ok": n_ok,
        "n_failed": len(failures),
        "failures": failures,
        "cache_dir": str(cache_root),
        "config_hash": rep_cfg.config_hash,
        "ablation": ablation,
    }
    (cache_root / "_cache_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary
