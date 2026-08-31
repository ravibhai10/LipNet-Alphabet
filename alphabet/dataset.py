"""Phase 5 dataset: manifest-driven, Phase 3 representation-cache backed.

Data flow (§16 speaker-independent split preserved):
    manifest.jsonl  ->  entries for the requested split
                    ->  load <repr_cache_dir>/<recording_id>/fused.npy
                        (the canonical Phase 3 output; provenance-checked)
                    ->  target ids from the RECORDING-LEVEL label
                    ->  collate: pad to Tmax + mask

Multi-letter support (§15): the manifest only stores a single ``letter``.
A recording whose metadata.json carries an optional ``target_text`` field
(e.g. "HELLO") is treated as multi_letter; the label otherwise defaults to
the manifest letter and target_type == single_letter. One dataset, one
decoder — no architecture duplication.

Class imbalance (§18): letter recording counts are measured with
``letter_counts``; balanced sampling is exposed as an inverse-frequency
weight list (configurable, never forced).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from alphabet.targets import TargetType, classify_target
from alphabet.vocab import LetterVocabulary

logger = logging.getLogger(__name__)


@dataclass
class SampleMeta:
    """Machine-readable per-sample provenance (used by failure analysis)."""

    recording_id: str
    speaker_id: str
    session_id: str
    label: str                     # target text (single letter by default)
    target_type: str               # TargetType value
    split: Optional[str]
    rel_path: str
    frame_count: int
    duration_ms: float
    quality_summary: Dict = field(default_factory=dict)


def load_manifest_entries(dataset_root: Path, split: Optional[str] = None,
                          require_split: bool = True) -> List:
    """Load ManifestEntry records, optionally filtered to one split."""
    from data.schema import ManifestEntry

    manifest_path = Path(dataset_root) / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest not found: {manifest_path} "
            "(run `python -m alphabet preprocess` first)")
    entries: List[ManifestEntry] = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = ManifestEntry.from_jsonl(line)
            if require_split and not e.split:
                raise ValueError(
                    f"manifest entry {e.recording_id} has no split; "
                    "speaker-independent splits are mandatory (§16)")
            if split is not None and e.split != split:
                continue
            entries.append(e)
    return entries


def resolve_label(dataset_root: Path, entry) -> Tuple[str, TargetType]:
    """Label text + target_type for a manifest entry.

    Default: the manifest letter (single_letter). If the recording's
    metadata.json carries ``target_text`` (multi-letter future), use it.
    """
    text = entry.letter
    target_type = TargetType.SINGLE_LETTER
    meta_path = Path(dataset_root) / entry.path / "metadata.json"
    if meta_path.exists():
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            if raw.get("target_text"):
                text = str(raw["target_text"]).strip().upper()
                target_type = classify_target(text)
        except (json.JSONDecodeError, OSError):
            logger.warning("unreadable metadata.json for %s; using letter label",
                           entry.recording_id)
    return text, target_type


class RepresentationCacheError(RuntimeError):
    """Cached Phase 3 representation missing or provenance-stale."""


def cache_key(entry) -> str:
    """Unique cache key for a manifest entry.

    ``recording_id`` is only unique within a session (see data/manifest.py),
    so the cache is keyed by the recording's dataset-relative path with
    separators flattened. Used identically by cache building and loading.
    Accepts a manifest entry (uses ``entry.path``) or a raw relative path str.
    """
    path = entry if isinstance(entry, str) else entry.path
    return str(path).replace("\\", "/").replace("/", "__")
def _check_cache_provenance(rec_dir: Path, expected_config_hash: str) -> Dict:
    meta_path = rec_dir / "metadata.json"
    if not meta_path.is_file():
        raise RepresentationCacheError(f"no cached representation: {rec_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("config_hash") != expected_config_hash:
        raise RepresentationCacheError(
            f"stale cache for {rec_dir.name}: config_hash "
            f"{meta.get('config_hash')!r} != expected {expected_config_hash!r} "
            "(re-run `python -m alphabet cache-reprs`)")
    return meta


class AlphabetSequenceDataset(Dataset):
    """Fused-sequence dataset for letter-CTC training/evaluation.

    Provenance: every recording's cached representation is checked against
    ``expected_config_hash`` (Phase 2 config hash, optionally suffixed by the
    ablation name) on every load — stale caches fail loudly.
    """

    def __init__(
        self,
        entries: List,
        dataset_root: Path,
        repr_cache_dir: Path,
        expected_config_hash: str,
        vocab: LetterVocabulary,
        augment=None,               # SequenceAugmenter or None (val/test: None)
        seed: int = 0,
    ):
        self.entries = entries
        self.dataset_root = Path(dataset_root)
        self.repr_cache_dir = Path(repr_cache_dir)
        self.expected_config_hash = str(expected_config_hash)
        self.vocab = vocab
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.samples: List[Tuple[SampleMeta, Path]] = []
        for e in entries:
            text, target_type = resolve_label(self.dataset_root, e)
            rec_dir = self.repr_cache_dir / cache_key(e)
            meta = SampleMeta(
                recording_id=e.recording_id, speaker_id=e.speaker_id,
                session_id=e.session_id, label=text,
                target_type=target_type.value, split=e.split,
                rel_path=e.path, frame_count=e.frame_count,
                duration_ms=e.duration_ms, quality_summary=dict(e.quality_summary),
            )
            self.samples.append((meta, rec_dir))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        meta, rec_dir = self.samples[idx]
        _check_cache_provenance(rec_dir, self.expected_config_hash)
        fused_path = rec_dir / "fused.npy"
        fused = np.load(fused_path).astype(np.float32)
        if fused.ndim != 2:
            raise RepresentationCacheError(
                f"{fused_path}: expected [T, D]; got {fused.shape}")
        if not np.isfinite(fused).all():
            raise RepresentationCacheError(f"{fused_path}: NaN/Inf in fused features")
        if fused.shape[0] < 1:
            raise RepresentationCacheError(f"{fused_path}: empty sequence")
        if self.augment is not None:
            fused = self.augment(fused, self.rng)
        target_ids = self.vocab.encode(meta.label)
        return {
            "fused": fused,
            "length": int(fused.shape[0]),
            "target_ids": target_ids,
            "meta": meta,
        }


def collate_batch(batch: List[Dict]) -> Dict:
    """Pad [T_i, D] fused sequences -> [B, Tmax, D] + mask; build targets."""
    if not batch:
        raise ValueError("collate_batch received an empty batch")
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    t_max = int(lengths.max())
    dim = batch[0]["fused"].shape[-1]
    for b in batch:
        if b["fused"].shape[-1] != dim:
            raise ValueError("inconsistent fused dims inside batch")
    fused = torch.zeros(len(batch), t_max, dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), t_max, dtype=torch.bool)
    for i, b in enumerate(batch):
        t = b["length"]
        fused[i, :t] = torch.from_numpy(b["fused"])
        mask[i, :t] = True
    s_max = max(len(b["target_ids"]) for b in batch)
    targets = torch.zeros(len(batch), s_max, dtype=torch.long)  # pad = blank id 0
    for i, b in enumerate(batch):
        ids = b["target_ids"]
        targets[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return {
        "fused": fused,
        "mask": mask,
        "input_lengths": lengths,
        "targets": targets,
        "target_lengths": torch.tensor(
            [len(b["target_ids"]) for b in batch], dtype=torch.long),
        "metas": [b["meta"] for b in batch],
    }


def balanced_sampling_weights(metas: List[SampleMeta]) -> List[float]:
    """Inverse-frequency weights per recording (§18: measured, optional).

    Applying class weights directly to CTC is incorrect (the loss is
    sequence-level); the correct lever is sampling. Use only when the
    measured imbalance ratio warrants it (decided in training config).
    """
    counts: Dict[str, int] = {}
    for m in metas:
        counts[m.label] = counts.get(m.label, 0) + 1
    return [1.0 / counts[m.label] for m in metas]

