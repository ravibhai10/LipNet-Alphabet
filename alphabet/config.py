"""Phase 5 decoder configuration — parsed from the canonical YAML.

Everything about the decoder lives under the ``decoder:`` section of
``configs/alphabet_dataset.yaml`` (allowed through the frozen loader's
registry; ignored by all Phase 2-4 parsers). Precedence mirrors the project:
typed dataclass defaults < YAML < explicit dotted-key overrides.

The vocabulary itself is NOT configured here — it comes from the single
canonical class alphabet/vocab.py (LetterVocabulary), as required.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass
class AuxiliaryConfig:
    """Optional phonetic/viseme heads. DISABLED by default (§9/§12).

    Enabling them uses the DERIVED supervision from alphabet/phonetics.py
    (letter-name ARPABET v1.0; viseme map v1.0) and must be recorded as an
    explicitly labelled experiment — never mixed into the primary result.
    """

    phone_enabled: bool = False
    viseme_enabled: bool = False
    lambda_phone: float = 0.3
    lambda_viseme: float = 0.3
    require_phone_map_version: str = "letter_names_arpabet_us_v1.0"
    require_viseme_map_version: str = "bear_harvey_initial_v1.0"


@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 8
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1          # fraction of total steps
    min_lr_ratio: float = 0.05
    grad_clip: float = 1.0
    amp: bool = False                  # mixed precision: only where numerically safe
    num_workers: int = 0
    early_stopping_patience: int = 10
    checkpoint_dir: str = "runs/letter_decoder"
    balanced_sampling: str = "auto"    # auto | on | off  (§18: measured first)
    auto_imbalance_ratio: float = 1.5
    augment: bool = True               # train-time temporal augmentation
    letter_head_hidden: Optional[int] = None  # None -> linear 192->27


@dataclass
class UncertaintyConfig:
    """Calibration policy. Thresholds are fit on validation, never fixed."""

    temperature: Optional[float] = None   # None -> fit on validation
    n_bins: int = 15
    coverage_target: float = 0.90
    confidence_threshold: Optional[float] = None  # None -> select on validation
    margin_threshold: Optional[float] = None


@dataclass
class DecoderConfig:
    dataset_root: str = "alphabet_dataset"
    repr_cache_dir: str = "runs/repr_cache"
    ablation: str = "all"              # Phase 3 ablation switches (§22)
    auxiliary: AuxiliaryConfig = field(default_factory=AuxiliaryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)

    def to_dict(self) -> Dict:
        return asdict(self)


#: Phase 3 ablation switches, reused verbatim (§22).
ABLATION_MODES: Dict[str, Dict[str, bool]] = {
    "all": {"use_appearance": True, "use_geometry": True, "use_motion": True},
    "appearance_only": {"use_appearance": True, "use_geometry": False, "use_motion": False},
    "geometry_only": {"use_appearance": False, "use_geometry": True, "use_motion": False},
    "motion_only": {"use_appearance": False, "use_geometry": False, "use_motion": True},
    "appearance_geometry": {"use_appearance": True, "use_geometry": True, "use_motion": False},
    "appearance_motion": {"use_appearance": True, "use_geometry": False, "use_motion": True},
    "geometry_motion": {"use_appearance": False, "use_geometry": True, "use_motion": True},
}
def stream_ablation(name: str):
    """Build repr.pipeline.StreamAblation from a mode name (programmatic only)."""
    from repr.pipeline import StreamAblation

    if name not in ABLATION_MODES:
        raise ValueError(
            f"unknown ablation mode {name!r}; expected one of {sorted(ABLATION_MODES)}")
    return StreamAblation(**ABLATION_MODES[name])


def ablation_config_hash(base_hash: str, ablation: str) -> str:
    """Cache provenance hash for an ablation run (differs per mode)."""
    if ablation not in ABLATION_MODES:
        raise ValueError(f"unknown ablation mode {ablation!r}")
    if ablation == "all":
        return base_hash
    return f"{base_hash}-abl-{ablation}"


def load_decoder_config(
    yaml_path: Path | str = "configs/alphabet_dataset.yaml",
    overrides: Optional[Dict] = None,
) -> DecoderConfig:
    """Parse the ``decoder:`` section (all fields optional -> defaults)."""
    raw = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
    sec = raw.get("decoder") or {}
    if overrides:  # dotted-key overrides, highest precedence
        for key, value in overrides.items():
            node = sec
            parts = key.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = value

    aux_sec = sec.get("auxiliary", {})
    aux = AuxiliaryConfig(
        phone_enabled=bool(aux_sec.get("phone_enabled", False)),
        viseme_enabled=bool(aux_sec.get("viseme_enabled", False)),
        lambda_phone=float(aux_sec.get("lambda_phone", 0.3)),
        lambda_viseme=float(aux_sec.get("lambda_viseme", 0.3)),
    )
    tsec = sec.get("training", {})
    train = TrainingConfig(
        seed=int(tsec.get("seed", 42)),
        batch_size=int(tsec.get("batch_size", 8)),
        epochs=int(tsec.get("epochs", 60)),
        lr=float(tsec.get("lr", 3e-4)),
        weight_decay=float(tsec.get("weight_decay", 0.01)),
        warmup_ratio=float(tsec.get("warmup_ratio", 0.1)),
        min_lr_ratio=float(tsec.get("min_lr_ratio", 0.05)),
        grad_clip=float(tsec.get("grad_clip", 1.0)),
        amp=bool(tsec.get("amp", False)),
        num_workers=int(tsec.get("num_workers", 0)),
        early_stopping_patience=int(tsec.get("early_stopping_patience", 10)),
        checkpoint_dir=str(tsec.get("checkpoint_dir", "runs/letter_decoder")),
        balanced_sampling=str(tsec.get("balanced_sampling", "auto")),
        auto_imbalance_ratio=float(tsec.get("auto_imbalance_ratio", 1.5)),
        augment=bool(tsec.get("augment", True)),
        letter_head_hidden=tsec.get("letter_head_hidden"),
    )
    usec = sec.get("uncertainty", {})
    unc = UncertaintyConfig(
        temperature=usec.get("temperature"),
        n_bins=int(usec.get("n_bins", 15)),
        coverage_target=float(usec.get("coverage_target", 0.90)),
        confidence_threshold=usec.get("confidence_threshold"),
        margin_threshold=usec.get("margin_threshold"),
    )
    cfg = DecoderConfig(
        dataset_root=str(sec.get("dataset_root",
                                 raw.get("dataset_root", "alphabet_dataset"))),
        repr_cache_dir=str(sec.get("repr_cache_dir", "runs/repr_cache")),
        ablation=str(sec.get("ablation", "all")),
        auxiliary=aux, training=train, uncertainty=unc,
    )
    _validate(cfg)
    return cfg


def _validate(cfg: DecoderConfig) -> None:
    if cfg.ablation not in ABLATION_MODES:
        raise ValueError(f"decoder.ablation must be one of {sorted(ABLATION_MODES)}")
    if cfg.training.batch_size <= 0 or cfg.training.epochs <= 0:
        raise ValueError("decoder.training batch_size/epochs must be positive")
    if cfg.training.num_workers < 0:
        raise ValueError("decoder.training.num_workers must be non-negative")
    if cfg.training.lr <= 0:
        raise ValueError("decoder.training.lr must be positive")
    if not (0 < cfg.uncertainty.coverage_target <= 1):
        raise ValueError("decoder.uncertainty.coverage_target must be in (0, 1]")
    if cfg.auxiliary.lambda_phone < 0 or cfg.auxiliary.lambda_viseme < 0:
        raise ValueError("auxiliary loss weights must be >= 0")
