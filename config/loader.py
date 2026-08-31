"""Canonical configuration loader for the Alphabet dataset (Phase 2).

Architecture
------------
``configs/alphabet_dataset.yaml`` is the single authoritative configuration
file. :func:`load_phase2_config` reads and validates it, applies CLI overrides
(dotted keys), and constructs the *typed* runtime configuration objects
(``IngestionConfig``, ``TrackingConfig``, ``AppearanceConfig``,
``GeometryConfig``, ``MotionConfig``, ``FusionConfig``, ``ManifestConfig``)
plus the aggregate ``PreprocessingConfig`` consumed by the pipeline.

Precedence (highest first)::

    CLI override  >  YAML  >  typed dataclass default

A ``config_hash`` is derived from the effective YAML so every recording can be
traced back to the exact configuration that produced it.

Explicit failure: a missing file, missing required section, unknown field,
wrong type, or out-of-range value raises :class:`ConfigValidationError` with a
clear message.  The loader never silently falls back to defaults for a value
that was explicitly supplied but invalid.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from data.ingestion import IngestionConfig
from data.manifest import ManifestConfig
from data.preprocess import PreprocessingConfig
from repr.appearance import AppearanceConfig
from repr.fusion import FusionConfig
from repr.geometry import GeometryConfig
from repr.motion import MotionConfig
from tracking.aligner import TrackingConfig

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/alphabet_dataset.yaml")


class ConfigError(Exception):
    """Base class for configuration errors."""


class ConfigValidationError(ConfigError):
    """Raised when the configuration file is missing, malformed, or invalid."""


def _fail(msg: str) -> None:
    raise ConfigValidationError(msg)


# ---------------------------------------------------------------------------
# Type coercion + validation helpers
# ---------------------------------------------------------------------------

def _expect_mapping(section: Any, path: str) -> Dict[str, Any]:
    if not isinstance(section, dict):
        _fail(f"Section '{path}' must be a mapping, got {type(section).__name__}")
    return section


def _as_bool(value: Any, path: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    _fail(f"'{path}' must be a boolean, got {value!r} ({type(value).__name__})")


def _as_int(value: Any, path: str) -> int:
    if isinstance(value, bool):
        _fail(f"'{path}' must be an integer, got boolean {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
    _fail(f"'{path}' must be an integer, got {value!r} ({type(value).__name__})")


def _as_float(value: Any, path: str) -> float:
    if isinstance(value, bool):
        _fail(f"'{path}' must be a number, got boolean {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    _fail(f"'{path}' must be a number, got {value!r} ({type(value).__name__})")


def _as_int_pair(value: Any, path: str) -> Tuple[int, int]:
    if (not isinstance(value, (list, tuple))) or len(value) != 2:
        _fail(f"'{path}' must be a 2-element list, got {value!r}")
    return (_as_int(value[0], f"{path}[0]"), _as_int(value[1], f"{path}[1]"))


def _as_float_pair(value: Any, path: str) -> Tuple[float, float]:
    if (not isinstance(value, (list, tuple))) or len(value) != 2:
        _fail(f"'{path}' must be a 2-element list, got {value!r}")
    return (_as_float(value[0], f"{path}[0]"), _as_float(value[1], f"{path}[1]"))


def _as_int_list(value: Any, path: str) -> List[int]:
    if not isinstance(value, list):
        _fail(f"'{path}' must be a list, got {type(value).__name__}")
    return [_as_int(v, f"{path}[{i}]") for i, v in enumerate(value)]


def _in_range(value: float, lo: float, hi: float, path: str) -> float:
    if not lo <= value <= hi:
        _fail(f"'{path}' must be in [{lo}, {hi}], got {value}")
    return value


def _positive(value: float, path: str) -> float:
    if value <= 0:
        _fail(f"'{path}' must be positive, got {value}")
    return value


def _require_section(raw: Dict[str, Any], section: str) -> Dict[str, Any]:
    if section not in raw:
        _fail(f"Configuration is missing required section: '{section}'")
    return _expect_mapping(raw[section], section)


def _deep_merge_overrides(base: Dict[str, Any],
                          overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge dotted-key CLI overrides into a copy of ``base``.

    ``{"mouth_alignment.crop_margin_multiplier": 3.0}`` sets that single leaf,
    leaving every other YAML value untouched (override > YAML > default).
    """
    merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        node: Dict[str, Any] = merged
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                _fail(f"Unknown configuration override path: '{dotted}'")
            nxt = dict(nxt)
            node[part] = nxt
            node = nxt
        leaf = parts[-1]
        if isinstance(value, dict):
            existing = node.get(leaf)
            if isinstance(existing, dict):
                existing.update(value)
            else:
                node[leaf] = value
        else:
            node[leaf] = value
    return merged


# ---------------------------------------------------------------------------
# Known-field registry (for "unknown critical field" detection)
# ---------------------------------------------------------------------------

KNOWN_FIELDS = {
    "dataset_root": "str",
    "config_version": "str",
    "video": {"fps", "max_sequence_length", "min_sequence_length",
              "codec", "input_resolution", "webcam_device"},
    "face_tracking": {"detection_confidence", "tracking_confidence",
                      "max_num_faces", "refine_landmarks",
                      "min_frames_with_face", "max_missing_face_gap"},
    "mouth_alignment": {"normalization_scale", "crop_margin_multiplier",
                        "appearance_crop_size", "context_crop_size",
                        "mouth_landmark_indices", "num_mouth_landmarks"},
    "appearance": {"color_space", "normalization"},
    "geometry": {"num_scalars", "mlp_output_dim", "savgol_window",
                 "savgol_polyorder"},
    "motion": {"optical_flow", "mlp_output_dim", "flow_patch_size"},
    "quality": {"blur_threshold", "brightness_range", "min_mouth_size",
                "max_pose_angle", "duplicate_ssim_threshold"},
    "preprocessing": {"version", "save_frames", "save_landmarks_npz",
                      "save_crops", "batch_size", "num_workers", "resume"},
    "splits": {"train_ratio", "val_ratio", "test_ratio", "random_seed",
               "min_speakers_per_split"},
    "manifest": {"path", "schema_version"},
    "temporal": {"encoder_type", "input_dim", "d_model", "num_layers",
                 "num_heads", "ffn_dim", "conv_kernel", "dropout",
                 "attention_dropout", "positional_encoding",
                 "max_sequence_length"},
    "recording": {"schema_version", "letters", "speaker_prefix",
                  "session_prefix", "recording_prefix"},
    # Phase 5: letter-decoder section. Allowed through here so the whole
    # project keeps ONE canonical YAML; it is parsed and consumed exclusively
    # by alphabet/config.py (load_decoder_config) and is IGNORED by every
    # Phase 2-4 parser below -- adding it changes no frozen behavior.
    "decoder": {"vocabulary", "ablation", "auxiliary", "training",
                "uncertainty", "repr_cache_dir"},
}

KNOWN_OPTICAL_FLOW_FIELDS = {"enabled", "pyr_scale", "levels", "winsize",
                             "iterations", "poly_n", "poly_sigma", "flags"}
KNOWN_APPEARANCE_NORM_FIELDS = {"mean", "std"}


def _check_unknown_fields(raw: Dict[str, Any]) -> None:
    """Fail on unknown top-level / section fields (typo protection)."""
    for key in raw:
        if key not in KNOWN_FIELDS:
            _fail(f"Unknown top-level configuration field: '{key}'")
    for section, fields in KNOWN_FIELDS.items():
        if section not in raw or not isinstance(fields, set):
            continue  # skip scalar top-level fields (e.g. dataset_root)
        if not isinstance(raw[section], dict):
            continue  # type errors are reported by the section parsers
        sec = raw[section]
        for key in sec:
            if key not in fields:
                _fail(f"Unknown field in '{section}': '{key}'")
        if section == "appearance" and "normalization" in sec:
            norm = _expect_mapping(sec["normalization"], "appearance.normalization")
            for key in norm:
                if key not in KNOWN_APPEARANCE_NORM_FIELDS:
                    _fail(f"Unknown field in 'appearance.normalization': '{key}'")
        elif section == "motion" and "optical_flow" in sec:
            of = _expect_mapping(sec["optical_flow"], "motion.optical_flow")
            for key in of:
                if key not in KNOWN_OPTICAL_FLOW_FIELDS:
                    _fail(f"Unknown field in 'motion.optical_flow': '{key}'")
        elif section == "motion":
            for key in sec:
                if key not in fields:
                    _fail(f"Unknown field in 'motion': '{key}'")
# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------

def _parse_video(raw: Dict[str, Any]) -> Dict[str, Any]:
    sec = _require_section(raw, "video")
    r: Dict[str, Any] = {}
    r["fps"] = _positive(_as_int(sec.get("fps", 30), "video.fps"), "video.fps")
    r["max_sequence_length"] = _positive(
        _as_int(sec.get("max_sequence_length", 76), "video.max_sequence_length"),
        "video.max_sequence_length")
    r["min_sequence_length"] = _positive(
        _as_int(sec.get("min_sequence_length", 15), "video.min_sequence_length"),
        "video.min_sequence_length")
    if r["min_sequence_length"] > r["max_sequence_length"]:
        _fail("video.min_sequence_length must be <= video.max_sequence_length")
    r["codec"] = str(sec.get("codec", "mp4v"))
    r["input_resolution"] = _as_int_pair(
        sec.get("input_resolution", [1280, 720]), "video.input_resolution")
    if r["input_resolution"][0] <= 0 or r["input_resolution"][1] <= 0:
        _fail(f"video.input_resolution must be positive, got {r['input_resolution']}")
    if "webcam_device" in sec:
        dev = _as_int(sec["webcam_device"], "video.webcam_device")
        if dev < 0:
            _fail(f"video.webcam_device must be >= 0, got {dev}")
    return r


def _parse_face_tracking(raw: Dict[str, Any]) -> Dict[str, Any]:
    sec = _require_section(raw, "face_tracking")
    r: Dict[str, Any] = {}
    r["detection_confidence"] = _in_range(
        _as_float(sec.get("detection_confidence", 0.5), "face_tracking.detection_confidence"),
        0.0, 1.0, "face_tracking.detection_confidence")
    r["tracking_confidence"] = _in_range(
        _as_float(sec.get("tracking_confidence", 0.5), "face_tracking.tracking_confidence"),
        0.0, 1.0, "face_tracking.tracking_confidence")
    r["max_num_faces"] = _positive(
        _as_int(sec.get("max_num_faces", 1), "face_tracking.max_num_faces"),
        "face_tracking.max_num_faces")
    r["refine_landmarks"] = _as_bool(
        sec.get("refine_landmarks", True), "face_tracking.refine_landmarks")
    r["min_frames_with_face"] = _positive(
        _as_int(sec.get("min_frames_with_face", 10), "face_tracking.min_frames_with_face"),
        "face_tracking.min_frames_with_face")
    r["max_missing_face_gap"] = _positive(
        _as_int(sec.get("max_missing_face_gap", 5), "face_tracking.max_missing_face_gap"),
        "face_tracking.max_missing_face_gap")
    return r
def _parse_mouth_alignment(raw: Dict[str, Any]) -> Dict[str, Any]:
    sec = _require_section(raw, "mouth_alignment")
    r: Dict[str, Any] = {}
    r["normalization_scale"] = str(sec.get("normalization_scale", "inner_mouth_width"))
    if r["normalization_scale"] not in ("inner_mouth_width", "inner_mouth_height", "face_width"):
        _fail(f"mouth_alignment.normalization_scale must be one of "
              f"['inner_mouth_width', 'inner_mouth_height', 'face_width'], "
              f"got {r['normalization_scale']!r}")
    r["crop_margin_multiplier"] = _positive(
        _as_float(sec.get("crop_margin_multiplier", 2.2), "mouth_alignment.crop_margin_multiplier"),
        "mouth_alignment.crop_margin_multiplier")
    r["appearance_crop_size"] = _as_int_pair(
        sec.get("appearance_crop_size", [64, 64]), "mouth_alignment.appearance_crop_size")
    r["context_crop_size"] = _as_int_pair(
        sec.get("context_crop_size", [112, 112]), "mouth_alignment.context_crop_size")
    for key in ("appearance_crop_size", "context_crop_size"):
        if r[key][0] <= 0 or r[key][1] <= 0:
            _fail(f"mouth_alignment.{key} must be positive, got {r[key]}")
    if "num_mouth_landmarks" in sec:
        r["num_mouth_landmarks"] = _positive(
            _as_int(sec["num_mouth_landmarks"], "mouth_alignment.num_mouth_landmarks"),
            "mouth_alignment.num_mouth_landmarks")
    if "mouth_landmark_indices" in sec:
        r["mouth_landmark_indices"] = _as_int_list(
            sec["mouth_landmark_indices"], "mouth_alignment.mouth_landmark_indices")
    return r


def _parse_appearance(raw: Dict[str, Any]) -> Dict[str, Any]:
    sec = _require_section(raw, "appearance")
    r: Dict[str, Any] = {}
    r["color_space"] = str(sec.get("color_space", "RGB")).upper()
    if r["color_space"] not in ("RGB", "BGR"):
        _fail(f"appearance.color_space must be 'RGB' or 'BGR', got {r['color_space']!r}")
    norm = sec.get("normalization", {})
    if norm:
        norm = _expect_mapping(norm, "appearance.normalization")
        mean = norm.get("mean", [0.485, 0.456, 0.406])
        std = norm.get("std", [0.229, 0.224, 0.225])
        if not (isinstance(mean, (list, tuple)) and isinstance(std, (list, tuple))):
            _fail("appearance.normalization.mean/std must be lists")
        mean_list = list(mean)
        std_list = list(std)
        if len(mean_list) != 3 or len(std_list) != 3:
            _fail("appearance.normalization.mean/std must have exactly 3 values")
        r["normalization_mean"] = tuple(
            _as_float(v, f"appearance.normalization.mean[{i}]") for i, v in enumerate(mean_list))
        r["normalization_std"] = tuple(
            _as_float(v, f"appearance.normalization.std[{i}]") for i, v in enumerate(std_list))
        for i, s in enumerate(r["normalization_std"]):
            if s <= 0:
                _fail(f"appearance.normalization.std[{i}] must be positive, got {s}")
    return r


def _parse_geometry(raw: Dict[str, Any]) -> Dict[str, Any]:
    sec = _require_section(raw, "geometry")
    r: Dict[str, Any] = {}
    r["num_scalars"] = _positive(
        _as_int(sec.get("num_scalars", 20), "geometry.num_scalars"), "geometry.num_scalars")
    mlp_out = _as_int(sec.get("mlp_output_dim", 64), "geometry.mlp_output_dim")
    r["mlp_hidden"] = _positive(mlp_out * 3 // 2, "geometry.mlp_hidden")
    r["mlp_output"] = _positive(mlp_out, "geometry.mlp_output_dim")
    r["savgol_window"] = _positive(
        _as_int(sec.get("savgol_window", 5), "geometry.savgol_window"),
        "geometry.savgol_window")
    r["savgol_polyorder"] = _positive(
        _as_int(sec.get("savgol_polyorder", 2), "geometry.savgol_polyorder"),
        "geometry.savgol_polyorder")
    return r
def _parse_motion(raw: Dict[str, Any]) -> Dict[str, Any]:
    sec = _require_section(raw, "motion")
    r: Dict[str, Any] = {}
    of = sec.get("optical_flow", {}) or {}
    r["optical_flow_enabled"] = _as_bool(of.get("enabled", True), "motion.optical_flow.enabled")
    r["flow_pyr_scale"] = _positive(
        _as_float(of.get("pyr_scale", 0.5), "motion.optical_flow.pyr_scale"),
        "motion.optical_flow.pyr_scale")
    r["flow_levels"] = _positive(
        _as_int(of.get("levels", 3), "motion.optical_flow.levels"),
        "motion.optical_flow.levels")
    r["flow_winsize"] = _positive(
        _as_int(of.get("winsize", 15), "motion.optical_flow.winsize"),
        "motion.optical_flow.winsize")
    r["flow_iterations"] = _positive(
        _as_int(of.get("iterations", 3), "motion.optical_flow.iterations"),
        "motion.optical_flow.iterations")
    r["flow_poly_n"] = _positive(
        _as_int(of.get("poly_n", 5), "motion.optical_flow.poly_n"),
        "motion.optical_flow.poly_n")
    r["flow_poly_sigma"] = _positive(
        _as_float(of.get("poly_sigma", 1.2), "motion.optical_flow.poly_sigma"),
        "motion.optical_flow.poly_sigma")
    r["flow_flags"] = _as_int(of.get("flags", 0), "motion.optical_flow.flags")
    mlp_out = _as_int(sec.get("mlp_output_dim", 64), "motion.mlp_output_dim")
    r["mlp_hidden"] = _positive(mlp_out * 3 // 2, "motion.mlp_hidden")
    r["mlp_output"] = _positive(mlp_out, "motion.mlp_output_dim")
    r["flow_patch_size"] = _as_int_pair(
        sec.get("flow_patch_size", [16, 16]), "motion.flow_patch_size")
    return r


def _parse_quality(raw: Dict[str, Any]) -> Dict[str, Any]:
    sec = _require_section(raw, "quality")
    r: Dict[str, Any] = {}
    r["blur_threshold"] = _as_float(sec.get("blur_threshold", 100.0), "quality.blur_threshold")
    if r["blur_threshold"] < 0:
        _fail(f"quality.blur_threshold must be >= 0, got {r['blur_threshold']}")
    r["brightness_range"] = _as_float_pair(
        sec.get("brightness_range", [20, 235]), "quality.brightness_range")
    if not (0 <= r["brightness_range"][0] < r["brightness_range"][1] <= 255):
        _fail(f"quality.brightness_range must satisfy 0 <= lo < hi <= 255, "
              f"got {r['brightness_range']}")
    r["min_mouth_size"] = _positive(
        _as_int(sec.get("min_mouth_size", 400), "quality.min_mouth_size"),
        "quality.min_mouth_size")
    r["max_pose_angle"] = _positive(
        _as_float(sec.get("max_pose_angle", 30.0), "quality.max_pose_angle"),
        "quality.max_pose_angle")
    r["duplicate_ssim_threshold"] = _in_range(
        _as_float(sec.get("duplicate_ssim_threshold", 0.98),
                  "quality.duplicate_ssim_threshold"),
        0.0, 1.0, "quality.duplicate_ssim_threshold")
    return r


def _parse_preprocessing(raw: Dict[str, Any]) -> Dict[str, Any]:
    sec = _require_section(raw, "preprocessing")
    r: Dict[str, Any] = {}
    r["version"] = str(sec.get("version", "1.0"))
    r["save_frames"] = _as_bool(sec.get("save_frames", False), "preprocessing.save_frames")
    r["save_landmarks_npz"] = _as_bool(
        sec.get("save_landmarks_npz", True), "preprocessing.save_landmarks_npz")
    r["save_crops"] = _as_bool(sec.get("save_crops", False), "preprocessing.save_crops")
    r["batch_size"] = _positive(
        _as_int(sec.get("batch_size", 32), "preprocessing.batch_size"),
        "preprocessing.batch_size")
    r["num_workers"] = _as_int(sec.get("num_workers", 4), "preprocessing.num_workers")
    if r["num_workers"] < 0:
        _fail(f"preprocessing.num_workers must be >= 0, got {r['num_workers']}")
    r["resume"] = _as_bool(sec.get("resume", True), "preprocessing.resume")
    return r


def _parse_splits(raw: Dict[str, Any]) -> Dict[str, Any]:
    sec = _require_section(raw, "splits")
    r: Dict[str, Any] = {}
    defaults = {"train_ratio": 0.70, "val_ratio": 0.15, "test_ratio": 0.15}
    for key in ("train_ratio", "val_ratio", "test_ratio"):
        r[key] = _in_range(
            _as_float(sec.get(key, defaults[key]), f"splits.{key}"),
            0.0, 1.0, f"splits.{key}")
    total = r["train_ratio"] + r["val_ratio"] + r["test_ratio"]
    if abs(total - 1.0) > 1e-6:
        _fail(f"splits ratios must sum to 1.0, got {total:.4f}")
    r["random_seed"] = _as_int(sec.get("random_seed", 42), "splits.random_seed")
    r["min_speakers_per_split"] = _positive(
        _as_int(sec.get("min_speakers_per_split", 3), "splits.min_speakers_per_split"),
        "splits.min_speakers_per_split")
    return r


def _parse_manifest(raw: Dict[str, Any]) -> Dict[str, Any]:
    sec = _require_section(raw, "manifest")
    r: Dict[str, Any] = {}
    r["path"] = str(sec.get("path", "manifest.jsonl"))
    r["schema_version"] = str(sec.get("schema_version", "1.0"))
    return r


def _parse_temporal(raw: Dict[str, Any], fused_dim: int) -> Dict[str, Any]:
    """Parse the ``temporal:`` section (Phase 4 Conformer parameters).

    All values are validated here; the section is optional in YAML (defaults
    below) but any present value must be well-formed. ``fused_dim`` is the
    Phase 3 fusion output dimension; the encoder input must match it.
    """
    sec = raw.get("temporal", {})
    if not isinstance(sec, dict):
        _fail(f"'temporal' section must be a mapping, got {type(sec).__name__}")

    def _int(field: str, default: int, minimum: int = 1) -> int:
        v = sec.get(field, default)
        if isinstance(v, bool) or not isinstance(v, int):
            _fail(f"temporal.{field} must be an integer, got {v!r}")
        if v < minimum:
            _fail(f"temporal.{field} must be >= {minimum}, got {v}")
        return v

    def _float01(field: str, default: float) -> float:
        v = sec.get(field, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            _fail(f"temporal.{field} must be a number, got {v!r}")
        v = float(v)
        if not (0.0 <= v <= 1.0):
            _fail(f"temporal.{field} must be within [0, 1], got {v}")
        return v

    r: Dict[str, Any] = {}
    et = str(sec.get("encoder_type", "conformer"))
    if et != "conformer":
        _fail(
            "temporal.encoder_type must be 'conformer'; other temporal "
            f"backends are not part of the design (got {et!r})"
        )
    r["encoder_type"] = et

    r["input_dim"] = _int("input_dim", 192)
    if r["input_dim"] != fused_dim:
        _fail(
            f"temporal.input_dim ({r['input_dim']}) must equal the Phase 3 "
            f"fusion output dimension ({fused_dim})"
        )
    r["d_model"] = _int("d_model", 192)
    r["num_layers"] = _int("num_layers", 4)
    r["num_heads"] = _int("num_heads", 4)
    r["ffn_dim"] = _int("ffn_dim", 768)
    kernel = _int("conv_kernel", 7)
    if kernel % 2 == 0:
        _fail(f"temporal.conv_kernel must be odd (depthwise 'same' padding), got {kernel}")
    r["conv_kernel"] = kernel
    r["dropout"] = _float01("dropout", 0.1)
    r["attention_dropout"] = _float01("attention_dropout", 0.1)

    pe = str(sec.get("positional_encoding", "sinusoidal_learned"))
    if pe not in ("sinusoidal", "learned", "sinusoidal_learned"):
        _fail(
            "temporal.positional_encoding must be one of "
            f"('sinusoidal', 'learned', 'sinusoidal_learned'); got {pe!r}"
        )
    r["positional_encoding"] = pe

    if "d_model" in r and "num_heads" in r and r["d_model"] % r["num_heads"] != 0:
        _fail(
            f"temporal.num_heads must divide temporal.d_model "
            f"({r['d_model']} % {r['num_heads']} != 0)"
        )
    r["max_sequence_length"] = _int("max_sequence_length", 512)
    return r
# ---------------------------------------------------------------------------
# Public typed configuration bundles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityConfig:
    """Quality thresholds and flags (from YAML ``quality:``)."""
    blur_threshold: float = 100.0
    brightness_range: Tuple[float, float] = (20.0, 235.0)
    min_mouth_size: int = 400
    max_pose_angle: float = 30.0
    duplicate_ssim_threshold: float = 0.98


@dataclass(frozen=True)
class SplitConfig:
    """Dataset split parameters (from YAML ``splits:``)."""
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    min_speakers_per_split: int = 3


# Field names intentionally match temporal.encoder.TemporalEncoderConfig so
# ``build_temporal_model`` can adopt this object directly without torch being
# imported by the config layer.
@dataclass(frozen=True)
class TemporalConfig:
    """Conformer temporal-encoder parameters (from YAML ``temporal:``)."""
    encoder_type: str = "conformer"
    input_dim: int = 192
    d_model: int = 192
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 768
    conv_kernel: int = 7
    dropout: float = 0.1
    attention_dropout: float = 0.1
    positional_encoding: str = "sinusoidal_learned"
    max_sequence_length: int = 512


@dataclass(frozen=True)
class Phase2Config:
    """Complete, immutable runtime configuration for one preprocessing run.

    Every sub-object is frozen so no stage can mutate another stage's
    parameters mid-run.
    """

    dataset_root: Path
    config_version: str
    config_hash: str
    # Per-module typed configuration objects.
    ingestion: IngestionConfig
    tracking: TrackingConfig
    appearance: AppearanceConfig
    geometry: GeometryConfig
    motion: MotionConfig
    fusion: FusionConfig
    manifest: ManifestConfig
    # Aggregates.
    quality: QualityConfig
    splits: SplitConfig
    # Phase 4 temporal encoder (Conformer) parameters.
    temporal: TemporalConfig
    # Pipeline aggregate (carries config_version / config_hash).
    preprocessing: PreprocessingConfig


def _compute_config_hash(parsed: Dict[str, Any], config_version: str) -> str:
    """Stable hash over the effective configuration (sorted keys)."""
    payload = {"config_version": config_version, "config": parsed}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_typed(raw: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Phase2Config:
    # Apply CLI overrides first (highest precedence).
    if overrides:
        raw = _deep_merge_overrides(raw, overrides)

    _check_unknown_fields(raw)

    dataset_root = Path(str(raw.get("dataset_root", "alphabet_dataset")))
    config_version = str(raw.get("config_version", "1.0"))

    video = _parse_video(raw)
    ft = _parse_face_tracking(raw)
    ma = _parse_mouth_alignment(raw)
    app = _parse_appearance(raw)
    geo = _parse_geometry(raw)
    mot = _parse_motion(raw)
    quality = _parse_quality(raw)
    prep = _parse_preprocessing(raw)
    splits = _parse_splits(raw)
    manifest = _parse_manifest(raw)

    # Phase 3 fused representation dimension (canonical FusionConfig value);
    # the temporal encoder input must match it exactly.
    fused_dim = FusionConfig().fused_dim
    temporal = _parse_temporal(raw, fused_dim=fused_dim)

    effective = {
        "video": video, "face_tracking": ft, "mouth_alignment": ma,
        "appearance": app, "geometry": geo, "motion": mot,
        "quality": quality, "preprocessing": prep, "splits": splits,
        "manifest": manifest, "temporal": temporal,
    }
    config_hash = _compute_config_hash(effective, config_version)

    brightness = tuple(int(v) for v in quality["brightness_range"])

    appearance_cfg = AppearanceConfig(
        crop_size=ma["appearance_crop_size"],
        normalization_mean=tuple(app.get("normalization_mean", (0.485, 0.456, 0.406))),
        normalization_std=tuple(app.get("normalization_std", (0.229, 0.224, 0.225))),
        color_space=app.get("color_space", "RGB"),
    )

    geometry_cfg = GeometryConfig(
        num_scalars=geo["num_scalars"],
        mlp_hidden=geo["mlp_hidden"],
        mlp_output=geo["mlp_output"],
        normalization_scale=ma["normalization_scale"],
    )

    motion_cfg = MotionConfig(
        savgol_window=geo["savgol_window"],
        savgol_polyorder=geo["savgol_polyorder"],
        optical_flow_enabled=mot["optical_flow_enabled"],
        flow_pyr_scale=mot["flow_pyr_scale"],
        flow_levels=mot["flow_levels"],
        flow_winsize=mot["flow_winsize"],
        flow_iterations=mot["flow_iterations"],
        flow_poly_n=mot["flow_poly_n"],
        flow_poly_sigma=mot["flow_poly_sigma"],
        flow_flags=mot["flow_flags"],
        mlp_hidden=mot["mlp_hidden"],
        mlp_output=mot["mlp_output"],
        flow_patch_size=mot["flow_patch_size"],
        num_landmarks=ma.get("num_mouth_landmarks", 20),
        num_scalars=geo["num_scalars"],
    )

    fusion_cfg = FusionConfig(
        appearance_dim=appearance_cfg.output_dim,
        geometry_dim=geometry_cfg.mlp_output,
        motion_dim=motion_cfg.mlp_output,
        fused_dim=192,
        use_context=True,
        temperature=1.0,
    )

    tracking_cfg = TrackingConfig(
        detection_confidence=ft["detection_confidence"],
        tracking_confidence=ft["tracking_confidence"],
        max_num_faces=ft["max_num_faces"],
        refine_landmarks=ft["refine_landmarks"],
        min_frames_with_face=ft["min_frames_with_face"],
        max_missing_face_gap=ft["max_missing_face_gap"],
        normalization_scale=ma["normalization_scale"],
        crop_margin_multiplier=ma["crop_margin_multiplier"],
        appearance_crop_size=ma["appearance_crop_size"],
        context_crop_size=ma["context_crop_size"],
        num_mouth_landmarks=ma.get("num_mouth_landmarks", 20),
        mouth_landmark_indices=ma.get("mouth_landmark_indices", []),
        blur_threshold=quality["blur_threshold"],
        brightness_range=brightness,
        min_mouth_size=quality["min_mouth_size"],
        max_pose_angle=quality["max_pose_angle"],
        save_frames=prep["save_frames"],
        save_landmarks_npz=prep["save_landmarks_npz"],
        save_crops=prep["save_crops"],
        preprocessing_version=prep["version"],
    )

    ingestion_cfg = IngestionConfig(
        dataset_root=dataset_root,
        fps=video["fps"],
        max_sequence_length=video["max_sequence_length"],
        min_sequence_length=video["min_sequence_length"],
        input_resolution=video["input_resolution"],
        codec=video["codec"],
        blur_threshold=quality["blur_threshold"],
        brightness_range=brightness,
        min_mouth_size=quality["min_mouth_size"],
        max_pose_angle=quality["max_pose_angle"],
        duplicate_ssim_threshold=quality["duplicate_ssim_threshold"],
        save_frames=prep["save_frames"],
        save_landmarks_npz=prep["save_landmarks_npz"],
        save_crops=prep["save_crops"],
        preprocessing_version=prep["version"],
        config_version=config_version,
        config_hash=config_hash,
    )

    preprocessing_cfg = PreprocessingConfig(
        dataset_root=dataset_root,
        fps=video["fps"],
        max_sequence_length=video["max_sequence_length"],
        min_sequence_length=video["min_sequence_length"],
        input_resolution=video["input_resolution"],
        detection_confidence=ft["detection_confidence"],
        tracking_confidence=ft["tracking_confidence"],
        max_num_faces=ft["max_num_faces"],
        refine_landmarks=ft["refine_landmarks"],
        crop_margin_multiplier=ma["crop_margin_multiplier"],
        appearance_crop_size=ma["appearance_crop_size"],
        context_crop_size=ma["context_crop_size"],
        blur_threshold=quality["blur_threshold"],
        brightness_range=brightness,
        min_mouth_size=quality["min_mouth_size"],
        max_pose_angle=quality["max_pose_angle"],
        save_frames=prep["save_frames"],
        save_landmarks_npz=prep["save_landmarks_npz"],
        save_crops=prep["save_crops"],
        preprocessing_version=prep["version"],
        config_version=config_version,
        config_hash=config_hash,
        train_ratio=splits["train_ratio"],
        val_ratio=splits["val_ratio"],
        test_ratio=splits["test_ratio"],
        random_seed=splits["random_seed"],
        min_speakers_per_split=splits["min_speakers_per_split"],
        resume=prep["resume"],
        manifest_filename=manifest["path"],
        appearance_output_dim=appearance_cfg.output_dim,
        geometry_num_scalars=geometry_cfg.num_scalars,
        geometry_mlp_hidden=geometry_cfg.mlp_hidden,
        geometry_mlp_output=geometry_cfg.mlp_output,
        geometry_normalization_scale=geometry_cfg.normalization_scale,
        motion_savgol_window=motion_cfg.savgol_window,
        motion_savgol_polyorder=motion_cfg.savgol_polyorder,
        motion_optical_flow_enabled=motion_cfg.optical_flow_enabled,
        motion_mlp_output=motion_cfg.mlp_output,
    )

    manifest_cfg = ManifestConfig(
        dataset_root=dataset_root,
        manifest_path=Path(manifest["path"]),
        schema_version=manifest["schema_version"],
    )

    return Phase2Config(
        dataset_root=dataset_root,
        config_version=config_version,
        config_hash=config_hash,
        ingestion=ingestion_cfg,
        tracking=tracking_cfg,
        appearance=appearance_cfg,
        geometry=geometry_cfg,
        motion=motion_cfg,
        fusion=fusion_cfg,
        manifest=manifest_cfg,
        quality=QualityConfig(
            blur_threshold=quality["blur_threshold"],
            brightness_range=brightness,
            min_mouth_size=quality["min_mouth_size"],
            max_pose_angle=quality["max_pose_angle"],
            duplicate_ssim_threshold=quality["duplicate_ssim_threshold"],
        ),
        splits=SplitConfig(
            train_ratio=splits["train_ratio"],
            val_ratio=splits["val_ratio"],
            test_ratio=splits["test_ratio"],
            random_seed=splits["random_seed"],
            min_speakers_per_split=splits["min_speakers_per_split"],
        ),
        temporal=TemporalConfig(**temporal),
        preprocessing=preprocessing_cfg,
    )


def load_phase2_config(path: Path | str = DEFAULT_CONFIG_PATH,
                       overrides: Optional[Dict[str, Any]] = None) -> Phase2Config:
    """Load, validate and build the typed Phase 2 configuration.

    Parameters
    ----------
    path:
        Path to the YAML configuration file. Missing file -> hard failure.
    overrides:
        Mapping of dotted keys (e.g. ``{"mouth_alignment.crop_margin_multiplier": 3.0}``)
        to apply on top of the YAML values (CLI override > YAML > default).
    """
    path = Path(path)
    if not path.exists():
        raise ConfigValidationError(f"Configuration file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigValidationError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigValidationError(
            f"Configuration root must be a mapping, got {type(raw).__name__}")
    return _build_typed(raw, overrides)


def load_config_dict(path: Path | str = DEFAULT_CONFIG_PATH,
                     overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load the raw YAML as a dict (validated), applying dotted-key overrides."""
    path = Path(path)
    if not path.exists():
        raise ConfigValidationError(f"Configuration file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigValidationError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigValidationError(
            f"Configuration root must be a mapping, got {type(raw).__name__}")
    if overrides:
        raw = _deep_merge_overrides(raw, overrides)
    _check_unknown_fields(raw)
    return raw