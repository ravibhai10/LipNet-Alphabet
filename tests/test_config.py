"""Tests for the canonical configuration loader (config/loader.py).

Proves that ``configs/alphabet_dataset.yaml`` actually controls runtime
behavior — not merely parses.
"""

import copy
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.loader import (
    DEFAULT_CONFIG_PATH,
    ConfigValidationError,
    Phase2Config,
    load_phase2_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "alphabet_dataset.yaml"


@pytest.fixture(scope="module")
def base_yaml() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cfg() -> Phase2Config:
    return load_phase2_config(CONFIG_PATH)


def write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "test_config.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 1. YAML -> typed configuration mapping
# ---------------------------------------------------------------------------

class TestYamlToTypedMapping:
    def test_config_file_exists(self):
        assert CONFIG_PATH.exists(), f"Missing {CONFIG_PATH}"

    def test_crop_margin_from_yaml(self, cfg):
        assert abs(cfg.tracking.crop_margin_multiplier - 2.2) < 1e-9
        assert abs(cfg.preprocessing.crop_margin_multiplier - 2.2) < 1e-9

    def test_fps_from_yaml(self, cfg):
        assert cfg.ingestion.fps == 30
        assert cfg.preprocessing.fps == 30

    def test_resolution_from_yaml(self, cfg):
        assert tuple(cfg.ingestion.input_resolution) == (1280, 720)

    def test_quality_thresholds_from_yaml(self, cfg):
        assert cfg.tracking.blur_threshold == 100.0
        assert cfg.quality.blur_threshold == 100.0
        assert tuple(cfg.quality.brightness_range) == (20, 235)
        assert cfg.quality.min_mouth_size == 400
        assert cfg.quality.max_pose_angle == 30.0
        assert cfg.quality.duplicate_ssim_threshold == 0.98

    def test_tracking_params_from_yaml(self, cfg):
        assert cfg.tracking.detection_confidence == 0.5
        assert cfg.tracking.tracking_confidence == 0.5
        assert cfg.tracking.max_num_faces == 1
        assert cfg.tracking.refine_landmarks is True
        assert cfg.tracking.min_frames_with_face == 10
        assert cfg.tracking.max_missing_face_gap == 5

    def test_mouth_alignment_from_yaml(self, cfg):
        assert cfg.tracking.normalization_scale == "inner_mouth_width"
        assert tuple(cfg.tracking.appearance_crop_size) == (64, 64)
        assert tuple(cfg.tracking.context_crop_size) == (112, 112)

    def test_representation_dims_from_yaml(self, cfg):
        # Geometry: 20 scalars -> 96 hidden -> 64 output
        assert cfg.geometry.num_scalars == 20
        assert cfg.geometry.mlp_hidden == 96
        assert cfg.geometry.mlp_output == 64
        # Motion MLP output 64
        assert cfg.motion.mlp_output == 64
        # Appearance crop size shared with tracking
        assert tuple(cfg.appearance.crop_size) == (64, 64)
        # Fusion dims derived from stream outputs
        assert cfg.fusion.appearance_dim == 96
        assert cfg.fusion.geometry_dim == 64
        assert cfg.fusion.motion_dim == 64
        assert cfg.fusion.fused_dim == 192

    def test_splits_and_manifest_from_yaml(self, cfg):
        assert cfg.splits.train_ratio == 0.70
        assert cfg.splits.val_ratio == 0.15
        assert cfg.splits.test_ratio == 0.15
        assert cfg.splits.random_seed == 42
        assert cfg.manifest.manifest_path.name == "manifest.jsonl"
        assert cfg.manifest.schema_version == "1.0"

    def test_versions_recorded(self, cfg):
        assert cfg.config_version == "1.0"
        assert cfg.tracking.preprocessing_version == "1.0"
        assert len(cfg.config_hash) == 64  # sha256 hex digest


# ---------------------------------------------------------------------------
# 2. Changing YAML actually changes runtime behavior
# ---------------------------------------------------------------------------

class TestYamlDrivesRuntime:
    def test_crop_margin_change_propagates(self, tmp_path, base_yaml):
        """The task-mandated test: change 2.2 -> 3.0 in YAML, reload."""
        d = copy.deepcopy(base_yaml)
        d["mouth_alignment"]["crop_margin_multiplier"] = 3.0
        p = write_yaml(tmp_path, d)

        cfg = load_phase2_config(p)
        assert abs(cfg.tracking.crop_margin_multiplier - 3.0) < 1e-9
        assert abs(cfg.preprocessing.crop_margin_multiplier - 3.0) < 1e-9

        from data.preprocess import PreprocessingPipeline
        pipeline = PreprocessingPipeline.from_phase2_config(cfg)
        assert abs(pipeline.tracking_config.crop_margin_multiplier - 3.0) < 1e-9
        assert abs(pipeline.config.crop_margin_multiplier - 3.0) < 1e-9

    def test_fps_change_propagates(self, tmp_path, base_yaml):
        d = copy.deepcopy(base_yaml)
        d["video"]["fps"] = 60
        p = write_yaml(tmp_path, d)

        cfg = load_phase2_config(p)
        assert cfg.ingestion.fps == 60
        assert cfg.preprocessing.fps == 60

        from data.preprocess import PreprocessingPipeline
        pipeline = PreprocessingPipeline.from_phase2_config(cfg)
        assert pipeline.ingestion_config.fps == 60

    def test_blur_threshold_change_propagates(self, tmp_path, base_yaml):
        d = copy.deepcopy(base_yaml)
        d["quality"]["blur_threshold"] = 55.5
        p = write_yaml(tmp_path, d)

        cfg = load_phase2_config(p)
        assert cfg.quality.blur_threshold == 55.5
        assert cfg.tracking.blur_threshold == 55.5

        from data.preprocess import PreprocessingPipeline
        pipeline = PreprocessingPipeline.from_phase2_config(cfg)
        assert pipeline.tracking_config.blur_threshold == 55.5


# ---------------------------------------------------------------------------
# 3. Override precedence: CLI override > YAML > typed default
# ---------------------------------------------------------------------------

class TestOverridePrecedence:
    def test_override_beats_yaml(self, cfg):
        overridden = load_phase2_config(
            CONFIG_PATH, {"mouth_alignment.crop_margin_multiplier": 3.0})
        assert abs(overridden.tracking.crop_margin_multiplier - 3.0) < 1e-9
        # Original loaded config untouched
        assert abs(cfg.tracking.crop_margin_multiplier - 2.2) < 1e-9

    def test_single_key_override_leaves_rest_of_yaml(self):
        overridden = load_phase2_config(CONFIG_PATH, {"video.fps": 25})
        assert overridden.ingestion.fps == 25
        # Everything else still comes from YAML:
        assert abs(overridden.tracking.crop_margin_multiplier - 2.2) < 1e-9
        assert overridden.splits.train_ratio == 0.70

    def test_unknown_override_path_fails(self):
        with pytest.raises(ConfigValidationError, match="Unknown configuration override"):
            load_phase2_config(CONFIG_PATH, {"no_such_section.key": 1})

    def test_invalid_override_value_fails(self):
        with pytest.raises(ConfigValidationError):
            load_phase2_config(CONFIG_PATH, {"video.fps": -10})


# ---------------------------------------------------------------------------
# 4. Validation failures are explicit (never silent fallbacks)
# ---------------------------------------------------------------------------

class TestConfigValidationFailures:
    def _mutated(self, tmp_path, base_yaml, mutate):
        d = copy.deepcopy(base_yaml)
        mutate(d)
        return write_yaml(tmp_path, d)

    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigValidationError, match="not found"):
            load_phase2_config(tmp_path / "nonexistent.yaml")

    def test_missing_required_section(self, tmp_path, base_yaml):
        d = copy.deepcopy(base_yaml)
        del d["video"]
        p = write_yaml(tmp_path, d)
        with pytest.raises(ConfigValidationError, match="missing required section: 'video'"):
            load_phase2_config(p)

    def test_negative_fps(self, tmp_path, base_yaml):
        p = self._mutated(tmp_path, base_yaml,
                          lambda d: d["video"].update(fps=-5))
        with pytest.raises(ConfigValidationError, match=r"video\.fps.*positive"):
            load_phase2_config(p)

    def test_zero_crop_margin(self, tmp_path, base_yaml):
        p = self._mutated(tmp_path, base_yaml,
                          lambda d: d["mouth_alignment"].update(crop_margin_multiplier=0))
        with pytest.raises(ConfigValidationError, match="crop_margin_multiplier.*positive"):
            load_phase2_config(p)

    def test_confidence_out_of_range(self, tmp_path, base_yaml):
        p = self._mutated(tmp_path, base_yaml,
                          lambda d: d["face_tracking"].update(tracking_confidence=1.5))
        with pytest.raises(ConfigValidationError, match=r"tracking_confidence.*\[0\.0, 1\.0\]"):
            load_phase2_config(p)

    def test_wrong_type(self, tmp_path, base_yaml):
        p = self._mutated(tmp_path, base_yaml,
                          lambda d: d["quality"].update(blur_threshold="high"))
        with pytest.raises(ConfigValidationError, match="blur_threshold.*number"):
            load_phase2_config(p)

    def test_sequence_length_ordering(self, tmp_path, base_yaml):
        p = self._mutated(tmp_path, base_yaml,
                          lambda d: d["video"].update(min_sequence_length=999))
        with pytest.raises(ConfigValidationError, match="min_sequence_length"):
            load_phase2_config(p)

    def test_splits_must_sum_to_one(self, tmp_path, base_yaml):
        def mutate(d):
            d["splits"].update(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)
        p = self._mutated(tmp_path, base_yaml, mutate)
        with pytest.raises(ConfigValidationError, match="sum to 1.0"):
            load_phase2_config(p)

    def test_unknown_top_level_field(self, tmp_path, base_yaml):
        p = self._mutated(tmp_path, base_yaml,
                          lambda d: d.__setitem__("totally_unknown", {"x": 1}))
        with pytest.raises(ConfigValidationError, match="Unknown top-level"):
            load_phase2_config(p)

    def test_unknown_section_field(self, tmp_path, base_yaml):
        p = self._mutated(tmp_path, base_yaml,
                          lambda d: d["video"].update(not_a_real_field=1))
        with pytest.raises(ConfigValidationError, match="Unknown field in 'video'"):
            load_phase2_config(p)

    def test_malformed_yaml(self, tmp_path):
        p = tmp_path / "broken.yaml"
        p.write_text("video: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigValidationError, match="Invalid YAML"):
            load_phase2_config(p)

    def test_non_mapping_root(self, tmp_path):
        p = tmp_path / "list.yaml"
        p.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ConfigValidationError, match="root must be a mapping"):
            load_phase2_config(p)


# ---------------------------------------------------------------------------
# 5. Immutability + hash/versioning
# ---------------------------------------------------------------------------

class TestImmutabilityAndHash:
    def test_quality_config_is_frozen(self, cfg):
        with pytest.raises(Exception):
            cfg.quality.blur_threshold = 0.0  # type: ignore[misc]

    def test_split_config_is_frozen(self, cfg):
        with pytest.raises(Exception):
            cfg.splits.train_ratio = 0.99  # type: ignore[misc]

    def test_hash_stable_for_same_yaml(self):
        a = load_phase2_config(CONFIG_PATH)
        b = load_phase2_config(CONFIG_PATH)
        assert a.config_hash == b.config_hash

    def test_hash_changes_when_yaml_changes(self, tmp_path, base_yaml):
        original = load_phase2_config(CONFIG_PATH).config_hash
        d = copy.deepcopy(base_yaml)
        d["video"]["fps"] = 24
        changed = load_phase2_config(write_yaml(tmp_path, d)).config_hash
        assert changed != original

    def test_metadata_roundtrip_keeps_version_fields(self, tmp_path, cfg):
        from data.schema import (
            FaceTrackingInfo, MouthInfo, QualityMetrics, RecordingMetadata,
            DATA_SOURCE_SYNTHETIC_TEST,
        )
        meta = RecordingMetadata(
letter="a", speaker_id="speaker_001",
            session_id="session_001", recording_id="rec_test",
                            data_source=DATA_SOURCE_SYNTHETIC_TEST,
            source="webcam", created_at="2026-01-01T00:00:00",
            fps=30.0, frame_count=90, width=1280, height=720,
            duration_ms=3000.0, preprocessing_version="1.0",
            config_version=cfg.config_version, config_hash=cfg.config_hash,
            face_tracking=FaceTrackingInfo(), mouth=MouthInfo(),
            quality=QualityMetrics(),
        )
        out = tmp_path / "metadata.json"
        meta.save(out)
        loaded = RecordingMetadata.load(out)
        assert loaded.config_version == cfg.config_version
        assert loaded.config_hash == cfg.config_hash
        assert loaded.preprocessing_version == "1.0"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
