"""Tests for the Phase 3 gated multistream representation pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from repr.pipeline import (
    AlignmentError,
    CacheProvenanceError,
    GatedFusionWithAblation,
    NumericalValidationError,
    RepresentationPipeline,
    RepresentationPipelineConfig,
    StreamAblation,
    StreamNormalizer,
    apply_ablation,
    batch_sequences,
    validate_stream_inputs,
)

T = 12
DA, DG, DM, DF = 96, 64, 64, 192


class MockExtractor:
    """Deterministic identity-style extractor for fast unit tests."""

    def __init__(self, dim):
        self.dim = dim

    def extract_sequence(self, *args, **kwargs):
        x = args[0]
        arr = np.asarray(x, dtype=np.float32).reshape(len(x), -1)
        if arr.shape[-1] >= self.dim:
            out = arr[:, : self.dim]
        else:
            reps = int(np.ceil(self.dim / max(arr.shape[-1], 1)))
            out = np.tile(arr, (1, reps))[:, : self.dim]
        return out.astype(np.float32)


@pytest.fixture(scope="module")
def pipeline() -> RepresentationPipeline:
    cfg = RepresentationPipelineConfig(
        appearance_dim=DA, geometry_dim=DG, motion_dim=DM, fused_dim=DF)
    return RepresentationPipeline(
        cfg,
        appearance_extractor=MockExtractor(DA),
        geometry_extractor=MockExtractor(DG),
        motion_extractor=MockExtractor(DM),
    )


@pytest.fixture(scope="module")
def inputs():
    rng = np.random.default_rng(42)
    crops = [rng.random((16, 16, 3)).astype(np.float32) for _ in range(T)]
    landmarks = rng.random((T, 20, 2)).astype(np.float32) * 0.1 + 0.45
    scalars = rng.random((T, 20)).astype(np.float32)
    frame_indices = np.arange(T, dtype=np.int64)
    timestamps = (np.arange(T, dtype=np.float64) + 1.0) * 33.333
    return crops, landmarks, scalars, frame_indices, timestamps


# ---------------------------------------------------------------------------
# Input contract validation
# ---------------------------------------------------------------------------

class TestInputContract:
    def test_valid_streams_pass(self):
        a = torch.randn(T, DA)
        g = torch.randn(T, DG)
        m = torch.randn(T, DM)
        shape = validate_stream_inputs(a, g, m)
        assert shape == (T,)

    def test_nan_rejected(self):
        a = torch.randn(T, DA)
        a[3, 5] = float("nan")
        with pytest.raises(NumericalValidationError, match="NaN or Inf"):
            validate_stream_inputs(a, torch.randn(T, DG), torch.randn(T, DM))

    def test_inf_rejected(self):
        g = torch.randn(T, DG)
        g[0, 0] = float("inf")
        with pytest.raises(NumericalValidationError, match="NaN or Inf"):
            validate_stream_inputs(torch.randn(T, DA), g, torch.randn(T, DM))

    def test_zero_length_rejected(self):
        with pytest.raises(AlignmentError, match="zero-length sequence"):
            validate_stream_inputs(
                torch.zeros(0, DA), torch.zeros(0, DG), torch.zeros(0, DM))

    def test_zero_dim_rejected(self):
        with pytest.raises(AlignmentError, match="zero embedding dimension"):
            validate_stream_inputs(
                torch.zeros(T, 0), torch.zeros(T, DG), torch.zeros(T, DM))

    def test_frame_count_mismatch_rejected(self):
        with pytest.raises(AlignmentError, match="Frame-count mismatch"):
            validate_stream_inputs(
                torch.randn(T, DA), torch.randn(T + 1, DG), torch.randn(T, DM))

    def test_wrong_ndim_rejected(self):
        with pytest.raises(AlignmentError, match=r"shape \[T, D\] or \[B, T, D\]"):
            validate_stream_inputs(
                torch.randn(T, DA), torch.randn(T), torch.randn(T, DM))

    def test_mismatched_timestamps_rejected(self):
        ts = np.linspace(0, 100, T - 1)  # one short
        with pytest.raises(AlignmentError, match="Timestamp count"):
            validate_stream_inputs(
                torch.randn(T, DA), torch.randn(T, DG), torch.randn(T, DM),
                timestamps=ts)

    def test_non_increasing_timestamps_rejected(self):
        ts = np.arange(T, dtype=np.float64) * 33.0
        ts[5] = ts[4]  # duplicate -> not strictly increasing
        with pytest.raises(AlignmentError, match="strictly increasing"):
            validate_stream_inputs(
                torch.randn(T, DA), torch.randn(T, DG), torch.randn(T, DM),
                timestamps=ts)


# ---------------------------------------------------------------------------
# Shapes, gates, normalization
# ---------------------------------------------------------------------------

class TestShapesAndGates:
    def test_end_to_end_shapes(self, pipeline, inputs):
        crops, landmarks, scalars, fi, ts = inputs
        result, timings = pipeline.process_recording(
            "rec_test", crops, landmarks, scalars, fi, ts)

        assert result.fused.shape == (T, DF)
        assert result.appearance.shape == (T, DA)
        assert result.geometry.shape == (T, DG)
        assert result.motion.shape == (T, DM)
        assert result.gates.shape == (T, 3)
        assert result.valid_length == T

    def test_gate_properties(self, pipeline, inputs):
        crops, landmarks, scalars, fi, ts = inputs
        result, _ = pipeline.process_recording(
            "rec_test", crops, landmarks, scalars, fi, ts)

        sums = result.gates.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-4)
        assert (result.gates >= 0).all()
        # Softmax over 3 streams: each weight strictly positive and < 1.
        assert (result.gates > 0).all() and (result.gates < 1).all()

    def test_finite_outputs(self, pipeline, inputs):
        crops, landmarks, scalars, fi, ts = inputs
        result, _ = pipeline.process_recording(
            "rec_test", crops, landmarks, scalars, fi, ts)
        for name in ("fused", "appearance", "geometry", "motion", "gates"):
            arr = getattr(result, name)
            assert np.isfinite(arr).all(), f"{name} contains NaN/Inf"

    def test_temporal_structure_preserved(self):
        """Fusion must not mix information across timesteps.

        Uses normalization="none": per-sequence z-score statistics are a
        deliberate global affine transform, so locality is verified on the
        raw streams where a single-frame perturbation must affect exactly
        one fused frame.
        """
        rng = np.random.default_rng(42)
        crops = [rng.random((16, 16, 3)).astype(np.float32) for _ in range(T)]
        landmarks = (rng.random((T, 20, 2)).astype(np.float32) * 0.1 + 0.45)
        scalars = rng.random((T, 20)).astype(np.float32)
        fi = np.arange(T, dtype=np.int64)
        ts = (np.arange(T, dtype=np.float64) + 1.0) * 33.333

        cfg = RepresentationPipelineConfig(
            appearance_dim=DA, geometry_dim=DG, motion_dim=DM, fused_dim=DF,
            normalization="none")
        pipe = RepresentationPipeline(
            cfg,
            appearance_extractor=MockExtractor(DA),
            geometry_extractor=MockExtractor(DG),
            motion_extractor=MockExtractor(DM))

        r1, _ = pipe.process_recording("rec_t1", crops, landmarks, scalars, fi, ts)

        crops2 = list(crops)
        crops2[7] = np.zeros_like(crops[7])
        r2, _ = pipe.process_recording("rec_t2", crops2, landmarks, scalars, fi, ts)

        diff_per_frame = np.abs(r1.fused - r2.fused).max(axis=-1)
        assert diff_per_frame[7] > 1e-6, "perturbed frame produced no change"
        others = np.delete(diff_per_frame, 7)
        assert others.max() < 1e-6, (
            f"perturbing t=7 changed other frames: max={others.max()}")

    def test_normalizer_deterministic_and_per_stream(self):
        rng = np.random.default_rng(0)
        x = rng.normal(loc=5.0, scale=3.0, size=(50, 8))
        n1 = StreamNormalizer("standard")
        out1 = n1.fit_transform("geometry", x)
        n2 = StreamNormalizer("standard")
        out2 = n2.fit_transform("geometry", x)
        np.testing.assert_allclose(out1, out2)  # deterministic
        np.testing.assert_allclose(out1.mean(axis=0), 0.0, atol=1e-4)
        np.testing.assert_allclose(out1.std(axis=0), 1.0, atol=1e-3)

    def test_normalizer_state_roundtrip(self):
        x = np.random.default_rng(1).normal(size=(30, 5))
        n1 = StreamNormalizer("standard").fit("motion", x)
        state = n1.state_dict()
        n2 = StreamNormalizer("none")
        n2.load_state_dict(state)
        np.testing.assert_allclose(n1.transform("motion", x),
                                   n2.transform("motion", x))

    def test_normalizer_unknown_method_rejected(self):
        with pytest.raises(Exception, match="Unknown normalization method"):
            StreamNormalizer("quantum")

    def test_ablation_zeroes_disabled_streams(self):
        a, g, m = torch.randn(T, DA), torch.randn(T, DG), torch.randn(T, DM)
        abl = StreamAblation(use_appearance=True, use_geometry=False,
                             use_motion=False)
        a2, g2, m2 = apply_ablation(a, g, m, abl)
        assert torch.equal(a2, a)
        assert (g2 == 0).all() and (m2 == 0).all()

    def test_ablation_appearance_only_end_to_end(self, inputs):
        cfg = RepresentationPipelineConfig(
            appearance_dim=DA, geometry_dim=DG, motion_dim=DM, fused_dim=DF,
            ablation=StreamAblation(use_appearance=True, use_geometry=False,
                                    use_motion=False))
        pipe = RepresentationPipeline(
            cfg,
            appearance_extractor=MockExtractor(DA),
            geometry_extractor=MockExtractor(DG),
            motion_extractor=MockExtractor(DM))
        crops, landmarks, scalars, fi, ts = inputs
        result, _ = pipe.process_recording("rec_abl", crops, landmarks,
                                           scalars, fi, ts)
        assert result.fused.shape == (T, DF)
        assert np.isfinite(result.fused).all()
        sums = result.gates.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-4)

    def test_all_streams_disabled_rejected(self):
        """Fusion with every stream disabled must fail loudly, not emit zeros."""
        cfg = RepresentationPipelineConfig(
            appearance_dim=DA, geometry_dim=DG, motion_dim=DM, fused_dim=DF,
            ablation=StreamAblation(False, False, False))
        z_a = torch.zeros(1, T, DA)
        z_g = torch.zeros(1, T, DG)
        z_m = torch.zeros(1, T, DM)
        with pytest.raises(Exception, match="all three streams are disabled"):
            fusion = GatedFusionWithAblation(cfg)
            fusion(z_a, z_g, z_m)


# ---------------------------------------------------------------------------
# Variable-length sequences + batching
# ---------------------------------------------------------------------------

class TestBatching:
    def test_pad_and_mask(self):
        seqs = [np.random.randn(T, DF), np.random.randn(T - 4, DF),
                np.random.randn(T - 8, DF)]
        batch, mask = batch_sequences(seqs)
        assert batch.shape == (3, T, DF)
        assert mask.shape == (3, T)
        assert mask.dtype == bool

    def test_padding_does_not_alter_valid_frames(self):
        s = np.random.randn(T, DF).astype(np.float32)
        shorter = np.random.randn(6, DF).astype(np.float32)
        batch, mask = batch_sequences([s, shorter])
        # Row 0 fully valid: identical to input.
        assert (mask[0] == True).all()  # noqa: E712
        np.testing.assert_array_equal(batch[0], s)
        # Padded region of row 1 is zeros AND masked out.
        assert not mask[1, 6:].any()
        assert (batch[1, 6:] == 0).all()
        np.testing.assert_array_equal(batch[1, :6], shorter)

    def test_empty_batch_rejected(self):
        with pytest.raises(AlignmentError, match="zero sequences"):
            batch_sequences([])

    def test_zero_length_sequence_in_batch_rejected(self):
        with pytest.raises(AlignmentError, match="Zero-length sequence"):
            batch_sequences([np.random.randn(T, DF), np.zeros((0, DF))])

    def test_inconsistent_dims_rejected(self):
        with pytest.raises(AlignmentError, match="Inconsistent embedding dims"):
            batch_sequences([np.random.randn(T, DF), np.random.randn(T, 7)])

    def test_wrong_mask_shape_rejected(self):
        am = np.ones((2, T + 5), dtype=bool)
        with pytest.raises(AlignmentError, match="attention_mask shape"):
            batch_sequences([np.random.randn(T, DF)] * 2, attention_mask=am)


# ---------------------------------------------------------------------------
# Cache provenance
# ---------------------------------------------------------------------------

class TestCacheProvenance:
    def test_cache_roundtrip_and_stale_hash_rejected(
            self, pipeline, inputs, tmp_path):
        crops, landmarks, scalars, fi, ts = inputs
        result, _ = pipeline.process_recording(
            "rec_cache", crops, landmarks, scalars, fi, ts)
        result.config_hash = "deadbeef"

        cache_dir = tmp_path / "representations"
        path = pipeline.save_to_cache(result, cache_dir)
        assert (path / "metadata.json").is_file()

        # Correct hash loads fine.
        loaded = pipeline.load_from_cache("rec_cache", cache_dir,
                                          config_hash="deadbeef",
                                          representation_version=result.representation_version)
        np.testing.assert_allclose(loaded.fused, result.fused)

        # Wrong hash must be refused — never silently reused.
        with pytest.raises(CacheProvenanceError, match="Refusing stale reuse"):
            pipeline.load_from_cache("rec_cache", cache_dir,
                                     config_hash="cafebabe")

        # Wrong representation version likewise.
        with pytest.raises(CacheProvenanceError, match="representation version"):
            pipeline.load_from_cache("rec_cache", cache_dir,
                                     config_hash="deadbeef",
                                     representation_version="9.9")

    def test_cache_metadata_contains_required_fields(
            self, pipeline, inputs, tmp_path):
        crops, landmarks, scalars, fi, ts = inputs
        result, _ = pipeline.process_recording(
            "rec_meta", crops, landmarks, scalars, fi, ts)
        path = pipeline.save_to_cache(result, tmp_path / "repr_cache")
        import json
        meta = json.loads((path / "metadata.json").read_text())
        for key in ("recording_id", "config_hash",
                    "representation_version", "frame_count"):
            assert key in meta, f"cache metadata missing '{key}'"
        assert meta["recording_id"] == "rec_meta"
        assert meta["frame_count"] == T

    def test_load_missing_recording_rejected(self, pipeline, tmp_path):
        with pytest.raises(CacheProvenanceError, match="No cached representation"):
            pipeline.load_from_cache("ghost", tmp_path, config_hash="x")


# ---------------------------------------------------------------------------
# TemporalEncoder interface + gradient flow
# ---------------------------------------------------------------------------

class TestTemporalEncoderInterface:
    def test_to_temporal_encoder_input(self, pipeline, inputs):
        crops, landmarks, scalars, fi, ts = inputs
        result, _ = pipeline.process_recording(
            "rec_iface", crops, landmarks, scalars, fi, ts)

        seq, valid_len = result.to_temporal_encoder_input()
        assert isinstance(seq, np.ndarray)
        assert seq.shape == (valid_len, DF)
        assert valid_len == T
        # Interface contract for the future TemporalEncoder: [T, D_fused].
        assert seq.ndim == 2

    def test_gradient_flow_through_fusion(self, inputs):
        cfg = RepresentationPipelineConfig(
            appearance_dim=DA, geometry_dim=DG, motion_dim=DM, fused_dim=DF)
        fusion = GatedFusionWithAblation(cfg)
        fusion.train()

        a = torch.randn(2, 5, DA, requires_grad=True)
        g = torch.randn(2, 5, DG, requires_grad=True)
        m = torch.randn(2, 5, DM, requires_grad=True)

        fused, gates = fusion(a, g, m)
        loss = fused.square().mean() + gates.square().mean()
        loss.backward()

        for name, t in (("appearance", a), ("geometry", g), ("motion", m)):
            assert t.grad is not None, f"no gradient reached {name}"
            assert torch.isfinite(t.grad).all(), f"{name} grad has NaN/Inf"
            assert t.grad.abs().sum() > 0, f"{name} grad is all zeros"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
