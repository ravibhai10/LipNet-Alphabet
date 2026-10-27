"""Safe interface-level tests for the AV-HuBERT adapter.

These tests verify the adapter contract (input format, output projection,
error handling) using **mocked model outputs**. They run without the real
checkpoint and without fairseq installed. A real checkpoint smoke test is
a separate step (pending official asset download + fairseq availability),
per policy: never claim the model loads until actually tested.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.avhubert.adapter import AVHubertFeatureAdapter
from pipeline.avhubert.loader import ModelUnavailableError, AVHubertLoader


def _make_adapter(d_out: int = 768, n_frames: int = 63):
    class FakeLoader(AVHubertLoader):
        def __init__(self):
            super().__init__(ckpt_path=None)
            self.available = True
            self.d_out = d_out
            self.n_frames = n_frames

        def encode(self, video_path):
            buf = np.arange(self.n_frames * 64, dtype=np.float32).reshape(self.n_frames, 64)
            return buf.astype(np.float32)

    projection = np.ones((192, 768), dtype=np.float32)
    return AVHubertFeatureAdapter(loader=FakeLoader(), projection=projection), FakeLoader()


def test_adapter_encode_contract():
    adapter, _ = _make_adapter()
    arr = adapter.encode("dummy.mp4")
    assert arr.ndim == 2
    assert adapter.loader.d_out == 768
    assert adapter.loader.n_frames == 63


def test_adapter_project_contract():
    adapter, _ = _make_adapter()
    feats = np.random.default_rng(1).normal(size=(63, 768)).astype(np.float32)
    out = adapter.project(feats, to_dim=192)
    assert out.shape == (63, 192)
    assert not np.allclose(out, 0.0)


def test_projection_preserves_rank():
    adapter = AVHubertFeatureAdapter(
        projection=np.eye(192, dtype=np.float32),
    )
    batch = np.random.default_rng(1).normal(size=(10, 192)).astype(np.float32)
    out = adapter.project(batch, to_dim=192)
    assert out.shape == (10, 192)


def test_invalid_input_dimension():
    adapter, _ = _make_adapter()
    with pytest.raises(ValueError):
        adapter.project(np.zeros(64), to_dim=192)


def test_model_unavailable_when_not_loaded():
    adapter = AVHubertFeatureAdapter()
    with pytest.raises(ModelUnavailableError):
        adapter.encode("dummy.mp4")


def test_projection_requires_explicit_trained_matrix():
    adapter = AVHubertFeatureAdapter()
    with pytest.raises(ModelUnavailableError, match="untrained fallback"):
        adapter.project(np.zeros((2, 64), dtype=np.float32), to_dim=192)


def test_loader_uses_project_level_checkpoint_directory():
    loader = AVHubertLoader()
    assert loader._models_root == (
        loader._models_root.parents[2] / "models" / "external" / "avhubert"
    )
