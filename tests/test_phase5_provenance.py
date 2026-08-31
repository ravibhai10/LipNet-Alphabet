import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphabet.dataset import (
    AlphabetSequenceDataset, RepresentationCacheError, SampleMeta, cache_key,
)
from alphabet.vocab import DEFAULT_VOCABULARY


class TestDatasetProvenance:
    """Provenance enforcement: the dataset refuses stale/mismatched caches."""

    def _fixture(self, tmp_path, config_hash="GOODHASH"):
        rel = "recordings/A/s1/sess1/rec000"
        rec_dir = tmp_path / cache_key(rel)
        rec_dir.mkdir(parents=True)
        fused = np.ones((12, 192), dtype=np.float32)
        np.save(rec_dir / "fused.npy", fused)
        (rec_dir / "metadata.json").write_text(json.dumps(
            {"recording_id": "r0", "config_hash": config_hash,
             "representation_version": "1.0", "frame_count": 12}))
        return rel, rec_dir

    def _entry(self, rel, letter="A"):
        """Create a lightweight ManifestEntry-like object for testing."""
        return SimpleNamespace(
            recording_id="r0", speaker_id="s1", session_id="sess1",
            letter=letter, path=rel, frame_count=12, duration_ms=400.0,
            quality_summary={}, split="train",
        )

    def test_matching_hash_loads(self, tmp_path):
        rel, rec_dir = self._fixture(tmp_path, "GOODHASH")
        ds = AlphabetSequenceDataset(
            [self._entry(rel)], dataset_root=tmp_path, repr_cache_dir=tmp_path,
            expected_config_hash="GOODHASH", vocab=DEFAULT_VOCABULARY)
        item = ds[0]
        assert item["fused"].shape == (12, 192)
        assert item["target_ids"] == [1]  # A == 1

    def test_stale_hash_rejected(self, tmp_path):
        rel, rec_dir = self._fixture(tmp_path, "OLDHASH")
        ds = AlphabetSequenceDataset(
            [self._entry(rel)], dataset_root=tmp_path, repr_cache_dir=tmp_path,
            expected_config_hash="GOODHASH", vocab=DEFAULT_VOCABULARY)
        with pytest.raises(RepresentationCacheError, match="stale"):
            ds[0]

    def test_missing_cache_rejected(self, tmp_path):
        ds = AlphabetSequenceDataset(
            [self._entry("recordings/A/s1/sess1/missing")],
            dataset_root=tmp_path, repr_cache_dir=tmp_path,
            expected_config_hash="GOODHASH", vocab=DEFAULT_VOCABULARY)
        with pytest.raises(RepresentationCacheError, match="no cached"):
            ds[0]

    def test_nan_features_rejected(self, tmp_path):
        rel, rec_dir = self._fixture(tmp_path, "GOODHASH")
        fused = np.load(rec_dir / "fused.npy")
        fused[3, 7] = np.nan
        np.save(rec_dir / "fused.npy", fused)
        ds = AlphabetSequenceDataset(
            [self._entry(rel)], dataset_root=tmp_path, repr_cache_dir=tmp_path,
            expected_config_hash="GOODHASH", vocab=DEFAULT_VOCABULARY)
        with pytest.raises(RepresentationCacheError, match="NaN"):
            ds[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
