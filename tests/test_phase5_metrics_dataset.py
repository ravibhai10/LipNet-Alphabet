"""Phase 5 tests: metrics + dataset collate/padding + split/provenance rules."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphabet.dataset import (
    AlphabetSequenceDataset, RepresentationCacheError, cache_key,
    collate_batch, load_manifest_entries, resolve_label,
)
from alphabet.metrics import (
    abstention_rate, cer, confusion_matrix, levenshtein, letter_accuracy,
    macro_f1, sequence_accuracy, top_k_accuracy,
)
from alphabet.vocab import DEFAULT_VOCABULARY


class TestMetrics:
    def test_levenshtein_and_cer(self):
        assert levenshtein("HELLO", "HELO") == 1
        assert cer("HELLO", "HELO") == pytest.approx(0.2)
        assert cer("A", "A") == 0.0
        assert cer("", "") == 0.0
        assert cer("", "A") == 1.0

    def test_accuracies_and_f1(self):
        v = DEFAULT_VOCABULARY
        trues = ["A", "B", "C", "D", "E"]
        preds = ["A", "B", "C", "E", "E"]
        assert letter_accuracy(trues, preds) == 0.8
        assert sequence_accuracy(trues, preds) == 0.8
        assert 0.0 < macro_f1(trues, preds, v) <= 1.0

    def test_top_k_accuracy(self):
        rows = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]
        labels = [1, 1]
        assert top_k_accuracy(rows, labels, 1) == 0.5  # only row1's argmax is 1
        assert top_k_accuracy(rows, labels, 2) == 1.0  # both rows contain 1 in top-2

    def test_confusion_matrix_shape_and_counts(self):
        v = DEFAULT_VOCABULARY
        m = confusion_matrix(["A", "B"], ["A", "A"], v)
        assert len(m) == 26
        assert m["A"][v.letters.index("A")] == 1
        assert sum(sum(row) for row in m.values()) == 2

    def test_abstention_rate(self):
        assert abstention_rate([True, False, False, True]) == 0.5


class TestDataset:
    def test_cache_key_is_path_unique(self):
        class E:
            path = "recordings/A/s1/sess1/rec000"
        assert cache_key(E()) == "recordings__A__s1__sess1__rec000"

    def test_collate_variable_lengths_and_targets(self):
        batch = [
            {"fused": np.ones((10, 192), dtype=np.float32), "length": 10,
             "target_ids": [1], "meta": None},
            {"fused": np.ones((4, 192), dtype=np.float32), "length": 4,
             "target_ids": [8, 5, 12], "meta": None},
        ]
        out = collate_batch(batch)
        assert out["fused"].shape == (2, 10, 192)
        assert out["mask"].tolist() == [[True] * 10, [True] * 4 + [False] * 6]
        assert out["input_lengths"].tolist() == [10, 4]
        assert out["targets"].tolist() == [[1, 0, 0], [8, 5, 12]]
        assert out["target_lengths"].tolist() == [1, 3]

    def test_resolve_label_supports_multiletter_metadata(self, tmp_path):
        class E:
            letter = "A"
            path = "recordings/A/s/x/r"
        rec = tmp_path / E.path
        rec.mkdir(parents=True)
        text, ttype = resolve_label(tmp_path, E())
        assert text == "A" and ttype.value == "single_letter"
        (rec / "metadata.json").write_text(json.dumps({"target_text": "HELLO"}))
        text, ttype = resolve_label(tmp_path, E())
        assert text == "HELLO" and ttype.value == "multi_letter"

    def test_manifest_requires_splits(self, tmp_path):
        (tmp_path / "manifest.jsonl").write_text(json.dumps({
            "recording_id": "r", "speaker_id": "s", "session_id": "x",
            "letter": "A", "path": "recordings/A/s/x/r", "frame_count": 10,
            "duration_ms": 300.0, "valid_frame_count": 10,
            "quality_summary": {}, "preprocessing_version": "1.0",
        }) + "\n")
        with pytest.raises(ValueError, match="no split"):
            load_manifest_entries(tmp_path, require_split=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
