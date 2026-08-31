"""Phase 5 §26: tiny-subset overfit — data -> target -> model -> loss wiring.

Uses clearly-labelled SYNTHETIC fused sequences (the dataset currently has no
recordings). Success criterion: train accuracy 1.0 with loss ~0, proving the
whole path is connected. Small model, CPU-friendly step budget.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphabet.config import DecoderConfig
from alphabet.sanity import make_synthetic_fused, tiny_overfit


class TestSyntheticData:
    def test_synthetic_prototypes_are_distinct_and_finite(self):
        seqs, labels = make_synthetic_fused(n_per_letter=2, t_frames=32,
                                            letters="ABC", seed=0)
        assert len(seqs) == 6 and len(labels) == 6
        for s in seqs:
            assert s.shape == (32, 192)
        assert all(ord(l) - ord("A") + 1 >= 1 for l in labels)
        # different letters -> different prototypes
        import numpy as np
        a0 = seqs[0][16]  # middle of the ridge
        b0 = seqs[2][16]
        assert not np.allclose(a0, b0)

    def test_silence_ends_and_letter_ridge_middle(self):
        seqs, _ = make_synthetic_fused(n_per_letter=1, t_frames=32,
                                       letters="A", seed=1)
        s = seqs[0]
        assert abs(s[2]).max() < 0.5     # leading silence ~ 0 + noise
        assert abs(s[16]).max() > 0.5    # letter ridge present


@pytest.mark.timeout(600)
class TestTinyOverfit:
    def test_overfit_reaches_full_train_accuracy(self):
        dec_cfg = DecoderConfig()  # aux heads disabled by default
        result = tiny_overfit(
            dec_cfg, phase2_cfg=_FakePhase2(),
            n_per_letter=2, t_frames=32, letters="ABCDEF",
            steps=400, lr=5e-3, seed=0, use_synthetic=True)
        assert result["status"] == "PASS", result
        assert result["final_accuracy"] == 1.0
        assert result["final_loss"] < 0.1
        assert result["steps_run"] <= 400

    def test_overfit_uses_expected_class_count(self):
        """Letter head must be 27-way; CTC blank participates in decoding."""
        from alphabet.heads import MultiTaskAlphabetModel
        from temporal.encoder import TemporalEncoderConfig
        model = MultiTaskAlphabetModel(
            TemporalEncoderConfig(d_model=64, ffn_dim=128, num_layers=2),
            DEFAULT_VOCAB().size)
        assert model.letter_head.proj.out_features == 27


class _FakePhase2:
    """Lightweight stand-in for the Phase 2 config used by tiny_overfit.

    Only ``dataset_root`` matters for the real-data path; ``temporal`` is a
    small TemporalEncoderConfig for the synthetic overfit wiring test.
    """

    dataset_root = "alphabet_dataset"

    from temporal.encoder import TemporalEncoderConfig
    temporal = TemporalEncoderConfig(d_model=64, ffn_dim=128, num_layers=2,
                                     input_dim=192)


def DEFAULT_VOCAB():
    from alphabet.vocab import DEFAULT_VOCABULARY
    return DEFAULT_VOCABULARY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
