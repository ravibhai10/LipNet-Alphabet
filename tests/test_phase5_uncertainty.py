"""Phase 5 tests: uncertainty (top-k/entropy/margin/abstention) + calibration."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphabet.calibration import (
    expected_calibration_error, fit_temperature, select_abstention_thresholds,
)
from alphabet.uncertainty import (
    aggregate_letter_posterior, decide_state, single_letter_uncertainty,
)
from alphabet.vocab import DEFAULT_VOCABULARY, OutputState

V = DEFAULT_VOCABULARY.size


def _logits_batch():
    torch.manual_seed(0)
    b, t = 3, 15
    logits = torch.randn(b, t, V) * 3
    mask = torch.ones(b, t, dtype=torch.bool)
    mask[0, 10:] = False  # variable length within batch
    return logits, mask


class TestUncertainty:
    def test_aggregate_posterior_valid_and_blank_excluded(self):
        logits, mask = _logits_batch()
        post = aggregate_letter_posterior(logits, mask)
        assert post.shape == (3, 26)
        assert torch.allclose(post.sum(-1), torch.ones(3), atol=1e-5)
        assert (post >= 0).all()

    def test_topk_entropy_margin_present(self):
        logits, mask = _logits_batch()
        for r in single_letter_uncertainty(logits, mask, DEFAULT_VOCABULARY):
            assert len(r.top_k) == 5
            assert all(l in DEFAULT_VOCABULARY.letters for l, _ in r.top_k)
            assert 0.0 <= r.confidence <= 1.0
            assert r.entropy >= 0.0
            assert 0.0 <= r.margin <= 1.0

    def test_confident_logits_give_high_confidence_low_entropy(self):
        logits, mask = _logits_batch()
        logits[:, :, 1] = 50.0  # strongly favor 'A' at every frame
        r = single_letter_uncertainty(logits, mask, DEFAULT_VOCABULARY)[0]
        assert r.prediction == "A"
        assert r.confidence > 0.99
        assert r.entropy < 0.1

    def test_temperature_sharpens_and_softens(self):
        logits, mask = _logits_batch()
        hot = single_letter_uncertainty(logits, mask, DEFAULT_VOCABULARY,
                                        temperature=10.0)[0]
        cold = single_letter_uncertainty(logits, mask, DEFAULT_VOCABULARY,
                                         temperature=0.5)[0]
        assert cold.confidence >= hot.confidence
        assert cold.entropy <= hot.entropy

    def test_abstention_state_machine(self):
        logits, mask = _logits_batch()
        results = single_letter_uncertainty(logits, mask, DEFAULT_VOCABULARY)
        r_unknown = decide_state(results[0], confidence_threshold=1.1,
                                 margin_threshold=0.0)
        assert r_unknown.state is OutputState.UNKNOWN and r_unknown.abstain
        r_amb = decide_state(results[1], confidence_threshold=0.0,
                             margin_threshold=1.1)
        assert r_amb.state is OutputState.AMBIGUOUS and r_amb.abstain
        r_ok = decide_state(results[2], confidence_threshold=0.0,
                            margin_threshold=0.0)
        assert r_ok.state is OutputState.LETTER and not r_ok.abstain


class TestCalibration:
    def test_fit_temperature_softens_overconfidence(self):
        """With (realistic) validation errors, overconfident logits need T>1."""
        torch.manual_seed(0)
        n, c = 300, 26
        base = torch.randn(n, c) * 2.0
        labels = base.argmax(-1).clone()
        wrong = torch.rand(n) < 0.25          # 25% validation errors
        labels[wrong] = (labels[wrong] + 1 + (torch.rand(wrong.sum()) * (c - 1)).long()) % c
        temp = fit_temperature(base * 5.0, labels)  # overconfident model
        assert temp > 1.0

    def test_ece_perfect_and_overconfident(self):
        conf = [0.9] * 100
        correct = [True] * 90 + [False] * 10
        assert abs(expected_calibration_error(conf, correct, 10)["ece"]) < 1e-9
        out_bad = expected_calibration_error([1.0] * 100, correct, 10)
        assert out_bad["ece"] > 0.05
        assert len(out_bad["bins"]) == 10

    def test_threshold_selection_respects_coverage(self):
        conf = [0.95] * 60 + [0.60] * 40
        marg = [0.5] * 100
        correct = [True] * 60 + [True] * 10 + [False] * 30
        t = select_abstention_thresholds(conf, marg, correct, coverage_target=0.6)
        assert t["confidence_threshold"] >= 0.6
        covered = sum(c >= t["confidence_threshold"] for c in conf) / 100
        assert covered >= 0.6
        assert t["validation_covered_accuracy"] >= 0.9

    def test_unreachable_coverage_reports_shortfall(self):
        t = select_abstention_thresholds([0.5] * 3, [0.1] * 3, [True] * 3,
                                         coverage_target=0.999)
        assert t["confidence_threshold"] == 0.5  # most permissive fallback


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
