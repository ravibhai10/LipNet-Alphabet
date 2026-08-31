"""Phase 5 tests: CTC correctness, decoding, losses."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphabet.decoding import ClassMappingGreedyCTCDecoder, GreedyCTCDecoder
from alphabet.losses import MultitaskLoss, ctc_loss
from alphabet.targets import collate_targets
from alphabet.vocab import DEFAULT_VOCABULARY

V = DEFAULT_VOCABULARY.size  # 27


def _logits_from_pattern(patterns, t_extra=4):
    """Build [1, T, V] logits that argmax to the given per-frame tokens."""
    t = len(patterns) + t_extra
    logits = torch.full((1, t, V), -5.0)
    for i, tok in enumerate(patterns):
        logits[0, i, tok] = 8.0
    return logits


class TestCTCLoss:
    def test_shapes_and_lengths_contract(self):
        torch.manual_seed(0)
        b, t = 4, 20
        logits = torch.randn(b, t, V)
        mask = torch.ones(b, t, dtype=torch.bool)
        mask[0, 15:] = False  # variable lengths
        mask[1, 9:] = False
        tb = collate_targets(DEFAULT_VOCABULARY, ["A", "HELLO", "LL", "Z"])
        loss = ctc_loss(logits, mask, tb.targets, tb.target_lengths)
        assert loss.ndim == 0 and torch.isfinite(loss)

    def test_target_longer_than_input_rejected(self):
        logits = torch.randn(1, 3, V)
        mask = torch.ones(1, 3, dtype=torch.bool)
        targets = torch.tensor([[12, 12, 12, 12]])  # S=4 > T=3
        with pytest.raises(ValueError, match="target longer than input"):
            ctc_loss(logits, mask, targets, torch.tensor([4]))

    def test_zero_valid_frames_rejected(self):
        logits = torch.randn(1, 5, V)
        mask = torch.zeros(1, 5, dtype=torch.bool)
        with pytest.raises(ValueError, match="zero valid frames"):
            ctc_loss(logits, mask, torch.tensor([[1]]), torch.tensor([1]))

    def test_padding_does_not_change_loss(self):
        """CTC must read only input_lengths frames — padding is inert."""
        torch.manual_seed(1)
        logits = torch.randn(2, 30, V)
        mask = torch.ones(2, 30, dtype=torch.bool)
        mask[0, 12:] = False
        mask[1, 20:] = False
        tb = collate_targets(DEFAULT_VOCABULARY, ["A", "HELLO"])
        l1 = ctc_loss(logits, mask, tb.targets, tb.target_lengths)
        logits2 = logits.clone()
        logits2[:, 20:, :] = 123.0  # garbage in pad region only
        l2 = ctc_loss(logits2, mask, tb.targets, tb.target_lengths)
        assert torch.allclose(l1, l2)

    def test_single_letter_target_loss_finite(self):
        logits = torch.randn(3, 25, V)
        mask = torch.ones(3, 25, dtype=torch.bool)
        tb = collate_targets(DEFAULT_VOCABULARY, ["A", "B", "C"])
        assert torch.isfinite(ctc_loss(logits, mask, tb.targets, tb.target_lengths))

    def test_repeated_letters_more_constrained_than_single(self):
        """LL needs a blank between the two Ls -> strictly harder alignment.

        Construct logits where every frame strongly favors class 12 (L) with
        blanks suppressed. For target [L] the all-L path is valid (CTC collapses
        consecutive identical non-blank labels), so probability is high and loss
        is low. For target [LL] at least one blank frame must appear between the
        two L groups, but blanks are unlikely everywhere, so loss is higher.
        """
        logits = torch.full((1, 10, V), -5.0)
        logits[0, :, 12] = 5.0  # all frames strongly favor L; blank at idx 0 suppressed
        mask = torch.ones(1, 10, dtype=torch.bool)
        l_single = ctc_loss(logits, mask, torch.tensor([[12]]), torch.tensor([1]))
        l_double = ctc_loss(logits, mask, torch.tensor([[12, 12]]), torch.tensor([2]))
        assert torch.isfinite(l_single) and torch.isfinite(l_double)
        assert l_double > l_single  # blank required between Ls but suppressed -> harder


class TestGreedyDecoding:
    def test_example_from_spec(self):
        """blank H H blank E E blank L L blank -> H E L"""
        blank, H, E, L = 0, 8, 5, 12
        logits = _logits_from_pattern(
            [blank, H, H, blank, E, E, blank, L, L, blank])
        dec = GreedyCTCDecoder(blank).decode_single(logits, logits.shape[1])
        assert dec.text == "8512"  # base decoder emits class ids as strings
        assert dec.emission_frames == [1, 4, 7]

    def test_mapped_decoder_letters(self):
        blank, H, E, L = 0, 8, 5, 12
        logits = _logits_from_pattern(
            [blank, H, H, blank, E, E, blank, L, L, blank])
        dec = ClassMappingGreedyCTCDecoder(
            DEFAULT_VOCABULARY.index_to_letter, blank).decode_single(
            logits, logits.shape[1])
        assert dec.text == "HEL"

    def test_repeated_letter_survives_collapse(self):
        """blank L L blank L blank -> L L (distinct from L)."""
        blank, L = 0, 12
        logits = _logits_from_pattern([blank, L, L, blank, L, blank])
        dec = ClassMappingGreedyCTCDecoder(
            DEFAULT_VOCABULARY.index_to_letter, blank).decode_single(
            logits, logits.shape[1])
        assert dec.text == "LL"

    def test_all_blank_decodes_empty(self):
        logits = _logits_from_pattern([0, 0, 0, 0])
        dec = GreedyCTCDecoder(0).decode_single(logits, 4)
        assert dec.text == ""

    def test_respects_input_lengths(self):
        blank, A, B = 0, 1, 2
        logits = _logits_from_pattern([blank, A, blank, B, blank, blank])
        dec = ClassMappingGreedyCTCDecoder(
            DEFAULT_VOCABULARY.index_to_letter, blank)
        assert dec.decode_single(logits, 3).text == "A"
        assert dec.decode_single(logits, 6).text == "AB"

    def test_batch_decoding_with_variable_lengths(self):
        blank, A, B = 0, 1, 2
        l1 = _logits_from_pattern([blank, A, blank, B, blank])  # T=9
        l2 = _logits_from_pattern([blank, B, blank])  # T=7, pad to 9
        pad = torch.full((1, 2, l2.shape[-1]), -5.0)
        l2 = torch.cat([l2, pad], dim=1)
        logits = torch.cat([l1, l2], dim=0)
        lengths = torch.tensor([9, 7])
        decs = ClassMappingGreedyCTCDecoder(
            DEFAULT_VOCABULARY.index_to_letter, blank).decode_batch(
            logits, lengths)
        assert [d.text for d in decs] == ["AB", "B"]

    def test_invalid_lengths_rejected(self):
        logits = _logits_from_pattern([0, 1, 0])
        with pytest.raises(ValueError):
            GreedyCTCDecoder(0).decode_single(logits, 0)
        with pytest.raises(ValueError):
            GreedyCTCDecoder(0).decode_single(logits, 99)
