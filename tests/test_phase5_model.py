"""Phase 5 tests: model forward/backward, parameter counting, aux heads."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphabet.heads import MultiTaskAlphabetModel
from alphabet.losses import MultitaskLoss, ctc_loss
from alphabet.targets import collate_targets
from alphabet.vocab import DEFAULT_VOCABULARY, DEFAULT_VOCABULARY as DEFAULT_VOCAB

V = DEFAULT_VOCABULARY.size  # 27


class TestModelAndGradientFlow:
    def _model(self, **over):
        from alphabet.heads import MultiTaskAlphabetModel
        from temporal.encoder import TemporalEncoderConfig
        cfg = TemporalEncoderConfig(d_model=64, ffn_dim=128, num_layers=2,
                                    input_dim=192, **over)
        return MultiTaskAlphabetModel(cfg, DEFAULT_VOCABULARY.size)

    def test_forward_shapes_and_masking(self):
        model = self._model()
        b, t = 3, 17
        x = torch.randn(b, t, 192)
        mask = torch.ones(b, t, dtype=torch.bool)
        mask[0, 11:] = False
        out = model(x, mask)
        assert out["letter_logits"].shape == (b, t, 27)
        assert torch.isfinite(out["letter_logits"]).all()
        assert (out["letter_logits"][0, 11:] == 0).all()  # padding zeroed

    def test_exact_parameter_count_matches_modules(self):
        from temporal.encoder import TemporalEncoderConfig
        from alphabet.heads import MultiTaskAlphabetModel
        cfg = TemporalEncoderConfig(d_model=64, ffn_dim=128, num_layers=2)
        model = MultiTaskAlphabetModel(cfg, DEFAULT_VOCABULARY.size)
        counts = model.parameter_counts()
        assert counts["total_trainable"] == sum(
            p.numel() for p in model.parameters())
        assert counts["encoder"] == sum(
            p.numel() for p in model.encoder.parameters())
        assert counts["letter_head"] == sum(
            p.numel() for p in model.letter_head.parameters())
        assert counts["phone_head"] == 0 and counts["viseme_head"] == 0

    def test_auxiliary_heads_absent_by_default_present_when_requested(self):
        from alphabet.heads import MultiTaskAlphabetModel
        from alphabet.phonetics import PhoneVocabulary, VisemeVocabulary
        from temporal.encoder import TemporalEncoderConfig
        cfg = TemporalEncoderConfig(d_model=64, ffn_dim=128, num_layers=2)
        m0 = MultiTaskAlphabetModel(cfg, DEFAULT_VOCABULARY.size)
        assert m0.phone_head is None and m0.viseme_head is None
        m1 = MultiTaskAlphabetModel(
            cfg, DEFAULT_VOCABULARY.size,
            phone_vocab_size=PhoneVocabulary().size,
            viseme_vocab_size=VisemeVocabulary().size)
        out = m1(torch.randn(1, 10, 192))
        assert out["phone_logits"].shape == (1, 10, PhoneVocabulary().size)
        assert out["viseme_logits"].shape == (1, 10, VisemeVocabulary().size)

    def test_gradient_flow_through_full_stack(self):
        model = self._model()
        x = torch.randn(2, 12, 192, requires_grad=True)
        mask = torch.ones(2, 12, dtype=torch.bool)
        mask[1, 8:] = False
        tb = collate_targets(DEFAULT_VOCABULARY, ["A", "HELLO"])
        loss = MultitaskLoss()(model(x, mask), mask, tb.targets,
                               tb.target_lengths)["total"]
        loss.backward()
        assert torch.isfinite(x.grad).all()
        params = list(model.parameters())
        grads = [p.grad for p in params]
        assert all(g is not None for g in grads)
        assert all(torch.isfinite(g).all() for g in grads)
        assert params[0].grad.abs().sum() > 0  # encoder weights receive gradient

    def test_nan_input_raises_not_silent(self):
        from temporal.encoder import TemporalInputError
        model = self._model()
        x = torch.randn(1, 10, 192)
        x[0, 3, 5] = float("nan")
        with pytest.raises(TemporalInputError):
            model(x, torch.ones(1, 10, dtype=torch.bool))

    def test_ctc_loss_raises_on_nonfinite(self):
        logits = torch.full((1, 5, V), float("inf"))
        mask = torch.ones(1, 5, dtype=torch.bool)
        with pytest.raises(FloatingPointError):
            ctc_loss(logits, mask, torch.tensor([[1]]), torch.tensor([1]))

    def test_multitask_loss_ignores_absent_aux_heads(self):
        model = self._model()
        mask = torch.ones(2, 12, dtype=torch.bool)
        tb = collate_targets(DEFAULT_VOCABULARY, ["A", "B"])
        parts = MultitaskLoss(lambda_phone=0.5, lambda_viseme=0.5)(
            model(torch.randn(2, 12, 192), mask), mask, tb.targets,
            tb.target_lengths)
        assert set(parts) == {"letter", "total"}
        assert torch.allclose(parts["letter"], parts["total"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
