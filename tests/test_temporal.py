"""Phase 4 tests — Conformer temporal encoder.

Covers: shapes/dimension preservation, masking & variable lengths,
padding-invariance, positional-order sensitivity, gradient flow,
numerical/input-contract failures, and YAML configuration integration.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_phase2_config
from config.loader import ConfigValidationError
from temporal.encoder import (
    ConformerBlock,
    TemporalEncoder,
    TemporalEncoderConfig,
    TemporalInputError,
    build_temporal_model,
)

torch.manual_seed(0)


def _make_encoder(**kwargs) -> TemporalEncoder:
    cfg_kwargs = dict(
        d_model=192, num_layers=2, num_heads=4, ffn_dim=256,
        conv_kernel=7, dropout=0.0, attention_dropout=0.0,
    )
    cfg_kwargs.update(kwargs)
    enc = build_temporal_model(TemporalEncoderConfig(**cfg_kwargs))
    enc.eval()
    return enc


# ---------------------------------------------------------------------------
# Shapes / architecture
# ---------------------------------------------------------------------------

class TestShapes:
    def test_output_shape_preserves_time(self):
        enc = _make_encoder()
        x = torch.randn(3, 40, 192)
        y = enc(x)
        assert tuple(y.shape) == (3, 40, 192), "temporal dimension collapsed"

    def test_block_preserves_dimension(self):
        cfg = TemporalEncoderConfig()
        blk = ConformerBlock(cfg)
        x = torch.randn(2, 12, cfg.d_model)
        y = blk(x)
        assert y.shape == x.shape

    def test_input_proj_when_dims_differ(self):
        enc = _make_encoder(input_dim=192, d_model=128, num_heads=8)
        y = enc(torch.randn(1, 20, 192))
        assert tuple(y.shape) == (1, 20, 128)

    def test_parameter_count_positive(self):
        enc = _make_encoder(d_model=192, num_layers=4)
        assert enc.num_parameters() > 100_000

    def test_block_count(self):
        enc = _make_encoder(num_layers=5)
        assert len(enc.blocks) == 5

    def test_default_config_matches_design(self):
        cfg = TemporalEncoderConfig()
        assert cfg.input_dim == 192 and cfg.d_model == 192
        assert cfg.num_layers == 4 and cfg.num_heads == 4
        assert cfg.ffn_dim == 768 and cfg.conv_kernel == 7


# ---------------------------------------------------------------------------
# Masking / variable lengths
# ---------------------------------------------------------------------------

class TestMasking:
    def test_variable_lengths_one_batch(self):
        """T=40, 80, 120 in the same batch must all work."""
        enc = _make_encoder()
        lens = [40, 80, 120]
        T = max(lens)
        x = torch.randn(3, T, 192)
        mask = torch.zeros(3, T, dtype=torch.bool)
        for i, L in enumerate(lens):
            mask[i, :L] = True
        y = enc(x, mask)
        assert tuple(y.shape) == (3, T, 192)
        for i, L in enumerate(lens):
            assert torch.count_nonzero(y[i, L:]) == 0, "padded output not zeroed"
            assert float(y[i, :L].abs().max()) > 0

    def test_padding_does_not_alter_valid_frames(self):
        """Doubling the amount of right-padding must not change valid outputs."""
        enc = _make_encoder()
        L, P = 30, 25
        base = torch.randn(1, L, 192)
        y1 = enc(torch.cat([base, torch.zeros(1, P, 192)], dim=1),
                 torch.tensor([[True] * L + [False] * P]))
        y2 = enc(torch.cat([base, torch.zeros(1, 2 * P, 192)], dim=1),
                 torch.tensor([[True] * L + [False] * 2 * P]))
        # Padded region contributes only zeros (zeroed conv input + masked
        # attention); residual float32 accumulation noise in the masked
        # reductions is ~1e-6, so exact equality is allclose at 1e-5.
        assert torch.allclose(y1[0, :L], y2[0, :L], atol=1e-5), \
            "padding corrupted valid frames"

    def test_padded_equals_unpadded_float64(self):
        """Decisive padding-isolation test: padded run vs UNPADDED run in float64.

        Comparing two padded runs can cancel symmetric leakage, so this test
        compares against the unpadded run at machine precision. Regression for
        a real bug where pointwise-conv biases refilled padded positions and
        the depthwise temporal conv leaked them into valid frames (~1e-3).
        """
        torch.manual_seed(0)
        enc = _make_encoder(num_layers=4, ffn_dim=768).double()
        L, P = 150, 25
        base = torch.randn(1, L, 192, dtype=torch.float64) * 5.0
        x_pad = torch.cat(
            [base, torch.zeros(1, P, 192, dtype=torch.float64)], dim=1)
        mask = torch.zeros(1, L + P, dtype=torch.bool)
        mask[0, :L] = True
        with torch.no_grad():
            y_ref = enc(base)
            y_pad = enc(x_pad, mask)
        diff = float((y_ref[0] - y_pad[0, :L]).abs().max())
        assert diff < 1e-10, \
            f"padding leaked into valid frames (float64 diff {diff:.2e})"

    def test_batch_independence(self):
        """Samples processed jointly equal processing each alone."""
        enc = _make_encoder()
        xa = torch.randn(1, 35, 192)
        xb = torch.randn(1, 35, 192)
        joint = enc(torch.cat([xa, xb]))
        solo_a = enc(xa)
        solo_b = enc(xb)
        assert torch.allclose(joint[0:1], solo_a, atol=1e-5)
        assert torch.allclose(joint[1:2], solo_b, atol=1e-5)

    def test_masked_run_matches_unmasked_on_valid_frames(self):
        """Regression: normalization must not change semantics with/without mask.

        Guards against a masked-LayerNorm implementation whose statistics
        differ between masked and unmasked paths (e.g. normalizing over time
        instead of features), which silently changes valid-frame outputs.
        """
        enc = _make_encoder()
        x = torch.randn(1, 30, 192)
        y_nomask = enc(x)
        mask = torch.ones(1, 30, dtype=torch.bool)
        y_mask = enc(x, mask)
        assert torch.allclose(y_nomask, y_mask, atol=1e-5), \
            "mask changed valid-frame outputs -> normalization not per-position"


# ---------------------------------------------------------------------------
# Positional information
# ---------------------------------------------------------------------------

class TestPositionalOrder:
    def test_reversal_changes_representation(self):
        """A,B,C vs C,B,A must not be treated as equivalent."""
        enc = _make_encoder()
        t = torch.randn(1, 9, 192)
        fwd = enc(t)
        rev = enc(t.flip(dims=[1]))
        diff = float((fwd - rev.flip(dims=[1])).abs().max())
        assert diff > 1e-3, "encoder appears order-invariant"

    def test_longer_than_max_len_fails(self):
        enc = _make_encoder(max_sequence_length=16)
        with pytest.raises(TemporalInputError, match="max_sequence_length"):
            enc(torch.randn(1, 32, 192))


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

class TestGradients:
    def test_gradient_flows_end_to_end(self):
        enc = _make_encoder()
        x = torch.randn(2, 24, 192, requires_grad=True)
        mask = torch.ones(2, 24, dtype=torch.bool)
        mask[1, 16:] = False
        y = enc(x, mask)
        y.sum().backward()
        assert x.grad is not None and bool(torch.isfinite(x.grad).all())
        assert float(x.grad.abs().sum()) > 0
        for name, p in enc.named_parameters():
            assert p.grad is not None, f"no gradient reached {name}"
            assert bool(torch.isfinite(p.grad).all()), f"non-finite grad at {name}"


# ---------------------------------------------------------------------------
# Numerical / contract failures
# ---------------------------------------------------------------------------

class TestFailures:
    def setup_method(self):
        self.enc = _make_encoder()

    def test_nan_input(self):
        x = torch.randn(1, 10, 192)
        x[0, 3, 5] = float("nan")
        with pytest.raises(TemporalInputError, match="NaN or Inf"):
            self.enc(x)

    def test_inf_input(self):
        x = torch.randn(1, 10, 192)
        x[0, 3, 5] = float("inf")
        with pytest.raises(TemporalInputError, match="NaN or Inf"):
            self.enc(x)

    def test_zero_length_sequence(self):
        with pytest.raises(TemporalInputError, match="zero-length"):
            self.enc(torch.randn(1, 0, 192))

    def test_all_padding_mask(self):
        with pytest.raises(TemporalInputError, match="all-padding|no True"):
            self.enc(torch.randn(1, 10, 192),
                     torch.zeros(1, 10, dtype=torch.bool))

    def test_invalid_mask_shape(self):
        with pytest.raises(TemporalInputError, match="mask shape"):
            self.enc(torch.randn(2, 10, 192),
                     torch.ones(2, 11, dtype=torch.bool))

    def test_mask_batch_mismatch(self):
        with pytest.raises(TemporalInputError, match="mask shape"):
            self.enc(torch.randn(2, 10, 192),
                     torch.ones(3, 10, dtype=torch.bool))

    def test_wrong_feature_dimension(self):
        with pytest.raises(TemporalInputError, match="feature dim"):
            self.enc(torch.randn(1, 10, 64))

    def test_non_3d_input(self):
        with pytest.raises(TemporalInputError):
            self.enc(torch.randn(10, 192))

    def test_non_bool_0_1_mask_rejected(self):
        m = torch.full((1, 10), 0.5)
        with pytest.raises(TemporalInputError, match="only 0/1"):
            self.enc(torch.randn(1, 10, 192), m)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_heads_must_divide_d_model(self):
        with pytest.raises(TemporalInputError, match="divide"):
            TemporalEncoderConfig(d_model=192, num_heads=5)

    def test_even_kernel_rejected(self):
        with pytest.raises(TemporalInputError, match="odd"):
            TemporalEncoderConfig(conv_kernel=6)

    def test_bad_positional_encoding_rejected(self):
        with pytest.raises(TemporalInputError, match="positional_encoding"):
            TemporalEncoderConfig(positional_encoding="rotary")


# ---------------------------------------------------------------------------
# YAML integration
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "alphabet_dataset.yaml"


class TestYamlIntegration:
    def test_yaml_drives_model_architecture(self):
        pcfg = load_phase2_config(CONFIG_PATH)
        assert pcfg.temporal.input_dim == 192
        assert pcfg.temporal.num_layers == 4
        model = build_temporal_model(from_phase2_config=pcfg)
        assert model.config.d_model == pcfg.temporal.d_model
        assert model.config.num_layers == pcfg.temporal.num_layers

    def test_changing_yaml_changes_runtime(self):
        a = load_phase2_config(CONFIG_PATH)
        b = load_phase2_config(
            CONFIG_PATH,
            overrides={"temporal.num_layers": 2, "temporal.ffn_dim": 512},
        )
        assert b.temporal.num_layers == 2 and b.temporal.ffn_dim == 512
        ma = build_temporal_model(a.temporal)
        mb = build_temporal_model(b.temporal)
        assert len(ma.blocks) == 4 and len(mb.blocks) == 2
        # Provenance: different effective configuration -> different hash.
        assert a.config_hash != b.config_hash

    def test_yaml_input_dim_mismatch_fails(self):
        with pytest.raises(ConfigValidationError, match="fusion output dimension"):
            load_phase2_config(
                CONFIG_PATH, overrides={"temporal.input_dim": 256}
            )

    def test_yaml_invalid_heads_fail(self):
        with pytest.raises(ConfigValidationError, match="divide"):
            load_phase2_config(
                CONFIG_PATH, overrides={"temporal.num_heads": 7}
            )

    def test_yaml_dropout_out_of_range_fails(self):
        with pytest.raises(ConfigValidationError, match=r"\[0, 1\]"):
            load_phase2_config(
                CONFIG_PATH, overrides={"temporal.dropout": 1.5}
            )

    def test_yaml_unknown_temporal_field_fails(self):
        with pytest.raises(ConfigValidationError, match="Unknown field in 'temporal'"):
            load_phase2_config(
                CONFIG_PATH, overrides={"temporal.stacked_layers": 3}
            )

    def test_real_shaped_forward_from_yaml(self):
        pcfg = load_phase2_config(CONFIG_PATH)
        model = build_temporal_model(pcfg.temporal)
        model.eval()
        with torch.no_grad():
            y = model(torch.randn(1, 50, 192))
        assert tuple(y.shape) == (1, 50, pcfg.temporal.d_model)
