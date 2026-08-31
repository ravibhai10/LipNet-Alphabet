"""
Temporal encoder (Phase 4) — Conformer-style, per DESIGN.md §4.

Consumes the Phase 3 fused representation:
    x:      [B, T, 192] gated multistream representation
    mask:   [B, T] boolean attention mask (True = valid frame, False = padding)

Architecture (DESIGN.md "3. Letter-scale context (L2)"):
    - Conformer encoder, d_model = 192, 4 Conformer blocks
    - Each block: macaron FFN + multi-head self-attention + depthwise-conv
      module + second macaron FFN, with pre-norm residual connections.
    - Positional encoding: sinusoidal + learned

Temporal dimension is fully preserved. Padding never corrupts valid frames:
masked LayerNorm statistics, masked attention keys, and zeroed padded
positions inside the conv module.

This module contains NO phonetic/viseme/CTC/decoder heads.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


class TemporalInputError(ValueError):
    """Raised when temporal-encoder inputs violate the contract."""


@dataclass
class TemporalEncoderConfig:
    """Configuration for the conformer temporal encoder.

    All values are sourced from the YAML ``temporal:`` section via the
    canonical config loader. Defaults mirror the checked-in YAML.
    """

    encoder_type: str = "conformer"
    input_dim: int = 192          # Phase 3 fused dim (must match)
    d_model: int = 192            # Conformer model dimension
    num_layers: int = 4           # Number of Conformer blocks
    num_heads: int = 4            # Attention heads
    ffn_dim: int = 768            # Feed-forward expansion dimension
    conv_kernel: int = 7          # Depthwise conv kernel (odd)
    dropout: float = 0.1
    attention_dropout: float = 0.1
    positional_encoding: str = "sinusoidal_learned"  # sinusoidal | learned | sinusoidal_learned
    max_sequence_length: int = 512

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.input_dim <= 0:
            raise TemporalInputError("TemporalEncoderConfig dims must be positive")
        if self.num_heads <= 0:
            raise TemporalInputError("TemporalEncoderConfig num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise TemporalInputError(
                "temporal.num_heads must divide temporal.d_model "
                f"({self.d_model} % {self.num_heads} != 0)"
            )
        if self.conv_kernel % 2 == 0:
            raise TemporalInputError("temporal.conv_kernel must be odd")
        if self.conv_kernel < 1:
            raise TemporalInputError("temporal.conv_kernel must be >= 1")
        if self.positional_encoding not in ("sinusoidal", "learned", "sinusoidal_learned"):
            raise TemporalInputError(
                "temporal.positional_encoding must be one of "
                "('sinusoidal', 'learned', 'sinusoidal_learned'); got "
                f"{self.positional_encoding!r}"
            )


def validate_temporal_input(
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    config: TemporalEncoderConfig,
) -> None:
    """Enforce the Phase 4 input contract; raise TemporalInputError on failure."""
    if not isinstance(x, torch.Tensor):
        raise TemporalInputError(f"input must be a torch.Tensor, got {type(x).__name__}")
    if x.ndim != 3:
        raise TemporalInputError(
            f"input must be [B, T, D]; got {tuple(x.shape)}"
        )
    B, T, D = x.shape
    if D != config.input_dim:
        raise TemporalInputError(
            f"input feature dim {D} != config.input_dim {config.input_dim}"
        )
    if T == 0:
        raise TemporalInputError("zero-length sequence is not allowed")
    if not bool(torch.isfinite(x).all()):
        raise TemporalInputError("input contains NaN or Inf values")

    if mask is None:
        return
    if not isinstance(mask, torch.Tensor):
        raise TemporalInputError(f"mask must be a torch.Tensor, got {type(mask).__name__}")
    if tuple(mask.shape) != (B, T):
        raise TemporalInputError(
            f"mask shape {tuple(mask.shape)} != (B, T) = ({B}, {T})"
        )
    if mask.dtype != torch.bool:
        if not bool((mask.eq(0) | mask.eq(1)).all()):
            raise TemporalInputError("mask must contain only 0/1 or be boolean")
    if not bool((mask == 1).any()):
        raise TemporalInputError("all-padding sequence (mask has no True frame)")


class PositionalEncoding(nn.Module):
    """Sinusoidal + (optionally) learned positional encoding.

    Padded positions still receive positional vectors; the encoder zeroes
    padded positions after ``forward`` so they never influence later masked
    statistics.
    """

    def __init__(self, d_model: int, max_len: int, mode: str = "sinusoidal_learned",
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = int(max_len)
        self.mode = mode
        self.dropout = nn.Dropout(dropout)

        if mode in ("sinusoidal", "sinusoidal_learned"):
            pe = torch.zeros(self.max_len, d_model)
            position = torch.arange(self.max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model % 2 == 1:
                pe[:, 1::2] = torch.cos(position * div_term[:-1])
            else:
                pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("sinusoidal_pe", pe.unsqueeze(0))  # [1, max_len, D]
        else:
            self.register_buffer("sinusoidal_pe", torch.zeros(1, self.max_len, d_model))

        if mode in ("learned", "sinusoidal_learned"):
            init = torch.zeros(self.max_len, d_model)
            if mode == "sinusoidal_learned":
                init = self.sinusoidal_pe.squeeze(0).clone()
            self.learned_pe = nn.Parameter(init)
        else:
            self.learned_pe = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        if T > self.max_len:
            raise TemporalInputError(
                f"sequence length {T} > temporal.max_sequence_length "
                f"({self.max_len}); positional table is too small"
            )
        out = x
        if self.mode in ("sinusoidal", "sinusoidal_learned"):
            out = out + self.sinusoidal_pe[:, :T, :]
        if self.learned_pe is not None:
            out = out + self.learned_pe[:T, :].unsqueeze(0)
        return self.dropout(out)


class MaskedLayerNorm(nn.Module):
    """Per-position LayerNorm over the feature dimension.

    LayerNorm statistics are local to each timestep, so padding can never
    corrupt the normalization of valid frames by construction. The optional
    ``mask`` is accepted for API compatibility (callers pass it for uniform
    block signatures) and validated against the input shape, but does not
    alter the statistics -- this exactly matches the Conformer/LayerNorm
    specification and keeps masked and unmasked paths numerically identical
    on valid frames (verified to machine precision in float64).
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.size(-1) != self.d_model:
            raise TemporalInputError(
                f"MaskedLayerNorm expected feature dim {self.d_model}, "
                f"got {x.size(-1)}"
            )
        return F.layer_norm(x, (self.d_model,), self.weight, self.bias, self.eps)


class FeedForwardModule(nn.Module):
    """Macaron feed-forward: Linear -> GELU -> dropout -> Linear -> dropout.

    In a Conformer block it is applied twice at 0.5 weight (macaron FFN).
    """

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout2(self.w2(self.dropout1(self.act(self.w1(x)))))


class MultiHeadSelfAttention(nn.Module):
    """Masked multi-head self-attention (bidirectional, offline).

    Padded keys are masked out (logits set to a large negative), and padded
    query outputs are zeroed afterwards so padding never contributes.
    """

    def __init__(self, d_model: int, num_heads: int, attention_dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attention_dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # [3, B, H, T, hd]
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q * self.scale) @ k.transpose(-2, -1)  # [B, H, T, T]

        if mask is not None:
            # Masks keys: [B,1,1,T]; -1e9 (not -inf) keeps softmax finite for
            # fully-padded query rows, which are zeroed below.
            keys_mask = mask.to(scores.dtype).unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
            scores = scores.masked_fill(keys_mask == 0, -1e9)

        attn = self.attn_drop(torch.softmax(scores, dim=-1))
        out = attn @ v                                   # [B, H, T, hd]
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out(out)

        if mask is not None:
            out = out * mask.unsqueeze(-1)
        return out


class ConformerConvModule(nn.Module):
    """Depthwise-conv temporal module: pointwise -> GLU -> depthwise -> pointwise.

    Padded positions are zeroed on entry and exit so right-padding does not
    leak into valid frames regardless of pad length.
    """

    def __init__(self, d_model: int, kernel_size: int, dropout: float = 0.0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise TemporalInputError("conv_kernel must be odd")
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2, groups=d_model,
        )
        self.ln = MaskedLayerNorm(d_model)
        self.act = nn.GELU()
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        # Temporal-layout copy of the mask ([B, 1, T]) for re-masking inside
        # the transposed sections below.
        tm = mask.to(x.dtype).unsqueeze(1) if mask is not None else None

        x = x.transpose(1, 2)                            # [B, D, T]
        x = self.pointwise1(x)
        x, gate = x.chunk(2, dim=1)
        x = x * self.act(gate)                           # gated activation
        if tm is not None:
            # CRITICAL: pointwise1's bias refills padded positions (bias != 0
            # even for an all-zero input), so without this re-mask the
            # depthwise temporal conv would read non-zero junk from padding
            # into valid frames near the sequence boundary.
            x = x * tm
        x = self.depthwise(x)
        if tm is not None:
            # Depthwise bias likewise refills padded positions; re-zero so the
            # "padded outputs stay zero" invariant holds within the module and
            # no later stage can observe them.
            x = x * tm
        x = x.transpose(1, 2)                            # [B, T, D]
        x = self.ln(x, mask)
        x = self.act(x)
        x = x.transpose(1, 2)
        x = self.pointwise2(x)
        x = x.transpose(1, 2)                            # [B, T, D]
        x = self.dropout(x)

        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x


class ConformerBlock(nn.Module):
    """One Conformer block (macaron FFN + attention + conv + macaron FFN).

    Every sub-layer is pre-normalized with :class:`MaskedLayerNorm` and uses
    residual connections; the block output has padded positions zeroed.
    """

    def __init__(self, config: TemporalEncoderConfig):
        super().__init__()
        d = config.d_model
        # Macaron halves
        self.ffn_scale = 0.5
        self.norm_ffn1 = MaskedLayerNorm(d)
        self.ffn1 = FeedForwardModule(d, config.ffn_dim, config.dropout)

        self.norm_attn = MaskedLayerNorm(d)
        self.attn = MultiHeadSelfAttention(d, config.num_heads, config.attention_dropout)

        self.norm_conv = MaskedLayerNorm(d)
        self.conv = ConformerConvModule(d, config.conv_kernel, config.dropout)

        self.norm_ffn2 = MaskedLayerNorm(d)
        self.ffn2 = FeedForwardModule(d, config.ffn_dim, config.dropout)

        self.norm_final = MaskedLayerNorm(d)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.ffn_scale * self.ffn1(self.norm_ffn1(x, mask))
        x = x + self.attn(self.norm_attn(x, mask), mask)
        x = x + self.conv(self.norm_conv(x, mask), mask)
        x = x + self.ffn_scale * self.ffn2(self.norm_ffn2(x, mask))
        x = self.norm_final(x, mask)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x


class TemporalEncoder(nn.Module):
    """Conformer-style temporal encoder (DESIGN.md §4 L2).

    Input:
        x:    [B, T, input_dim]  (Phase 3 fused representation, input_dim=192)
        mask: [B, T] boolean (True = valid frame); None => all valid

    Output:
        [B, T, d_model] (d_model == input_dim default 192), temporal dimension
        preserved; padded positions are zeroed in the output.
    """

    def __init__(self, config: TemporalEncoderConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.d_model) \
            if config.input_dim != config.d_model else nn.Identity()
        self.pos = PositionalEncoding(
            config.d_model, config.max_sequence_length,
            mode=config.positional_encoding, dropout=config.dropout,
        )
        self.blocks = nn.ModuleList(
            [ConformerBlock(config) for _ in range(config.num_layers)]
        )
        self.out_norm = MaskedLayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode [B, T, D_in] -> [B, T, D_model] (temporal dimension kept)."""
        validate_temporal_input(x, mask, self.config)
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)

        h = self.input_proj(x)
        h = self.pos(h)
        h = h * mask.unsqueeze(-1)

        for block in self.blocks:
            h = block(h, mask)

        h = self.out_norm(h, mask)
        h = h * mask.unsqueeze(-1)
        return h

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_temporal_model(
    config=None,
    from_phase2_config=None,
) -> TemporalEncoder:
    """Factory.

    Pass either a ``TemporalEncoderConfig`` directly, or a loaded Phase2Config
    whose ``temporal`` attribute is a :class:`TemporalEncoderConfig` or the
    config-layer frozen ``TemporalConfig`` (field names are identical, so it
    is adopted field-by-field without importing torch at config-load time).
    ``from_phase2_config`` wins if both given.
    """
    if from_phase2_config is not None:
        config = from_phase2_config.temporal
    if config is None:
        return TemporalEncoder(TemporalEncoderConfig())
    if isinstance(config, TemporalEncoderConfig):
        cfg = config
    else:
        import dataclasses
        cfg = TemporalEncoderConfig(**dataclasses.asdict(config))
    return TemporalEncoder(cfg)