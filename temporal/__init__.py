"""Temporal encoder package (Phase 4).

Implements the Conformer-style temporal encoder consuming the Phase 3
gated multistream representation [B, T, 192] + [B, T] boolean mask.

This is a module-level placeholder; the encoder lives in :mod:`temporal.encoder`
and the public API is re-exported here for convenience.
"""

from temporal.encoder import (
    TemporalEncoderConfig,
    TemporalEncoder,
    ConformerBlock,
    ConformerConvModule,
    FeedForwardModule,
    PositionalEncoding,
    MaskedLayerNorm,
    TemporalInputError,
    validate_temporal_input,
    build_temporal_model,
)

__all__ = [
    "TemporalEncoderConfig",
    "TemporalEncoder",
    "ConformerBlock",
    "ConformerConvModule",
    "FeedForwardModule",
    "PositionalEncoding",
    "MaskedLayerNorm",
    "TemporalInputError",
    "validate_temporal_input",
    "build_temporal_model",
]