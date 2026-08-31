"""
Gated fusion module for multi-stream representation.

Per DESIGN.md:
- Learned gated fusion, not concatenation
- At each frame: α_t = softmax(FPN(v_t; g_t; m_t; c_t))
- f_t = α_v W_v v_t + α_g W_g g_t + α_m W_m m_t + α_c W_c c_t, each W to ℝ^192
- Learns per-timestep weighting of streams
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """Configuration for gated fusion."""
    appearance_dim: int = 96
    geometry_dim: int = 64
    motion_dim: int = 64
    context_dim: int = 64
    fused_dim: int = 192
    # Number of streams (appearance, geometry, motion [, context]). If None,
    # derived from ``use_context`` (4 with context, 3 without).
    num_streams: Optional[int] = None
    use_context: bool = True
    temperature: float = 1.0  # Softmax temperature

    def __post_init__(self):
        if self.num_streams is None:
            self.num_streams = 4 if self.use_context else 3


def _context_or_zeros(
    appearance: torch.Tensor,
    context: Optional[torch.Tensor],
    context_dim: int
) -> torch.Tensor:
    """Return the context stream, or zero-fill it when unavailable yet enabled.

    This keeps the fixed-size FPN input dimension consistent (no hard crash)
    when ``use_context=True`` but no context tensor is supplied at runtime.
    """
    if context is not None:
        return context
    B, T = appearance.shape[:2]
    return torch.zeros(
        (B, T, context_dim), dtype=appearance.dtype, device=appearance.device
    )


class FeatureProjectionNetwork(nn.Module):
    """
    Feature Projection Network (FPN) for computing fusion weights.

    Takes concatenated stream features and outputs per-stream weights.
    """

    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config

        # Input dimension = sum of stream dimensions
        input_dim = config.appearance_dim + config.geometry_dim + config.motion_dim
        if config.use_context:
            input_dim += config.context_dim

        # Small MLP to compute gating logits
        self.net = nn.Sequential(
            nn.Linear(input_dim, config.fused_dim),
            nn.LayerNorm(config.fused_dim),
            nn.GELU(),
            nn.Linear(config.fused_dim, config.num_streams),
        )

    def forward(
        self,
        appearance: torch.Tensor,   # [B, T, 96]
        geometry: torch.Tensor,     # [B, T, 64]
        motion: torch.Tensor,       # [B, T, 64]
        context: Optional[torch.Tensor] = None  # [B, T, 64]
    ) -> torch.Tensor:
        """
        Compute fusion weights for each stream at each timestep.

        Args:
            appearance: [B, T, 96]
            geometry: [B, T, 64]
            motion: [B, T, 64]
            context: [B, T, 64] or None

        Returns:
            [B, T, num_streams] weights (softmax normalized)
        """
        B, T = appearance.shape[:2]

        # Concatenate features (zero-fill context when enabled but absent)
        feats = [appearance, geometry, motion]
        if self.config.use_context:
            feats.append(
                _context_or_zeros(appearance, context, self.config.context_dim)
            )

        concat = torch.cat(feats, dim=-1)  # [B, T, input_dim]

        # Compute logits
        logits = self.net(concat)  # [B, T, num_streams]

        # Temperature-scaled softmax
        weights = F.softmax(logits / self.config.temperature, dim=-1)

        return weights


class StreamProjector(nn.Module):
    """
    Project each stream to fused dimension.

    Each stream has its own projection: W_v, W_g, W_m, W_c → ℝ^192
    """

    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config

        self.proj_appearance = nn.Linear(config.appearance_dim, config.fused_dim)
        self.proj_geometry = nn.Linear(config.geometry_dim, config.fused_dim)
        self.proj_motion = nn.Linear(config.motion_dim, config.fused_dim)

        if config.use_context:
            self.proj_context = nn.Linear(config.context_dim, config.fused_dim)
        else:
            self.proj_context = None

    def forward(
        self,
        appearance: torch.Tensor,   # [B, T, 96]
        geometry: torch.Tensor,     # [B, T, 64]
        motion: torch.Tensor,       # [B, T, 64]
        context: Optional[torch.Tensor] = None  # [B, T, 64]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Project all streams to fused dimension.

        Returns:
            Tuple of projected tensors, each [B, T, 192]
        """
        v_proj = self.proj_appearance(appearance)
        g_proj = self.proj_geometry(geometry)
        m_proj = self.proj_motion(motion)

        if self.config.use_context and self.proj_context is not None:
            c_proj = self.proj_context(
                _context_or_zeros(appearance, context, self.config.context_dim)
            )
        else:
            c_proj = None

        return v_proj, g_proj, m_proj, c_proj


class GatedFusion(nn.Module):
    """
    Gated fusion module.

    Computes: f_t = Σ α_i * W_i * x_i
    where α = softmax(FPN([v_t; g_t; m_t; c_t]))
    """

    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config

        self.fpn = FeatureProjectionNetwork(config)
        self.projectors = StreamProjector(config)

    def forward(
        self,
        appearance: torch.Tensor,   # [B, T, 96]
        geometry: torch.Tensor,     # [B, T, 64]
        motion: torch.Tensor,       # [B, T, 64]
        context: Optional[torch.Tensor] = None  # [B, T, 64]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse streams with learned gating.

        Args:
            appearance: [B, T, 96]
            geometry: [B, T, 64]
            motion: [B, T, 64]
            context: [B, T, 64] or None

        Returns:
            fused: [B, T, 192] fused representation
            weights: [B, T, num_streams] fusion weights
        """
        # Compute fusion weights
        weights = self.fpn(appearance, geometry, motion, context)  # [B, T, num_streams]

        # Project streams
        v_proj, g_proj, m_proj, c_proj = self.projectors(
            appearance, geometry, motion, context
        )

        # Weighted sum
        fused = (
            weights[:, :, 0:1] * v_proj +
            weights[:, :, 1:2] * g_proj +
            weights[:, :, 2:3] * m_proj
        )

        if self.config.use_context and c_proj is not None:
            fused = fused + weights[:, :, 3:4] * c_proj

        return fused, weights


class GatedFusionWithResidual(nn.Module):
    """
    Gated fusion with residual connections for training stability.

    f_t = x_t + Σ α_i * W_i * x_i
    where x_t is a base projection (e.g., appearance)
    """

    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config

        self.fpn = FeatureProjectionNetwork(config)
        self.projectors = StreamProjector(config)

        # Base projection (residual)
        self.base_proj = nn.Linear(config.appearance_dim, config.fused_dim)

    def forward(
        self,
        appearance: torch.Tensor,   # [B, T, 96]
        geometry: torch.Tensor,     # [B, T, 64]
        motion: torch.Tensor,       # [B, T, 64]
        context: Optional[torch.Tensor] = None  # [B, T, 64]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse streams with learned gating and residual.

        Returns:
            fused: [B, T, 192] fused representation
            weights: [B, T, num_streams] fusion weights
        """
        # Compute fusion weights
        weights = self.fpn(appearance, geometry, motion, context)

        # Project streams
        v_proj, g_proj, m_proj, c_proj = self.projectors(
            appearance, geometry, motion, context
        )

        # Base residual
        base = self.base_proj(appearance)

        # Weighted sum
        fused = base + (
            weights[:, :, 0:1] * v_proj +
            weights[:, :, 1:2] * g_proj +
            weights[:, :, 2:3] * m_proj
        )

        if self.config.use_context and c_proj is not None:
            fused = fused + weights[:, :, 3:4] * c_proj

        return fused, weights


def create_fusion_module(
    config: FusionConfig,
    use_residual: bool = False
) -> nn.Module:
    """
    Factory function to create fusion module.

    Args:
        config: Fusion configuration
        use_residual: Whether to use residual connection variant

    Returns:
        Fusion module
    """
    if use_residual:
        return GatedFusionWithResidual(config)
    return GatedFusion(config)


# Standalone function for non-torch usage (inference with numpy arrays)
def fuse_streams_numpy(
    appearance: np.ndarray,   # [T, 96]
    geometry: np.ndarray,     # [T, 64]
    motion: np.ndarray,       # [T, 64]
    context: Optional[np.ndarray] = None,  # [T, 64]
    weights: Optional[np.ndarray] = None,  # [T, 4] precomputed
    appearance_proj: Optional[np.ndarray] = None,  # [96, 192]
    geometry_proj: Optional[np.ndarray] = None,    # [64, 192]
    motion_proj: Optional[np.ndarray] = None,      # [64, 192]
    context_proj: Optional[np.ndarray] = None,     # [64, 192]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse streams using numpy (for inference without torch).

    Args:
        Stream features and optional projection matrices

    Returns:
        fused: [T, 192] fused representation
        weights: [T, 4] fusion weights used
    """
    T = appearance.shape[0]

    # Default weights (equal) if not provided
    if weights is None:
        weights = np.ones((T, 4), dtype=np.float32) / 4

    # Default projections (random if not provided - should be loaded from model)
    if appearance_proj is None:
        appearance_proj = np.eye(96, 192, dtype=np.float32)
    if geometry_proj is None:
        geometry_proj = np.eye(64, 192, dtype=np.float32)
    if motion_proj is None:
        motion_proj = np.eye(64, 192, dtype=np.float32)
    if context_proj is None:
        context_proj = np.eye(64, 192, dtype=np.float32)

    # Project
    v_proj = appearance @ appearance_proj  # [T, 192]
    g_proj = geometry @ geometry_proj      # [T, 192]
    m_proj = motion @ motion_proj          # [T, 192]

    fused = (
        weights[:, 0:1] * v_proj +
        weights[:, 1:2] * g_proj +
        weights[:, 2:3] * m_proj
    )

    if context is not None:
        c_proj = context @ context_proj
        fused = fused + weights[:, 3:4] * c_proj

    return fused, weights