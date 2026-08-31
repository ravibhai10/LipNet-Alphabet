"""
Tests for gated fusion module.
"""

import pytest
import numpy as np
import torch
from repr.fusion import (
    FusionConfig, FeatureProjectionNetwork, StreamProjector,
    GatedFusion, GatedFusionWithResidual, create_fusion_module,
    fuse_streams_numpy
)


class TestFeatureProjectionNetwork:
    """Test FPN for fusion weights."""

    def test_forward(self):
        config = FusionConfig()
        fpn = FeatureProjectionNetwork(config)

        B, T = 2, 10
        appearance = torch.randn(B, T, 96)
        geometry = torch.randn(B, T, 64)
        motion = torch.randn(B, T, 64)
        context = torch.randn(B, T, 64)

        weights = fpn(appearance, geometry, motion, context)

        assert weights.shape == (B, T, 4)
        # Weights should sum to 1 (softmax)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(B, T))

    def test_forward_no_context(self):
        config = FusionConfig(use_context=False)
        fpn = FeatureProjectionNetwork(config)

        B, T = 2, 10
        appearance = torch.randn(B, T, 96)
        geometry = torch.randn(B, T, 64)
        motion = torch.randn(B, T, 64)

        weights = fpn(appearance, geometry, motion, None)

        assert weights.shape == (B, T, 3)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(B, T))


class TestStreamProjector:
    """Test stream projectors."""

    def test_forward(self):
        config = FusionConfig()
        projector = StreamProjector(config)

        B, T = 2, 10
        appearance = torch.randn(B, T, 96)
        geometry = torch.randn(B, T, 64)
        motion = torch.randn(B, T, 64)
        context = torch.randn(B, T, 64)

        v_proj, g_proj, m_proj, c_proj = projector(appearance, geometry, motion, context)

        assert v_proj.shape == (B, T, 192)
        assert g_proj.shape == (B, T, 192)
        assert m_proj.shape == (B, T, 192)
        assert c_proj.shape == (B, T, 192)

    def test_forward_no_context(self):
        config = FusionConfig(use_context=False)
        projector = StreamProjector(config)

        B, T = 2, 10
        appearance = torch.randn(B, T, 96)
        geometry = torch.randn(B, T, 64)
        motion = torch.randn(B, T, 64)

        v_proj, g_proj, m_proj, c_proj = projector(appearance, geometry, motion, None)

        assert c_proj is None


class TestGatedFusion:
    """Test gated fusion module."""

    def test_forward(self):
        config = FusionConfig()
        fusion = GatedFusion(config)

        B, T = 2, 10
        appearance = torch.randn(B, T, 96)
        geometry = torch.randn(B, T, 64)
        motion = torch.randn(B, T, 64)
        context = torch.randn(B, T, 64)

        fused, weights = fusion(appearance, geometry, motion, context)

        assert fused.shape == (B, T, 192)
        assert weights.shape == (B, T, 4)

    def test_forward_no_context(self):
        config = FusionConfig(use_context=False)
        fusion = GatedFusion(config)

        B, T = 2, 10
        appearance = torch.randn(B, T, 96)
        geometry = torch.randn(B, T, 64)
        motion = torch.randn(B, T, 64)

        fused, weights = fusion(appearance, geometry, motion, None)

        assert fused.shape == (B, T, 192)
        assert weights.shape == (B, T, 3)


class TestGatedFusionWithResidual:
    """Test gated fusion with residual."""

    def test_forward(self):
        config = FusionConfig(use_context=False)
        fusion = GatedFusionWithResidual(config)

        B, T = 2, 10
        appearance = torch.randn(B, T, 96)
        geometry = torch.randn(B, T, 64)
        motion = torch.randn(B, T, 64)

        fused, weights = fusion(appearance, geometry, motion, None)

        assert fused.shape == (B, T, 192)
        assert weights.shape == (B, T, 3)


class TestCreateFusionModule:
    """Test factory function."""

    def test_create_standard(self):
        config = FusionConfig()
        module = create_fusion_module(config, use_residual=False)
        assert isinstance(module, GatedFusion)

    def test_create_residual(self):
        config = FusionConfig()
        module = create_fusion_module(config, use_residual=True)
        assert isinstance(module, GatedFusionWithResidual)


class TestFuseStreamsNumpy:
    """Test numpy fusion function."""

    def test_fuse_basic(self):
        T = 10
        appearance = np.random.randn(T, 96).astype(np.float32)
        geometry = np.random.randn(T, 64).astype(np.float32)
        motion = np.random.randn(T, 64).astype(np.float32)

        fused, weights = fuse_streams_numpy(appearance, geometry, motion)

        assert fused.shape == (T, 192)
        assert weights.shape == (T, 4)
        # Default equal weights
        assert np.allclose(weights, 0.25)

    def test_fuse_with_context(self):
        T = 10
        appearance = np.random.randn(T, 96).astype(np.float32)
        geometry = np.random.randn(T, 64).astype(np.float32)
        motion = np.random.randn(T, 64).astype(np.float32)
        context = np.random.randn(T, 64).astype(np.float32)

        fused, weights = fuse_streams_numpy(appearance, geometry, motion, context)

        assert fused.shape == (T, 192)
        assert weights.shape == (T, 4)

    def test_fuse_custom_weights(self):
        T = 10
        appearance = np.random.randn(T, 96).astype(np.float32)
        geometry = np.random.randn(T, 64).astype(np.float32)
        motion = np.random.randn(T, 64).astype(np.float32)
        weights = np.array([[0.5, 0.3, 0.1, 0.1]] * T, dtype=np.float32)

        fused, w = fuse_streams_numpy(appearance, geometry, motion, weights=weights)

        assert np.allclose(w, weights)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])