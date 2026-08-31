"""
Tests for motion stream extraction.
"""

import pytest
import numpy as np
import torch
from repr.motion import MotionFeatureExtractor, MotionConfig, MotionMLP, MotionExtractor


class TestMotionFeatureExtractor:
    """Test motion feature computation."""

    def test_compute_derivatives_simple(self):
        config = MotionConfig(savgol_window=5, savgol_polyorder=2)
        extractor = MotionFeatureExtractor(config)

        # Simple linear sequence
        sequence = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.float32).T  # [6, 1]
        velocity, acceleration = extractor.compute_derivatives(sequence, axis=0)

        assert velocity.shape == (6, 1)
        assert acceleration.shape == (6, 1)

        # Velocity should be approximately 1 for linear sequence
        assert np.allclose(velocity[1:-1], 1.0, atol=0.5)

    def test_compute_derivatives_2d(self):
        config = MotionConfig(savgol_window=5, savgol_polyorder=2)
        extractor = MotionFeatureExtractor(config)

        # 2D sequence [T, D]
        T = 10
        sequence = np.zeros((T, 3), dtype=np.float32)
        sequence[:, 0] = np.arange(T) * 2  # Linear
        sequence[:, 1] = np.arange(T) ** 2  # Quadratic
        sequence[:, 2] = np.sin(np.arange(T))  # Sinusoidal

        velocity, acceleration = extractor.compute_derivatives(sequence, axis=0)

        assert velocity.shape == (T, 3)
        assert acceleration.shape == (T, 3)

    def test_compute_landmark_motion(self):
        config = MotionConfig()
        extractor = MotionFeatureExtractor(config)

        # Landmarks [T, 20, 2]
        T = 10
        landmarks = np.zeros((T, 20, 2), dtype=np.float32)
        # Simple horizontal movement
        for t in range(T):
            landmarks[t, :, 0] = t * 0.01

        vel, acc = extractor.compute_landmark_motion(landmarks)

        assert vel.shape == (T, 20, 2)
        assert acc.shape == (T, 20, 2)

    def test_compute_scalar_motion(self):
        config = MotionConfig()
        extractor = MotionFeatureExtractor(config)

        # Scalars [T, 20]
        T = 10
        scalars = np.zeros((T, 20), dtype=np.float32)
        scalars[:, 0] = np.arange(T) * 0.1  # Width increasing

        vel, acc = extractor.compute_scalar_motion(scalars)

        assert vel.shape == (T, 20)
        assert acc.shape == (T, 20)


class TestMotionMLP:
    """Test MotionMLP module."""

    def test_forward_without_flow(self):
        config = MotionConfig(optical_flow_enabled=False, mlp_output=64)
        mlp = MotionMLP(config)

        B = 4
        input_dim = 20 * 4 + 20 * 2  # 120
        features = torch.randn(B, input_dim)

        output = mlp(features)

        assert output.shape == (B, 64)

    def test_forward_with_flow(self):
        config = MotionConfig(optical_flow_enabled=True, mlp_output=64)
        mlp = MotionMLP(config)

        B = 4
        # 120 + 16*16*2 = 120 + 512 = 632
        input_dim = 20 * 4 + 20 * 2 + 16 * 16 * 2
        features = torch.randn(B, input_dim)

        output = mlp(features)

        assert output.shape == (B, 64)


class TestMotionExtractor:
    """Test MotionExtractor."""

    def test_extract_sequence(self):
        config = MotionConfig(optical_flow_enabled=False, mlp_output=64)
        extractor = MotionExtractor(config)

        T = 10
        landmarks = np.random.randn(T, 20, 2).astype(np.float32)
        scalars = np.random.randn(T, 20).astype(np.float32)

        embeddings = extractor.extract_sequence(landmarks, scalars)

        assert embeddings.shape == (T, 64)
        assert np.all(np.isfinite(embeddings))

    def test_extract_sequence_short(self):
        """Test with very short sequence."""
        config = MotionConfig(optical_flow_enabled=False)
        extractor = MotionExtractor(config)

        # Single frame
        landmarks = np.random.randn(1, 20, 2).astype(np.float32)
        scalars = np.random.randn(1, 20).astype(np.float32)

        embeddings = extractor.extract_sequence(landmarks, scalars)

        assert embeddings.shape == (1, 64)


import torch  # Import at bottom