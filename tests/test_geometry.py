"""
Tests for geometry stream extraction.
"""

import pytest
import numpy as np
import torch
from repr.geometry import GeometryScalars, GeometryConfig, GeometryMLP, GeometryExtractor


class TestGeometryScalars:
    """Test geometry scalar computation."""

    def create_test_landmarks(self) -> np.ndarray:
        """Create synthetic mouth landmarks [20, 2] in normalized coordinates."""
        # Create landmarks roughly in a mouth shape
        landmarks = np.zeros((20, 2), dtype=np.float32)

        # Outer lip corners
        landmarks[16] = [-0.5, 0.0]   # Left outer corner (48)
        landmarks[19] = [0.5, 0.0]    # Right outer corner (51)

        # Inner lip corners
        landmarks[8] = [-0.4, 0.0]    # Left inner corner (61)
        landmarks[15] = [0.4, 0.0]    # Right inner corner (68)

        # Upper lip
        landmarks[0] = [-0.35, -0.1]  # Left inner upper (13)
        landmarks[3] = [0.0, -0.15]   # Mid inner upper (16)
        landmarks[7] = [0.35, -0.1]   # Right inner upper (20)

        # Lower lip
        landmarks[11] = [0.0, 0.15]   # Mid inner lower (64)
        landmarks[14] = [0.0, 0.2]    # Bottom inner lower (67)

        return landmarks

    def test_compute_scalars(self):
        config = GeometryConfig()
        scalars_computer = GeometryScalars(config)
        landmarks = self.create_test_landmarks()

        scalars = scalars_computer.compute_scalars(landmarks)

        assert scalars.shape == (20,)
        assert np.all(np.isfinite(scalars))

        # Check specific values
        # Mouth width (outer corners)
        assert scalars[0] > 0  # mouth_width
        assert scalars[1] > 0  # mouth_height
        assert scalars[2] > 0  # wh_ratio

    def test_scalar_indices_meaning(self):
        """Test that scalar indices correspond to expected meanings."""
        config = GeometryConfig()
        scalars_computer = GeometryScalars(config)
        landmarks = self.create_test_landmarks()

        scalars = scalars_computer.compute_scalars(landmarks)

        # Index mapping (from GeometryScalars docstring):
        # 0: mouth_width
        # 1: mouth_height
        # 2: wh_ratio
        # 3: inner_opening
        # 4: lip_spread
        # 5: upper_lip_height
        # 6: lower_lip_height
        # 7: jaw_distance
        # 8: aperture_area
        # 9: upper_curvature
        # 10: lower_curvature
        # 11: corner_angle
        # 12: symmetry_h
        # 13: roundness
        # 14: upper_protrusion
        # 15: lower_protrusion
        # 16: inner_mouth_width
        # 17: inner_mouth_height
        # 18: lip_thickness
        # 19: jaw_drop

        # Verify some relationships
        # outer mouth_width should exceed inner_mouth_width (lip thickness)
        assert scalars[0] > scalars[16]  # mouth_width (outer) > inner_mouth_width
        assert scalars[4] == scalars[16]  # lip_spread == inner_mouth_width


class TestGeometryMLP:
    """Test GeometryMLP module."""

    def test_forward_pass(self):
        config = GeometryConfig(mlp_output=64)
        mlp = GeometryMLP(config)

        B = 4
        coords = torch.randn(B, 20, 2)
        scalars = torch.randn(B, 20)

        output = mlp(coords, scalars)

        assert output.shape == (B, 64)

    def test_input_dimension(self):
        config = GeometryConfig(num_scalars=20)
        mlp = GeometryMLP(config)

        # Input should be 20*2 + 20 = 60
        assert mlp.net[0].in_features == 60


class TestGeometryExtractor:
    """Test GeometryExtractor."""

    def test_compute_scalars_batch(self):
        config = GeometryConfig()
        extractor = GeometryExtractor(config)

        # Batch of landmarks
        landmarks = np.random.randn(5, 20, 2).astype(np.float32)
        scalars = extractor.compute_scalars_batch(landmarks)

        assert scalars.shape == (5, 20)
        assert np.all(np.isfinite(scalars))

    def test_compute_dataset_statistics(self):
        config = GeometryConfig()
        extractor = GeometryExtractor(config)

        # Multiple sequences
        sequences = [
            np.random.randn(10, 20, 2).astype(np.float32),
            np.random.randn(15, 20, 2).astype(np.float32),
            np.random.randn(8, 20, 2).astype(np.float32),
        ]

        mean, std = extractor.compute_dataset_statistics(sequences)

        assert mean.shape == (20,)
        assert std.shape == (20,)
        assert np.all(std > 0)

    def test_zscore_scalars(self):
        config = GeometryConfig()
        extractor = GeometryExtractor(config)

        # Set fake statistics
        extractor.scalars_mean = np.zeros(20)
        extractor.scalars_std = np.ones(20) * 2.0

        scalars = np.ones(20) * 4.0  # Mean 4, std 0
        zscored = extractor.zscore_scalars(scalars)

        # Should be (4 - 0) / 2 = 2
        assert np.allclose(zscored, 2.0)


import torch  # Import at bottom