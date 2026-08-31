"""
Geometry stream representation extraction.

Per DESIGN.md:
- Per frame: 68 MediaPipe FaceMesh landmarks (inner+outer lips, jaw, perioral)
- Normalized to mouth-centered coordinate frame with scale = inner-mouth width
- Derive 20 canonical scalars: mouth width, height, W/H ratio, inner lip opening,
  lip corner displacement (spread), upper-lip height, lower-lip height,
  jaw-to-mouth-center distance, aperture-area proxy, and z-scored versions
- MLP(coords + scalars → 96 → 64) → g_t ∈ ℝ^64

This module extracts the geometry scalars and provides the MLP for embedding.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class GeometryConfig:
    """Configuration for geometry stream extraction."""
    num_scalars: int = 20
    mlp_hidden: int = 96
    mlp_output: int = 64
    normalization_scale: str = "inner_mouth_width"  # inner_mouth_width, inner_mouth_height, face_width


class GeometryScalars:
    """
    Compute 20 canonical geometry scalars from normalized mouth landmarks.

    Landmarks are expected to be [20, 2] in mouth-centered normalized coordinates
    where scale = inner mouth width (distance between lip corners).
    """

    # Landmark indices within the canonical 20-point mouth landmark set.
    # Order contract (see configs/alphabet_dataset.yaml mouth_alignment):
    #   0: 78   inner upper left      1: 74   inner upper mid-left
    #   2: 81   inner upper mid-right 3: 13   inner upper center
    #   4: 312  inner lower mid-right 5: 308  inner lower right
    #   6: 402  inner lower mid       7: 14   inner lower center
    #   8: 61   left corner (inner)   9: 291  right corner (inner)
    #   10-15   outer upper lip arc: 39, 37, 0, 267, 269, 391
    #   16: 61  left corner (outer)
    #   17: 0   upper lip center (outer)
    #   18: 17  lower lip center (outer)
    #   19: 291 right corner (outer)

    LEFT_CORNER_INNER = 8   # landmark 61
    RIGHT_CORNER_INNER = 9  # landmark 291
    LEFT_CORNER_OUTER = 16  # landmark 61 (outer reference)
    RIGHT_CORNER_OUTER = 19 # landmark 291

    # Upper lip points
    UPPER_LIP_TOP = 17      # landmark 0 (outer upper center)
    UPPER_LIP_MID = 3       # landmark 13 (inner upper center)

    # Lower lip points
    LOWER_LIP_MID = 18      # landmark 17 (outer lower center)
    LOWER_LIP_BOTTOM = 7    # landmark 14 (inner lower center)

    # Jaw points (using outer lip corners as proxy)
    JAW_LEFT = 16           # landmark 61
    JAW_RIGHT = 19          # landmark 291

    def __init__(self, config: GeometryConfig):
        self.config = config

    def compute_scalars(self, mouth_landmarks: np.ndarray) -> np.ndarray:
        """
        Compute 20 geometry scalars from normalized mouth landmarks.

        Args:
            mouth_landmarks: [20, 2] normalized coordinates (mouth-centered)

        Returns:
            [20] array of scalar values
        """
        if mouth_landmarks.shape != (20, 2):
            raise ValueError(f"Expected (20, 2) landmarks, got {mouth_landmarks.shape}")

        lm = mouth_landmarks

        # 1. Mouth width (outer corners)
        mouth_width = np.linalg.norm(lm[self.RIGHT_CORNER_OUTER] - lm[self.LEFT_CORNER_OUTER])

        # 2. Mouth height (upper to lower)
        upper_lip_y = lm[self.UPPER_LIP_MID, 1]
        lower_lip_y = lm[self.LOWER_LIP_MID, 1]
        mouth_height = abs(lower_lip_y - upper_lip_y)

        # 3. W/H ratio
        wh_ratio = mouth_width / (mouth_height + 1e-6)

        # 4. Inner lip opening (vertical distance between inner lips at center)
        inner_upper = lm[3, 1]  # landmark 16 (upper inner mid)
        inner_lower = lm[11, 1] # landmark 64 (lower inner mid)
        inner_opening = abs(inner_lower - inner_upper)

        # 5. Lip corner displacement (spread) - distance between inner corners
        lip_spread = np.linalg.norm(lm[self.RIGHT_CORNER_INNER] - lm[self.LEFT_CORNER_INNER])

        # 6. Upper lip height (distance from inner upper to outer upper)
        upper_lip_height = np.linalg.norm(lm[self.UPPER_LIP_MID] - lm[self.UPPER_LIP_TOP])

        # 7. Lower lip height (distance from inner lower to outer lower)
        lower_lip_height = np.linalg.norm(lm[self.LOWER_LIP_MID] - lm[self.LOWER_LIP_BOTTOM])

        # 8. Jaw-to-mouth-center distance
        mouth_center = (lm[self.LEFT_CORNER_INNER] + lm[self.RIGHT_CORNER_INNER]) / 2
        jaw_center = (lm[self.JAW_LEFT] + lm[self.JAW_RIGHT]) / 2
        jaw_distance = np.linalg.norm(jaw_center - mouth_center)

        # 9. Aperture area proxy (approximate ellipse area)
        aperture_area = np.pi * (mouth_width / 2) * (inner_opening / 2)

        # 10. Upper lip curvature (approximate)
        upper_lip_left = lm[0]
        upper_lip_right = lm[7]
        upper_lip_mid = lm[3]
        upper_curvature = self._compute_curvature(upper_lip_left, upper_lip_mid, upper_lip_right)

        # 11. Lower lip curvature
        lower_lip_left = lm[8]
        lower_lip_right = lm[15]
        lower_lip_mid = lm[11]
        lower_curvature = self._compute_curvature(lower_lip_left, lower_lip_mid, lower_lip_right)

        # 12. Lip corner angle (angle of line connecting corners)
        corner_vector = lm[self.RIGHT_CORNER_INNER] - lm[self.LEFT_CORNER_INNER]
        corner_angle = np.arctan2(corner_vector[1], corner_vector[0])

        # 13. Mouth symmetry (horizontal)
        left_width = np.linalg.norm(lm[self.UPPER_LIP_TOP] - lm[self.LEFT_CORNER_INNER])
        right_width = np.linalg.norm(lm[self.UPPER_LIP_TOP] - lm[self.RIGHT_CORNER_INNER])
        symmetry_h = abs(left_width - right_width) / (left_width + right_width + 1e-6)

        # 14. Mouth roundness (width/height relationship)
        roundness = 4 * aperture_area / (np.pi * mouth_width ** 2 + 1e-6)

        # 15. Upper lip protrusion (x-offset of upper lip mid from corners)
        upper_protrusion = lm[self.UPPER_LIP_MID, 0] - (lm[self.LEFT_CORNER_INNER, 0] + lm[self.RIGHT_CORNER_INNER, 0]) / 2

        # 16. Lower lip protrusion
        lower_protrusion = lm[self.LOWER_LIP_MID, 0] - (lm[self.LEFT_CORNER_INNER, 0] + lm[self.RIGHT_CORNER_INNER, 0]) / 2

        # 17. Inner mouth width (corner to corner)
        inner_mouth_width = lip_spread

        # 18. Inner mouth height
        inner_mouth_height = inner_opening

        # 19. Lip thickness (average of upper and lower)
        lip_thickness = (upper_lip_height + lower_lip_height) / 2

        # 20. Jaw drop (vertical distance from mouth center to jaw)
        jaw_drop = jaw_center[1] - mouth_center[1]

        scalars = np.array([
            mouth_width,           # 0
            mouth_height,          # 1
            wh_ratio,              # 2
            inner_opening,         # 3
            lip_spread,            # 4
            upper_lip_height,      # 5
            lower_lip_height,      # 6
            jaw_distance,          # 7
            aperture_area,         # 8
            upper_curvature,       # 9
            lower_curvature,       # 10
            corner_angle,          # 11
            symmetry_h,            # 12
            roundness,             # 13
            upper_protrusion,      # 14
            lower_protrusion,      # 15
            inner_mouth_width,     # 16
            inner_mouth_height,    # 17
            lip_thickness,         # 18
            jaw_drop,              # 19
        ], dtype=np.float32)

        return scalars

    def _compute_curvature(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Compute approximate curvature from three points."""
        # Using triangle area / base length as curvature proxy
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)

        # Heron's formula for triangle area
        s = (a + b + c) / 2
        area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))

        # Curvature = 4 * area / (a * b * c) * (a + b + c) / 2
        # Simplified: 2 * area / (a * b)
        curvature = 2 * area / (a * b + 1e-6)
        return float(curvature)

    def compute_zscored(self, scalars: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Compute z-scored version of scalars."""
        return (scalars - mean) / (std + 1e-6)


class GeometryMLP(nn.Module):
    """
    MLP for geometry stream: (coords + scalars) → 96 → 64

    Input: [B, 20*2 + 20] = [B, 60] (40 coords + 20 scalars)
    Output: [B, 64]
    """

    def __init__(self, config: GeometryConfig):
        super().__init__()
        self.config = config

        input_dim = config.num_scalars * 2 + config.num_scalars  # coords (x,y) + scalars

        self.net = nn.Sequential(
            nn.Linear(input_dim, config.mlp_hidden),
            nn.LayerNorm(config.mlp_hidden),
            nn.GELU(),
            nn.Linear(config.mlp_hidden, config.mlp_output),
            nn.LayerNorm(config.mlp_output),
        )

    def forward(self, coords: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            coords: [B, 20, 2] normalized mouth landmarks
            scalars: [B, 20] geometry scalars

        Returns:
            [B, 64] geometry embeddings
        """
        B = coords.shape[0]
        coords_flat = coords.view(B, -1)  # [B, 40]
        x = torch.cat([coords_flat, scalars], dim=-1)  # [B, 60]
        return self.net(x)


class GeometryExtractor:
    """
    Extract geometry stream features from mouth landmarks.

    Handles:
    - Computing 20 canonical scalars from landmarks
    - MLP embedding
    - Batch processing
    - Statistics accumulation for z-scoring
    """

    def __init__(self, config: GeometryConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.scalars_computer = GeometryScalars(config)
        self.mlp = GeometryMLP(config).to(self.device)
        self.mlp.eval()

        # Statistics for z-scoring (computed from training data)
        self.scalars_mean: Optional[np.ndarray] = None
        self.scalars_std: Optional[np.ndarray] = None

    def compute_scalars(self, mouth_landmarks: np.ndarray) -> np.ndarray:
        """
        Compute geometry scalars for a single frame.

        Args:
            mouth_landmarks: [20, 2] normalized landmarks

        Returns:
            [20] scalars array
        """
        return self.scalars_computer.compute_scalars(mouth_landmarks)

    def compute_scalars_batch(self, mouth_landmarks: np.ndarray) -> np.ndarray:
        """
        Compute geometry scalars for a batch.

        Args:
            mouth_landmarks: [B, 20, 2] normalized landmarks

        Returns:
            [B, 20] scalars array
        """
        B = mouth_landmarks.shape[0]
        scalars = np.zeros((B, self.config.num_scalars), dtype=np.float32)

        for i in range(B):
            scalars[i] = self.scalars_computer.compute_scalars(mouth_landmarks[i])

        return scalars

    def extract_batch(
        self,
        mouth_landmarks: np.ndarray,
        scalars: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """
        Extract geometry embeddings for a batch.

        Args:
            mouth_landmarks: [B, 20, 2] normalized landmarks
            scalars: Optional precomputed [B, 20] scalars

        Returns:
            [B, 64] embeddings
        """
        if scalars is None:
            scalars = self.compute_scalars_batch(mouth_landmarks)

        coords_tensor = torch.from_numpy(mouth_landmarks).float().to(self.device)
        scalars_tensor = torch.from_numpy(scalars).float().to(self.device)

        with torch.no_grad():
            embeddings = self.mlp(coords_tensor, scalars_tensor)

        return embeddings.cpu()

    def extract_sequence(
        self,
        mouth_landmarks: List[np.ndarray],
        scalars: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """
        Extract embeddings for a full sequence.

        Args:
            mouth_landmarks: List of [20, 2] landmarks (length T)
            scalars: Optional list of [20] precomputed scalars

        Returns:
            [T, 64] numpy array
        """
        if not mouth_landmarks:
            return np.zeros((0, self.config.mlp_output), dtype=np.float32)

        # Stack into batch
        lm_batch = np.stack(mouth_landmarks)

        if scalars is not None:
            scalars_batch = np.stack(scalars)
        else:
            scalars_batch = None

        # Process in batches
        batch_size = 64
        all_embeddings = []

        for i in range(0, len(lm_batch), batch_size):
            batch_lm = lm_batch[i:i+batch_size]
            batch_sc = scalars_batch[i:i+batch_size] if scalars_batch is not None else None
            embeddings = self.extract_batch(batch_lm, batch_sc)
            all_embeddings.append(embeddings.numpy())

        return np.vstack(all_embeddings)

    def compute_dataset_statistics(
        self,
        all_mouth_landmarks: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and std of scalars across dataset for z-scoring.

        Args:
            all_mouth_landmarks: List of [T, 20, 2] per-recording landmarks

        Returns:
            (mean, std) of shape [20]
        """
        all_scalars = []

        for lm_seq in all_mouth_landmarks:
            scalars_seq = self.compute_scalars_batch(lm_seq)
            all_scalars.append(scalars_seq)

        if not all_scalars:
            return np.zeros(self.config.num_scalars), np.ones(self.config.num_scalars)

        combined = np.vstack(all_scalars)
        mean = combined.mean(axis=0)
        std = combined.std(axis=0)
        std = np.maximum(std, 1e-6)

        self.scalars_mean = mean
        self.scalars_std = std

        return mean, std

    def zscore_scalars(self, scalars: np.ndarray) -> np.ndarray:
        """Apply z-scoring using computed statistics."""
        if self.scalars_mean is None or self.scalars_std is None:
            logger.warning("Statistics not computed, returning raw scalars")
            return scalars
        return self.scalars_computer.compute_zscored(scalars, self.scalars_mean, self.scalars_std)

    def save_embeddings(self, embeddings: np.ndarray, output_path: Path) -> None:
        """Save embeddings to .npz file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, embeddings=embeddings)

    def save_scalars(self, scalars: np.ndarray, output_path: Path) -> None:
        """Save scalars to .npz file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, scalars=scalars)


def compute_geometry_from_landmarks(
    landmarks_path: Path,
    output_path: Path,
    config: GeometryConfig,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to extract geometry from landmark .npz files.

    Args:
        landmarks_path: Path to mouth_landmarks.npz
        output_path: Path to save embeddings.npz
        config: Geometry configuration
        device: Torch device

    Returns:
        (embeddings [T, 64], scalars [T, 20])
    """
    data = np.load(landmarks_path)

    # Load landmarks in frame order
    frame_keys = sorted([k for k in data.keys() if k.startswith("frame_")])
    mouth_landmarks = [data[k] for k in frame_keys]

    extractor = GeometryExtractor(config, device)

    # Compute scalars
    scalars = extractor.compute_scalars_batch(np.stack(mouth_landmarks))

    # Extract embeddings
    embeddings = extractor.extract_sequence(mouth_landmarks, [scalars[i] for i in range(len(scalars))])

    # Save
    extractor.save_embeddings(embeddings, output_path)
    extractor.save_scalars(scalars, output_path.with_name("geometry_scalars.npz"))

    return embeddings, scalars