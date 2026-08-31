"""
Motion stream representation extraction.

Per DESIGN.md:
- Smoothed first- and second-order temporal derivatives (Savitzky-Golay, window 5)
  of (i) normalized landmarks, (ii) 20 geometry scalars
  - ΔW, Δlip-opening, Δjaw-drop, Δlip-spread, Δlandmarks ⇒ velocities/accelerations
- Coarse optical-flow patch (16×16×2, Farnebäck) as optional additional channel
- Small temporal MLP/conv → m_t ∈ ℝ^64

This module computes motion features from consecutive frames.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path
from scipy.signal import savgol_filter
import logging

logger = logging.getLogger(__name__)


@dataclass
class MotionConfig:
    """Configuration for motion stream extraction."""
    # Savitzky-Golay smoothing
    savgol_window: int = 5
    savgol_polyorder: int = 2

    # Optical flow (Farnebäck)
    optical_flow_enabled: bool = True
    flow_pyr_scale: float = 0.5
    flow_levels: int = 3
    flow_winsize: int = 15
    flow_iterations: int = 3
    flow_poly_n: int = 5
    flow_poly_sigma: float = 1.2
    flow_flags: int = 0

    # Motion MLP
    mlp_hidden: int = 96
    mlp_output: int = 64

    # Flow patch size
    flow_patch_size: Tuple[int, int] = (16, 16)

    # Landmark/geometry dimensions
    num_landmarks: int = 20
    num_scalars: int = 20


class MotionFeatureExtractor:
    """
    Compute motion features from sequences of landmarks and geometry scalars.

    Features computed:
    - Landmark velocities: [T, 20, 2] (first derivative)
    - Landmark accelerations: [T, 20, 2] (second derivative)
    - Scalar velocities: [T, 20] (first derivative of 20 scalars)
    - Scalar accelerations: [T, 20] (second derivative of 20 scalars)
    - Optical flow patches: [T, 16, 16, 2] (optional)
    """

    def __init__(self, config: MotionConfig):
        self.config = config

    def compute_derivatives(
        self,
        sequence: np.ndarray,
        axis: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute smoothed first and second derivatives using Savitzky-Golay.

        Args:
            sequence: [..., T, D] array where T is time dimension
            axis: Time axis (default 0)

        Returns:
            (velocity, acceleration) both same shape as input
        """
        # Move time axis to last for savgol_filter
        if axis != -1:
            sequence = np.moveaxis(sequence, axis, -1)

        orig_shape = sequence.shape
        T = orig_shape[-1]

        # Reshape to 2D: [..., D] x T
        seq_2d = sequence.reshape(-1, T)

        velocity = np.zeros_like(seq_2d)
        acceleration = np.zeros_like(seq_2d)

        window = min(self.config.savgol_window, T)
        if window % 2 == 0:
            window += 1
        window = max(3, window)

        for i in range(seq_2d.shape[0]):
            try:
                # First derivative
                velocity[i] = savgol_filter(
                    seq_2d[i], window, self.config.savgol_polyorder, deriv=1
                )
                # Second derivative
                acceleration[i] = savgol_filter(
                    seq_2d[i], window, self.config.savgol_polyorder, deriv=2
                )
            except ValueError:
                # Fallback: simple finite differences
                velocity[i, 1:] = np.diff(seq_2d[i])
                velocity[i, 0] = velocity[i, 1]
                acceleration[i, 1:] = np.diff(velocity[i])
                acceleration[i, 0] = acceleration[i, 1]

        velocity = velocity.reshape(orig_shape)
        acceleration = acceleration.reshape(orig_shape)

        # Move axis back
        if axis != -1:
            velocity = np.moveaxis(velocity, -1, axis)
            acceleration = np.moveaxis(acceleration, -1, axis)

        return velocity, acceleration

    def compute_landmark_motion(
        self,
        landmarks: np.ndarray  # [T, 20, 2]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute landmark velocities and accelerations.

        Args:
            landmarks: [T, 20, 2] normalized mouth landmarks

        Returns:
            (velocities [T, 20, 2], accelerations [T, 20, 2])
        """
        # Compute along time axis (0)
        velocities, accelerations = self.compute_derivatives(landmarks, axis=0)
        return velocities, accelerations

    def compute_scalar_motion(
        self,
        scalars: np.ndarray  # [T, 20]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute scalar velocities and accelerations.

        Args:
            scalars: [T, 20] geometry scalars

        Returns:
            (velocities [T, 20], accelerations [T, 20])
        """
        velocities, accelerations = self.compute_derivatives(scalars, axis=0)
        return velocities, accelerations

    def compute_optical_flow(
        self,
        frames: List[np.ndarray]  # List of [H, W] grayscale frames
    ) -> np.ndarray:
        """
        Compute Farnebäck optical flow between consecutive frames.

        Args:
            frames: List of grayscale frames

        Returns:
            [T-1, 16, 16, 2] flow patches centered on mouth
        """
        if not self.config.optical_flow_enabled or len(frames) < 2:
            return np.zeros((max(0, len(frames)-1), *self.config.flow_patch_size, 2), dtype=np.float32)

        flows = []

        for i in range(len(frames) - 1):
            prev = frames[i].astype(np.float32)
            curr = frames[i + 1].astype(np.float32)

            # Compute dense flow
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None,
                pyr_scale=self.config.flow_pyr_scale,
                levels=self.config.flow_levels,
                winsize=self.config.flow_winsize,
                iterations=self.config.flow_iterations,
                poly_n=self.config.flow_poly_n,
                poly_sigma=self.config.flow_poly_sigma,
                flags=self.config.flow_flags
            )

            # Extract 16x16 patch from center
            h, w = flow.shape[:2]
            ph, pw = self.config.flow_patch_size
            cy, cx = h // 2, w // 2
            y1 = max(0, cy - ph // 2)
            x1 = max(0, cx - pw // 2)
            y2 = min(h, y1 + ph)
            x2 = min(w, x1 + pw)

            patch = flow[y1:y2, x1:x2]

            # Pad if needed
            if patch.shape != (ph, pw, 2):
                padded = np.zeros((ph, pw, 2), dtype=np.float32)
                padded[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded

            flows.append(patch)

        return np.array(flows, dtype=np.float32)

    def compute_all_motion(
        self,
        landmarks: np.ndarray,      # [T, 20, 2]
        scalars: np.ndarray,        # [T, 20]
        appearance_frames: Optional[List[np.ndarray]] = None  # [T, H, W, 3] or [T, H, W]
    ) -> Dict[str, np.ndarray]:
        """
        Compute all motion features for a sequence.

        Args:
            landmarks: [T, 20, 2] mouth landmarks
            scalars: [T, 20] geometry scalars
            appearance_frames: Optional list of frames for optical flow

        Returns:
            Dictionary with all motion features
        """
        T = landmarks.shape[0]

        # Landmark motion
        lm_vel, lm_acc = self.compute_landmark_motion(landmarks)

        # Scalar motion
        sc_vel, sc_acc = self.compute_scalar_motion(scalars)

        result = {
            "landmark_velocity": lm_vel,      # [T, 20, 2]
            "landmark_acceleration": lm_acc,  # [T, 20, 2]
            "scalar_velocity": sc_vel,        # [T, 20]
            "scalar_acceleration": sc_acc,    # [T, 20]
        }

        # Optical flow
        if appearance_frames is not None:
            # Convert to grayscale if needed
            gray_frames = []
            for f in appearance_frames:
                if f.ndim == 3:
                    gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
                else:
                    gray = f
                gray_frames.append(gray)

            flow = self.compute_optical_flow(gray_frames)
            result["optical_flow"] = flow  # [T-1, 16, 16, 2]

        return result

    def compute_key_motion_scalars(
        self,
        scalars: np.ndarray,  # [T, 20]
        scalar_vel: np.ndarray  # [T, 20]
    ) -> Dict[str, np.ndarray]:
        """
        Extract key motion scalars from the geometry scalar velocities.

        Maps scalar indices to meaningful motion features:
        - 0: mouth_width -> ΔW
        - 1: mouth_height -> ΔH
        - 3: inner_opening -> Δlip-opening
        - 4: lip_spread -> Δlip-spread
        - 19: jaw_drop -> Δjaw-drop

        Args:
            scalars: [T, 20] geometry scalars
            scalar_vel: [T, 20] scalar velocities

        Returns:
            Dict with named motion features
        """
        return {
            "mouth_width_vel": scalar_vel[:, 0],      # ΔW
            "mouth_height_vel": scalar_vel[:, 1],     # ΔH
            "inner_opening_vel": scalar_vel[:, 3],    # Δlip-opening
            "lip_spread_vel": scalar_vel[:, 4],       # Δlip-spread
            "jaw_drop_vel": scalar_vel[:, 19],        # Δjaw-drop
            "mouth_width_acc": scalar_vel[:, 0],      # Δ²W (using vel as proxy, compute acc separately if needed)
        }


import cv2  # Import at bottom


class MotionMLP(nn.Module):
    """
    MLP for motion stream.

    Input features (concatenated per frame):
    - Landmark velocity: 20*2 = 40
    - Landmark acceleration: 20*2 = 40
    - Scalar velocity: 20
    - Scalar acceleration: 20
    - Optical flow patch: 16*16*2 = 512 (optional)
    Total: 120 (without flow) or 632 (with flow)

    Output: 64
    """

    def __init__(self, config: MotionConfig):
        super().__init__()
        self.config = config

        input_dim = config.num_landmarks * 4 + config.num_scalars * 2  # 40+40+20+20 = 120
        if config.optical_flow_enabled:
            input_dim += config.flow_patch_size[0] * config.flow_patch_size[1] * 2  # +512 = 632

        self.net = nn.Sequential(
            nn.Linear(input_dim, config.mlp_hidden),
            nn.LayerNorm(config.mlp_hidden),
            nn.GELU(),
            nn.Linear(config.mlp_hidden, config.mlp_output),
            nn.LayerNorm(config.mlp_output),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: [B, input_dim] concatenated motion features

        Returns:
            [B, 64] motion embeddings
        """
        return self.net(features)


class MotionExtractor:
    """
    Extract motion stream features from sequences.

    Handles:
    - Computing temporal derivatives
    - Optical flow
    - MLP embedding
    - Batch processing
    """

    def __init__(self, config: MotionConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.feature_extractor = MotionFeatureExtractor(config)
        self.mlp = MotionMLP(config).to(self.device)
        self.mlp.eval()

    def extract_batch(
        self,
        landmarks: np.ndarray,      # [B, 20, 2]
        scalars: np.ndarray,        # [B, 20]
        lm_vel: np.ndarray,         # [B, 20, 2]
        lm_acc: np.ndarray,         # [B, 20, 2]
        sc_vel: np.ndarray,         # [B, 20]
        sc_acc: np.ndarray,         # [B, 20]
        flow: Optional[np.ndarray] = None  # [B, 16, 16, 2]
    ) -> torch.Tensor:
        """
        Extract motion embeddings for a batch.

        Args:
            All inputs: [B, ...] batched features

        Returns:
            [B, 64] embeddings
        """
        B = landmarks.shape[0]

        # Flatten features
        feat_parts = [
            lm_vel.reshape(B, -1),     # [B, 40]
            lm_acc.reshape(B, -1),     # [B, 40]
            sc_vel.reshape(B, -1),     # [B, 20]
            sc_acc.reshape(B, -1),     # [B, 20]
        ]

        if flow is not None and self.config.optical_flow_enabled:
            feat_parts.append(flow.reshape(B, -1))  # [B, 512]

        features = np.concatenate(feat_parts, axis=1)  # [B, input_dim]
        features_tensor = torch.from_numpy(features).float().to(self.device)

        with torch.no_grad():
            embeddings = self.mlp(features_tensor)

        return embeddings.cpu()

    def extract_sequence(
        self,
        landmarks: np.ndarray,      # [T, 20, 2]
        scalars: np.ndarray,        # [T, 20]
        appearance_frames: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """
        Extract motion embeddings for a full sequence.

        Args:
            landmarks: [T, 20, 2] mouth landmarks
            scalars: [T, 20] geometry scalars
            appearance_frames: Optional list of frames for optical flow

        Returns:
            [T, 64] embeddings
        """
        T = landmarks.shape[0]

        if T < 2:
            return np.zeros((T, self.config.mlp_output), dtype=np.float32)

        # Compute all motion features
        motion_features = self.feature_extractor.compute_all_motion(
            landmarks, scalars, appearance_frames
        )

        lm_vel = motion_features["landmark_velocity"]
        lm_acc = motion_features["landmark_acceleration"]
        sc_vel = motion_features["scalar_velocity"]
        sc_acc = motion_features["scalar_acceleration"]
        flow = motion_features.get("optical_flow", None)

        # Optical flow stream. When enabled, guarantee a [T, ph, pw, 2]
        # tensor so the MLP input width (632 with flow) always matches:
        #   row 0 = zeros (frame 0 has no previous frame),
        #   rows 1..T-1 = frame-to-frame Farneback flow (or zeros if no frames).
        ph, pw = self.config.flow_patch_size
        if self.config.optical_flow_enabled:
            if flow is None:
                flow = np.zeros((T, ph, pw, 2), dtype=np.float32)
            else:
                flow = np.concatenate(
                    [np.zeros((1, ph, pw, 2), dtype=flow.dtype), flow], axis=0
                )[:T]

        # Process in batches
        batch_size = 32
        all_embeddings = []

        for i in range(0, T, batch_size):
            end = min(i + batch_size, T)
            batch_emb = self.extract_batch(
                landmarks[i:end], scalars[i:end],
                lm_vel[i:end], lm_acc[i:end],
                sc_vel[i:end], sc_acc[i:end],
                flow[i:end] if flow is not None else None
            )
            all_embeddings.append(batch_emb.numpy())

        return np.vstack(all_embeddings)

    def save_embeddings(self, embeddings: np.ndarray, output_path: Path) -> None:
        """Save embeddings to .npz file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, embeddings=embeddings)

    def save_motion_features(
        self,
        features: Dict[str, np.ndarray],
        output_path: Path
    ) -> None:
        """Save all motion features to .npz file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **features)


def compute_motion_from_data(
    landmarks_path: Path,
    scalars_path: Path,
    output_path: Path,
    config: MotionConfig,
    device: str = "cpu",
    appearance_frames: Optional[List[np.ndarray]] = None
) -> np.ndarray:
    """
    Convenience function to extract motion from saved data.

    Args:
        landmarks_path: Path to mouth_landmarks.npz [T, 20, 2]
        scalars_path: Path to geometry_scalars.npz [T, 20]
        output_path: Path to save embeddings.npz
        config: Motion configuration
        device: Torch device
        appearance_frames: Optional appearance frames for optical flow

    Returns:
        [T, 64] embeddings array
    """
    # Load landmarks
    lm_data = np.load(landmarks_path)
    frame_keys = sorted([k for k in lm_data.keys() if k.startswith("frame_")])
    landmarks = np.stack([lm_data[k] for k in frame_keys])  # [T, 20, 2]

    # Load scalars
    sc_data = np.load(scalars_path)
    scalars = sc_data["scalars"]  # [T, 20]

    extractor = MotionExtractor(config, device)
    embeddings = extractor.extract_sequence(landmarks, scalars, appearance_frames)

    extractor.save_embeddings(embeddings, output_path)

    return embeddings