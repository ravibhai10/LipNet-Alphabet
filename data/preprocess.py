"""
Dataset preprocessing pipeline.

Orchestrates the full preprocessing workflow:
1. Video ingestion
2. Face tracking & mouth alignment
3. Appearance stream extraction
4. Geometry stream extraction
5. Motion stream extraction
6. Gated fusion (optional, for verification)
7. Manifest creation
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging
from datetime import datetime

from data.schema import RecordingMetadata, validate_metadata
from data.ingestion import VideoIngestor, IngestionConfig, get_video_info
from tracking.aligner import (
    TrackingPipeline,
    TrackingConfig,
    update_frame_timestamps,
    RealFaceLandmarksUnavailableError,
)
from data.schema import LANDMARK_SOURCE_MEDIAPIPE
from repr.appearance import AppearanceExtractor, AppearanceConfig, compute_appearance_from_crops
from repr.geometry import GeometryExtractor, GeometryConfig, compute_geometry_from_landmarks
from repr.motion import MotionExtractor, MotionConfig, compute_motion_from_data
from data.manifest import ManifestManager, ManifestConfig, create_splits, save_split_info

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for full preprocessing pipeline.

    The canonical defaults live here (single source of truth) and mirror
    configs/alphabet_dataset.yaml exactly. Production runs must construct
    this via PreprocessingPipeline.from_phase2_config(loaded_yaml_config)
    so the YAML controls runtime behavior.
    """
    dataset_root: Path = Path("./alphabet_dataset")
    # Ingestion
    fps: int = 30
    max_sequence_length: int = 300
    min_sequence_length: int = 10
    input_resolution: tuple = (1280, 720)
    # Tracking
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    max_num_faces: int = 1
    refine_landmarks: bool = True
    crop_margin_multiplier: float = 2.2
    appearance_crop_size: tuple = (64, 64)
    context_crop_size: tuple = (112, 112)
    # Quality
    blur_threshold: float = 100.0
    brightness_range: tuple = (20, 235)
    min_mouth_size: int = 400
    max_pose_angle: float = 30.0
    # Preprocessing
    save_frames: bool = False
    save_landmarks_npz: bool = True
    save_crops: bool = True
    preprocessing_version: str = "1.0"
    config_version: str = "1.0"
    config_hash: str = ""
    # Splits
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    min_speakers_per_split: int = 1
    # Resume
    resume: bool = True
    # Manifest file name (relative to dataset_root; from YAML manifest.path)
    manifest_filename: str = "manifest.jsonl"
    # Representation stream dimensions (from YAML geometry/motion sections)
    appearance_output_dim: int = 96
    geometry_num_scalars: int = 20
    geometry_mlp_hidden: int = 96
    geometry_mlp_output: int = 64
    geometry_normalization_scale: str = "inner_mouth_width"
    motion_savgol_window: int = 5
    motion_savgol_polyorder: int = 2
    motion_optical_flow_enabled: bool = True
    motion_mlp_output: int = 64
    # Mouth landmark subset (from YAML mouth_alignment section)
    mouth_landmark_indices: tuple = tuple()


class PreprocessingPipeline:
    """
    Full preprocessing pipeline for alphabet dataset.

    Handles streaming processing to avoid loading entire videos into RAM.
    Supports resumable processing.
    """

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.dataset_root = Path(config.dataset_root)

        # Initialize sub-configs
        self.ingestion_config = IngestionConfig(
            dataset_root=config.dataset_root,
            fps=config.fps,
            max_sequence_length=config.max_sequence_length,
            min_sequence_length=config.min_sequence_length,
            input_resolution=config.input_resolution,
            codec="mp4v",
            blur_threshold=config.blur_threshold,
            brightness_range=config.brightness_range,
            min_mouth_size=config.min_mouth_size,
            max_pose_angle=config.max_pose_angle,
            duplicate_ssim_threshold=0.98,
            save_frames=config.save_frames,
            save_landmarks_npz=config.save_landmarks_npz,
            save_crops=config.save_crops,
            preprocessing_version=config.preprocessing_version,
            config_version=config.config_version,
            config_hash=config.config_hash,
        )

        self.tracking_config = TrackingConfig(
            detection_confidence=config.detection_confidence,
            tracking_confidence=config.tracking_confidence,
            max_num_faces=config.max_num_faces,
            refine_landmarks=config.refine_landmarks,
            min_frames_with_face=10,
            max_missing_face_gap=5,
            normalization_scale=config.geometry_normalization_scale,
            crop_margin_multiplier=config.crop_margin_multiplier,
            appearance_crop_size=config.appearance_crop_size,
            context_crop_size=config.context_crop_size,
            num_mouth_landmarks=len(config.mouth_landmark_indices),
            mouth_landmark_indices=list(config.mouth_landmark_indices),
            blur_threshold=config.blur_threshold,
            brightness_range=config.brightness_range,
            min_mouth_size=config.min_mouth_size,
            max_pose_angle=config.max_pose_angle,
            save_frames=config.save_frames,
            save_landmarks_npz=config.save_landmarks_npz,
            save_crops=config.save_crops,
            preprocessing_version=config.preprocessing_version,
        )

        self.appearance_config = AppearanceConfig(
            crop_size=config.appearance_crop_size,
            output_dim=config.appearance_output_dim,
        )

        self.geometry_config = GeometryConfig(
            num_scalars=config.geometry_num_scalars,
            mlp_hidden=config.geometry_mlp_hidden,
            mlp_output=config.geometry_mlp_output,
            normalization_scale=config.geometry_normalization_scale,
        )

        self.motion_config = MotionConfig(
            savgol_window=config.motion_savgol_window,
            savgol_polyorder=config.motion_savgol_polyorder,
            optical_flow_enabled=config.motion_optical_flow_enabled,
            mlp_output=config.motion_mlp_output,
        )

        # Initialize components
        self.ingestor = VideoIngestor(self.ingestion_config)
        self.tracking_pipeline = TrackingPipeline(self.tracking_config)
        self.appearance_extractor = AppearanceExtractor(self.appearance_config)
        self.geometry_extractor = GeometryExtractor(self.geometry_config)
        self.motion_extractor = MotionExtractor(self.motion_config)

        # Manifest
        self.manifest_manager = ManifestManager(ManifestConfig(
            dataset_root=config.dataset_root,
            manifest_path=config.dataset_root / config.manifest_filename,
        ))

    @classmethod
    def from_phase2_config(cls, cfg) -> "PreprocessingPipeline":
        """Build a pipeline from a validated ``Phase2Config`` (config/loader.py).

        This is the canonical entry point: every runtime parameter comes from
        the YAML-derived typed configuration — no hard-coded defaults here.
        """
        return cls(PreprocessingConfig(
            dataset_root=cfg.dataset_root,
            fps=cfg.ingestion.fps,
            max_sequence_length=cfg.ingestion.max_sequence_length,
            min_sequence_length=cfg.ingestion.min_sequence_length,
            input_resolution=tuple(cfg.ingestion.input_resolution),
            detection_confidence=cfg.tracking.detection_confidence,
            tracking_confidence=cfg.tracking.tracking_confidence,
            max_num_faces=cfg.tracking.max_num_faces,
            refine_landmarks=cfg.tracking.refine_landmarks,
            crop_margin_multiplier=cfg.tracking.crop_margin_multiplier,
            appearance_crop_size=tuple(cfg.tracking.appearance_crop_size),
            context_crop_size=tuple(cfg.tracking.context_crop_size),
            blur_threshold=cfg.quality.blur_threshold,
            brightness_range=tuple(int(v) for v in cfg.quality.brightness_range),
            min_mouth_size=cfg.quality.min_mouth_size,
            max_pose_angle=cfg.quality.max_pose_angle,
            save_frames=cfg.preprocessing.save_frames,
            save_landmarks_npz=cfg.preprocessing.save_landmarks_npz,
            save_crops=cfg.preprocessing.save_crops,
            preprocessing_version=cfg.tracking.preprocessing_version,
            config_version=cfg.config_version,
            config_hash=cfg.config_hash,
            train_ratio=cfg.splits.train_ratio,
            val_ratio=cfg.splits.val_ratio,
            test_ratio=cfg.splits.test_ratio,
            random_seed=cfg.splits.random_seed,
            min_speakers_per_split=cfg.splits.min_speakers_per_split,
            resume=cfg.preprocessing.resume,
            manifest_filename=str(cfg.manifest.manifest_path.name)
            if cfg.manifest.manifest_path.is_absolute()
            else str(cfg.manifest.manifest_path),
            appearance_output_dim=cfg.appearance.output_dim,
            geometry_num_scalars=cfg.geometry.num_scalars,
            geometry_mlp_hidden=cfg.geometry.mlp_hidden,
            geometry_mlp_output=cfg.geometry.mlp_output,
            geometry_normalization_scale=cfg.geometry.normalization_scale,
            motion_savgol_window=cfg.motion.savgol_window,
            motion_savgol_polyorder=cfg.motion.savgol_polyorder,
            motion_optical_flow_enabled=cfg.motion.optical_flow_enabled,
            motion_mlp_output=cfg.motion.mlp_output,
            mouth_landmark_indices=tuple(cfg.tracking.mouth_landmark_indices),
        ))

    def process_recording(
        self,
        recording_dir: Path,
        metadata: RecordingMetadata,
        force: bool = False
    ) -> RecordingMetadata:
        """
        Process a single recording through the full pipeline.

        Args:
            recording_dir: Path to recording directory
            metadata: Recording metadata
            force: Force reprocessing even if derived data exists

        Returns:
            Updated metadata
        """
        video_path = recording_dir / "video.mp4"
        landmarks_dir = recording_dir / "landmarks"
        crops_dir = recording_dir / "crops"
        appearance_dir = crops_dir / "appearance"
        context_dir = crops_dir / "context"

        # Stamp provenance so every metadata.json records exactly which
        # configuration produced it.
        if getattr(self.config, "config_version", None):
            metadata.config_version = self.config.config_version
        if getattr(self.config, "config_hash", ""):
            metadata.config_hash = self.config.config_hash

        # HARD SAFETY CHECK: the real dataset must never contain simulated
        # landmarks. Anything that is not genuine MediaPipe output stops
        # preprocessing before any artifact is written.
        if metadata.landmark_source != LANDMARK_SOURCE_MEDIAPIPE:
            raise RealFaceLandmarksUnavailableError(
                f"Recording {recording_dir} declares landmark_source="
                f"'{metadata.landmark_source}'. Only "
                f"'{LANDMARK_SOURCE_MEDIAPIPE}' landmarks may enter a "
                "production dataset."
            )

        # Check if already processed
        if not force and self._is_processed(recording_dir):
            logger.info(f"Skipping already processed: {recording_dir}")
            # Persist provenance updates even for skipped recordings so that
            # re-runs with a new configuration are traceable.
            metadata.save(recording_dir / "metadata.json")
            return metadata

        # Step 1: Face tracking & mouth alignment
        logger.info(f"Tracking: {recording_dir.name}")
        metadata = self.tracking_pipeline.process_recording(
            video_path, recording_dir, metadata
        )

        # HARD SAFETY CHECK: a recording where the real landmarker found no
        # face at all must fail loudly. It must never contribute silent
        # garbage (empty landmarks / zero embeddings / manifest entry) to the
        # production dataset.
        if metadata.face_tracking.frames_with_face <= 0:
            self._remove_derived_artifacts(recording_dir)
            raise RealFaceLandmarksUnavailableError(
                f"Recording {recording_dir}: MediaPipe FaceLandmarker detected "
                f"no face in any of {metadata.frame_count} frames. The recording "
                "is unusable for lip-reading data and was NOT added to the "
                "dataset (all derived artifacts removed)."
            )

        # Update frame timestamps with actual FPS
        update_frame_timestamps(recording_dir, metadata.fps)

        # Step 2: Extract appearance embeddings (if crops saved)
        if self.config.save_crops and appearance_dir.exists():
            logger.info(f"Extracting appearance: {recording_dir.name}")
            appearance_embeddings = compute_appearance_from_crops(
                appearance_dir,
                recording_dir / "appearance_embeddings.npz",
                self.appearance_config
            )

        # Step 3: Extract geometry embeddings from landmarks
        if self.config.save_landmarks_npz and (landmarks_dir / "mouth_landmarks.npz").exists():
            logger.info(f"Extracting geometry: {recording_dir.name}")
            geometry_embeddings, scalars = compute_geometry_from_landmarks(
                landmarks_dir / "mouth_landmarks.npz",
                recording_dir / "geometry_embeddings.npz",
                self.geometry_config
            )

        # Step 4: Extract motion embeddings
        if self.config.save_landmarks_npz:
            landmarks_path = landmarks_dir / "mouth_landmarks.npz"
            scalars_path = recording_dir / "geometry_scalars.npz"
            if landmarks_path.exists() and scalars_path.exists():
                logger.info(f"Extracting motion: {recording_dir.name}")
                # Load appearance frames for optical flow if available
                appearance_frames = None
                if appearance_dir.exists():
                    frame_files = sorted(appearance_dir.glob("frame_*.png"))
                    appearance_frames = [cv2.imread(str(f)) for f in frame_files]
                    appearance_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in appearance_frames if f is not None]

                motion_embeddings = compute_motion_from_data(
                    landmarks_path,
                    scalars_path,
                    recording_dir / "motion_embeddings.npz",
                    self.motion_config,
                    appearance_frames=appearance_frames
                )

        # Save updated metadata
        metadata.save(recording_dir / "metadata.json")

        # Validate
        errors = validate_metadata(metadata)
        if errors:
            logger.warning(f"Validation issues for {recording_dir.name}: {errors}")

        return metadata

    def _remove_derived_artifacts(self, recording_dir: Path) -> None:
        """Delete all derived artifacts for a failed recording.

        Ensures a rejected recording leaves NO landmarks, embeddings,
        frame data, crops, or manifest entry behind.
        """
        import shutil
        for sub in ("landmarks", "crops", "frames"):
            p = recording_dir / sub
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
        for f in (
            "geometry_embeddings.npz", "geometry_scalars.npz",
            "motion_embeddings.npz", "appearance_embeddings.npz",
        ):
            p = recording_dir / f
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

    def _is_processed(self, recording_dir: Path) -> bool:
        """Check if recording has been fully processed."""
        landmarks_dir = recording_dir / "landmarks"
        required_files = [
            landmarks_dir / "mouth_landmarks.npz",
            landmarks_dir / "mouth_bboxes.npz",
            landmarks_dir / "head_poses.npz",
            recording_dir / "frames" / "frame_data.json",
        ]

        for f in required_files:
            if not f.exists():
                return False

        return True

    def process_letter(self, letter: str, force: bool = False) -> List[RecordingMetadata]:
        """
        Process all recordings for a letter.

        Args:
            letter: Letter to process (A-Z)
            force: Force reprocessing

        Returns:
            List of processed metadata
        """
        letter_dir = self.dataset_root / "recordings" / letter.upper()
        if not letter_dir.exists():
            logger.warning(f"Letter directory not found: {letter_dir}")
            return []

        results = []
        for speaker_dir in sorted(letter_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            for session_dir in sorted(speaker_dir.iterdir()):
                if not session_dir.is_dir():
                    continue
                for recording_dir in sorted(session_dir.iterdir()):
                    if not recording_dir.is_dir():
                        continue

                    metadata_path = recording_dir / "metadata.json"
                    if not metadata_path.exists():
                        continue

                    metadata = RecordingMetadata.load(metadata_path)
                    metadata = self.process_recording(recording_dir, metadata, force)
                    results.append(metadata)

        return results

    def process_all(self, force: bool = False) -> List[RecordingMetadata]:
        """Process all recordings in dataset."""
        results = []
        for letter in sorted(self._get_letters()):
            results.extend(self.process_letter(letter, force))
        return results

    def _get_letters(self) -> List[str]:
        """Get list of letters with recordings."""
        recordings_root = self.dataset_root / "recordings"
        if not recordings_root.exists():
            return []
        return [d.name for d in recordings_root.iterdir()
                if d.is_dir() and len(d.name) == 1 and d.name.isalpha()]

    def build_manifest_and_splits(self) -> Dict[str, Any]:
        """Build manifest and create speaker-independent splits."""
        logger.info("Building manifest...")
        entries = self.manifest_manager.build_manifest()

        logger.info("Creating splits...")
        splits = create_splits(
            self.manifest_manager.manifest_path,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
            random_seed=self.config.random_seed,
            min_speakers_per_split=self.config.min_speakers_per_split,
        )

        # Rebuild manifest with splits
        entries = self.manifest_manager.build_manifest(splits)

        # Save split info
        save_split_info(
            splits,
            self.manifest_manager.manifest_path,
            self.dataset_root / "splits.json",
            self.config.random_seed
        )

        return {
            "entries": len(entries),
            "splits": splits,
        }


import cv2  # Import at bottom