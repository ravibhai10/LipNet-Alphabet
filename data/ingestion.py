"""
Video ingestion pipeline for the Alphabet Visual Speech dataset.

Supports:
1. Local MP4 file ingestion
2. Webcam recording

Handles decoding, validation, and metadata extraction.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Generator, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import logging

from data.schema import (
    RecordingMetadata, FaceTrackingInfo, MouthInfo, QualityMetrics,
    Letter, SourceType, validate_metadata,
    DATA_SOURCE_REAL_CAMERA, DATA_SOURCE_SYNTHETIC_TEST,
)

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Information extracted from a video file."""
    fps: float
    frame_count: int
    width: int
    height: int
    duration_ms: float
    codec: str
    is_valid: bool
    error_message: Optional[str] = None


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return VideoInfo(
            fps=0, frame_count=0, width=0, height=0,
            duration_ms=0, codec="", is_valid=False,
            error_message=f"Cannot open video: {video_path}"
        )
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((codec_int >> 8 * i) & 0xFF) for i in range(4)])
        is_valid = True
        error_msg = None
        if fps <= 0:
            is_valid = False
            error_msg = f"Invalid FPS: {fps}"
        elif frame_count <= 0:
            is_valid = False
            error_msg = f"Invalid frame count: {frame_count}"
        elif width <= 0 or height <= 0:
            is_valid = False
            error_msg = f"Invalid resolution: {width}x{height}"
        duration_ms = (frame_count / fps * 1000) if fps > 0 else 0
        return VideoInfo(
            fps=fps, frame_count=frame_count, width=width, height=height,
            duration_ms=duration_ms, codec=codec, is_valid=is_valid,
            error_message=error_msg
        )
    finally:
        cap.release()


def validate_video(video_path):
    info = get_video_info(video_path)
    if not info.is_valid:
        return False, info.error_message
    cap = cv2.VideoCapture(str(video_path))
    try:
        ret_first, frame_first = cap.read()
        if not ret_first or frame_first is None:
            return False, "Cannot read first frame"
        if info.frame_count > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, info.frame_count - 1)
            ret_last, frame_last = cap.read()
            if not ret_last or frame_last is None:
                return False, "Cannot read last frame"
        return True, None
    finally:
        cap.release()


@dataclass
class IngestionConfig:
    """Configuration for video ingestion."""
    dataset_root: Path
    fps: float = 30.0
    max_sequence_length: int = 76
    min_sequence_length: int = 15
    input_resolution: Tuple[int, int] = (1280, 720)
    codec: str = "mp4v"
    blur_threshold: float = 100.0
    brightness_range: Tuple[float, float] = (20.0, 235.0)
    min_mouth_size: int = 400
    max_pose_angle: float = 30.0
    duplicate_ssim_threshold: float = 0.98
    save_frames: bool = False
    save_landmarks_npz: bool = True
    save_crops: bool = False
    preprocessing_version: str = "1.0"
    config_version: str = "1.0"
    config_hash: str = ""


class VideoIngestor:
    """Ingest videos and create recording metadata."""

    def __init__(self, config):
        self.config = config
        self.recordings_root = config.dataset_root / "recordings"

    def ingest_local_mp4(
        self,
        video_path,
        letter,
        speaker_id,
        session_id,
        recording_id=None,
        lighting_tag=None,
        accent_tag=None,
        notes=None,
    ):
        """Ingest a local MP4 file.

        NOTE: data_source is set to SYNTHETIC_TEST because we cannot verify
        the provenance of arbitrary local MP4 files. Real webcam recordings
        should use record_webcam() which sets data_source=REAL_CAMERA.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        is_valid, error_msg = validate_video(video_path)
        if not is_valid:
            raise ValueError(f"Invalid video: {error_msg}")
        video_info = get_video_info(video_path)
        if recording_id is None:
            recording_id = self._generate_recording_id(letter, speaker_id, session_id)
        letter_dir = self.recordings_root / letter.upper() / speaker_id / session_id
        recording_dir = letter_dir / recording_id
        recording_dir.mkdir(parents=True, exist_ok=True)
        dest_video = recording_dir / "video.mp4"
        self._copy_video(video_path, dest_video)
        metadata = self._create_metadata(
            letter=letter,
            speaker_id=speaker_id,
            session_id=session_id,
            recording_id=recording_id,
            source=SourceType.LOCAL_MP4,
            video_info=video_info,
            lighting_tag=lighting_tag,
            accent_tag=accent_tag,
            notes=notes,
            data_source=DATA_SOURCE_SYNTHETIC_TEST,
        )
        metadata.save(recording_dir / "metadata.json")
        logger.info(f"Ingested recording: {recording_id} ({letter})")
        return metadata

    def record_webcam(
        self,
        letter,
        speaker_id,
        session_id,
        recording_id=None,
        duration_seconds=3.0,
        lighting_tag=None,
        accent_tag=None,
        notes=None,
        device_index=0,
        show_preview=True,
    ):
        """Record from webcam and save. Sets data_source=REAL_CAMERA."""
        import time
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam device {device_index}.")
        try:
            cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.input_resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.input_resolution[1])
            if recording_id is None:
                recording_id = self._generate_recording_id(letter, speaker_id, session_id)
            letter_dir = self.recordings_root / letter.upper() / speaker_id / session_id
            recording_dir = letter_dir / recording_id
            recording_dir.mkdir(parents=True, exist_ok=True)
            frames = []
            start_time = time.time()
            actual_duration = 0.0
            while actual_duration < duration_seconds:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                actual_duration = time.time() - start_time
                if show_preview:
                    display = frame.copy()
                    cv2.putText(display,
                        f"Recording: {letter} ({actual_duration:.1f}/{duration_seconds}s)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Recording", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            if show_preview:
                cv2.destroyWindow("Recording")
            if not frames:
                raise RuntimeError("No frames captured from webcam")
            dest_video = recording_dir / "video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
            out = cv2.VideoWriter(str(dest_video), fourcc, self.config.fps,
                (frames[0].shape[1], frames[0].shape[0]))
            for frame in frames:
                out.write(frame)
            out.release()
            video_info = get_video_info(dest_video)
            metadata = self._create_metadata(
                letter=letter, speaker_id=speaker_id, session_id=session_id,
                recording_id=recording_id, source=SourceType.WEBCAM,
                video_info=video_info, lighting_tag=lighting_tag,
                accent_tag=accent_tag, notes=notes,
                data_source=DATA_SOURCE_REAL_CAMERA,
            )
            metadata.save(recording_dir / "metadata.json")
            logger.info(f"Recorded {len(frames)} frames in {actual_duration:.0f}ms")
            return metadata
        finally:
            cap.release()

    def _generate_recording_id(self, letter, speaker_id, session_id):
        letter_dir = self.recordings_root / letter / speaker_id / session_id
        existing = 0
        if letter_dir.exists():
            for d in letter_dir.iterdir():
                if d.is_dir() and d.name.startswith("recording_"):
                    try:
                        num = int(d.name.split("_")[1])
                        existing = max(existing, num)
                    except (IndexError, ValueError):
                        pass
        return f"recording_{existing + 1:03d}"

    def _copy_video(self, src, dst):
        import shutil
        shutil.copy2(src, dst)

    def _create_metadata(
        self,
        letter, speaker_id, session_id, recording_id, source, video_info,
        lighting_tag=None, accent_tag=None, notes=None, data_source=None,
    ):
        """Create metadata for a recording. data_source is REQUIRED."""
        if data_source is None:
            raise ValueError(
                "data_source is REQUIRED. Pass 'real_camera' for real "
                "recordings or 'synthetic_test' for test fixtures."
            )
        now = datetime.utcnow().isoformat() + "Z"
        face_tracking = FaceTrackingInfo(
            frames_with_face=0,
            frames_without_face=video_info.frame_count,
            max_consecutive_missing=video_info.frame_count,
            mean_confidence=0.0, min_confidence=1.0,
        )
        mouth = MouthInfo(
            frames_with_mouth=0, missing_frames=video_info.frame_count,
            mean_mouth_width=0.0, mean_mouth_height=0.0,
            min_mouth_area=0.0, max_mouth_area=0.0,
        )
        quality = QualityMetrics(
            mean_brightness=0.0, std_brightness=0.0, mean_blur=0.0,
            mean_mouth_size=0.0, mean_pose_yaw=0.0, mean_pose_pitch=0.0,
            mean_pose_roll=0.0, duplicate_frame_count=0, quality_flags=[],
        )
        return RecordingMetadata(
            letter=letter, speaker_id=speaker_id, session_id=session_id,
            recording_id=recording_id, source=source, data_source=data_source,
            created_at=now, fps=video_info.fps,
            frame_count=video_info.frame_count,
            width=video_info.width, height=video_info.height,
            duration_ms=video_info.duration_ms,
            face_tracking=face_tracking, mouth=mouth, quality=quality,
            preprocessing_version=self.config.preprocessing_version,
            config_version=self.config.config_version,
            config_hash=self.config.config_hash,
            schema_version="1.0",
            lighting_tag=lighting_tag, accent_tag=accent_tag, notes=notes,
        )
