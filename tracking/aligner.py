"""
Face tracking and mouth alignment using MediaPipe Face Landmarker
(modern MediaPipe Tasks API).

Implements:
- Face detection and tracking with identity preservation
- Mouth landmark extraction
- Mouth-centered coordinate normalization
- Head pose estimation
- Motion energy VAD (Voice Activity Detection)

IMPORTANT: This module NEVER produces simulated/fake landmarks. If the real
MediaPipe FaceLandmarker cannot be initialized, every entry point raises
``RealFaceLandmarksUnavailableError`` (REAL_FACE_LANDMARKS_UNAVAILABLE).
Synthetic landmarks are permitted ONLY inside isolated unit tests, never in
the production preprocessing path.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass
import logging

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    FACE_LANDMARKER_AVAILABLE = True
except Exception as _mediapipe_err:  # noqa: BLE001 - MediaPipe optional at import time
    FACE_LANDMARKER_AVAILABLE = False
    mp = None
    mp_vision = None
    _MEDIAPIPE_IMPORT_ERROR = _mediapipe_err

from data.schema import (
    RecordingMetadata, FaceTrackingInfo, MouthInfo, QualityMetrics,
    FrameData, QualityFlag, validate_metadata,
    LANDMARK_SOURCE_MEDIAPIPE,
)

logger = logging.getLogger(__name__)

#: Default location of the MediaPipe FaceLandmarker model bundle.
DEFAULT_LANDMARKER_MODEL_PATH = Path("models/face_landmarker.task")


class RealFaceLandmarksUnavailableError(RuntimeError):
    """Raised when real MediaPipe face landmarks cannot be produced.

    Production preprocessing MUST stop when this is raised: no landmarks,
    geometry, motion, or manifest entries may be written for the recording.
    """

    CODE = "REAL_FACE_LANDMARKS_UNAVAILABLE"

    def __init__(self, detail: str):
        super().__init__(
            f"{self.CODE}: {detail}. Real face landmarks are mandatory for "
            "dataset generation; simulated landmarks are never emitted."
        )



# Canonical 20-point mouth landmark set (REAL MediaPipe FaceMesh indices).
# Order contract (consumed by repr/geometry.py GeometryScalars):
#   0-3   inner upper lip:      78, 74, 81, 13
#   4-7   inner lower lip:     312, 308, 402, 14
#   8     left corner (inner):  61
#   9     right corner (inner): 291
#   10-15 outer upper lip:      39, 37, 0, 267, 269, 391
#   16    left corner (outer):  61
#   17    upper lip center:      0
#   18    lower lip center:     17
#   19    right corner (outer): 291
MOUTH_LANDMARK_INDICES = [
    78, 74, 81, 13,      # inner upper lip
    312, 308, 402, 14,   # inner lower lip
    61, 291,             # left / right mouth corners (inner)
    39, 37, 0, 267, 269, 391,  # outer upper lip
    61,                  # left corner (outer)
    0,                   # upper lip center
    17,                  # lower lip center
    291,                 # right corner (outer)
]

# Full face bbox landmarks for pose estimation
FACE_OVAL_INDICES = list(range(10, 152))  # Approximate oval

# Key landmarks for head pose estimation
NOSE_TIP = 1
LEFT_EYE = 33
RIGHT_EYE = 263
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
CHIN = 152
FOREHEAD = 10


@dataclass
class TrackingConfig:
    """Configuration for face tracking and alignment.

    The canonical defaults live here (single source of truth) and mirror
    configs/alphabet_dataset.yaml exactly. Production runs should construct
    this from the loaded YAML config (Phase2Config.tracking).
    """
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_frames_with_face: int = 10
    max_missing_face_gap: int = 5
    normalization_scale: str = "inner_mouth_width"
    crop_margin_multiplier: float = 2.2
    appearance_crop_size: Tuple[int, int] = (64, 64)
    context_crop_size: Tuple[int, int] = (112, 112)
    num_mouth_landmarks: int = 20
    mouth_landmark_indices: List[int] = None
    blur_threshold: float = 100.0
    brightness_range: Tuple[int, int] = (20, 235)
    min_mouth_size: int = 400
    max_pose_angle: float = 30.0
    save_frames: bool = False
    save_landmarks_npz: bool = True
    save_crops: bool = True
    preprocessing_version: str = "1.0"

    def __post_init__(self):
        if self.mouth_landmark_indices is None:
            self.mouth_landmark_indices = MOUTH_LANDMARK_INDICES


@dataclass
class FrameTrackingResult:
    """Result of tracking a single frame."""
    frame_index: int
    timestamp_ms: float
    has_face: bool
    has_mouth: bool
    confidence: float
    landmarks: Optional[np.ndarray] = None  # [468, 3]
    mouth_landmarks: Optional[np.ndarray] = None  # [20, 2] normalized
    mouth_bbox: Optional[List[float]] = None  # [x, y, w, h] normalized
    head_pose: Optional[List[float]] = None  # [pitch, yaw, roll] degrees
    quality_flags: List[str] = None
    appearance_crop: Optional[np.ndarray] = None  # [64, 64, 3]
    context_crop: Optional[np.ndarray] = None  # [112, 112, 1]
    speaker_id: Optional[str] = None  # identity of tracked face

    def __post_init__(self):
        if self.quality_flags is None:
            self.quality_flags = []


class FaceTracker:
    """
    Face tracking with the MediaPipe Tasks FaceLandmarker (VIDEO mode).

    Maintains identity across frames, handles temporary misses,
    and records decisions in metadata.

    Raises ``RealFaceLandmarksUnavailableError`` if the real landmarker
    cannot be initialized — this class NEVER simulates landmarks.
    """

    def __init__(self, config: TrackingConfig, model_path: Optional[Path] = None):
        self.config = config
        self.landmarker = self._create_landmarker(model_path)
        self.last_face_landmarks = None
        self.missed_frames = 0
        self.track_confidence = 0.0
        # Monotonic timestamp (ms) required by the VIDEO running mode.
        self._last_timestamp_ms = -1

    def _create_landmarker(self, model_path: Optional[Path]):
        """Create the real FaceLandmarker. Fails loudly — never falls back."""
        if not FACE_LANDMARKER_AVAILABLE:
            raise RealFaceLandmarksUnavailableError(
                f"mediapipe.tasks is not importable ({type(_MEDIAPIPE_IMPORT_ERROR).__name__}: "
                f"{_MEDIAPIPE_IMPORT_ERROR}); install a MediaPipe version that provides "
                "mediapipe.tasks.python.vision.FaceLandmarker"
            )

        model_path = Path(model_path) if model_path else DEFAULT_LANDMARKER_MODEL_PATH
        if not model_path.is_file():
            raise RealFaceLandmarksUnavailableError(
                f"FaceLandmarker model not found at '{model_path}'. Download "
                "face_landmarker.task into models/ or pass model_path explicitly."
            )

        try:
            base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=max(1, int(self.config.max_num_faces)),
                min_face_detection_confidence=float(self.config.detection_confidence),
                min_face_presence_confidence=float(self.config.tracking_confidence),
                min_tracking_confidence=float(self.config.tracking_confidence),
                output_facial_transformation_matrixes=False,
                output_face_blendshapes=False,
            )
            return mp_vision.FaceLandmarker.create_from_options(options)
        except Exception as e:  # noqa: BLE001 - convert any init failure to explicit error
            raise RealFaceLandmarksUnavailableError(
                f"Failed to create MediaPipe FaceLandmarker ({type(e).__name__}: {e})"
            ) from e

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp_ms: float = 0.0,
        speaker_id: Optional[str] = None,
    ) -> FrameTrackingResult:
        """
        Process a single frame for face tracking with REAL MediaPipe landmarks.

        Args:
            frame: BGR image (H, W, 3)
            frame_index: index of the frame in the recording
            timestamp_ms: monotonic frame timestamp in milliseconds
            speaker_id: recording-level speaker identity to stamp on results

        Returns:
            FrameTrackingResult containing genuine MediaPipe landmarks.
        """
        height, width = frame.shape[:2]

        # The Tasks VIDEO mode requires strictly increasing timestamps.
        ts = float(timestamp_ms)
        if ts <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1.0
        self._last_timestamp_ms = ts

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame),
        )
        detection = self.landmarker.detect_for_video(mp_image, int(round(ts)))

        face_list = getattr(detection, "face_landmarks", None) or []
        if face_list:
            # Primary speaker selection: first detected track. With
            # num_faces == 1 MediaPipe itself performs temporal identity
            # tracking, so the same person keeps the same track.
            face_landmarks = face_list[0]
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in face_landmarks], dtype=np.float64
            )

            # Update tracking state
            self.last_face_landmarks = landmarks
            self.missed_frames = 0
            self.track_confidence = self.config.tracking_confidence

            return FrameTrackingResult(
                frame_index=frame_index,
                timestamp_ms=ts,
                has_face=True,
                has_mouth=True,
                confidence=self.track_confidence,
                landmarks=landmarks,
                speaker_id=speaker_id,
                quality_flags=[],
            )

        # No face detected by the real landmarker
        self.missed_frames += 1
        self.track_confidence = max(0.0, self.track_confidence - 0.1)

        # If recently had a face, use last known position (temporal fill),
        # explicitly flagged so downstream stages can treat it as inferred.
        if (
            self.last_face_landmarks is not None
            and self.missed_frames <= self.config.max_missing_face_gap
        ):
            return FrameTrackingResult(
                frame_index=frame_index,
                timestamp_ms=ts,
                has_face=True,
                has_mouth=True,
                confidence=self.track_confidence,
                landmarks=self.last_face_landmarks.copy(),
                speaker_id=speaker_id,
                quality_flags=[QualityFlag.NO_FACE.value],  # Mark as inferred
            )

        return FrameTrackingResult(
            frame_index=frame_index,
            timestamp_ms=ts,
            has_face=False,
            has_mouth=False,
            confidence=0.0,
            landmarks=None,
            speaker_id=speaker_id,
            quality_flags=[QualityFlag.NO_FACE.value],
        )

    def _simulate_tracking(self, frame: np.ndarray, frame_index: int, timestamp_ms: float) -> FrameTrackingResult:
        """DEPRECATED synthetic fallback — removed. Kept only as documentation
        that simulated landmarks must NEVER re-enter the production path."""
        raise RealFaceLandmarksUnavailableError(
            "Simulated face landmarks are forbidden in the production pipeline."
        )

    def reset(self):
        """Reset tracking state."""
        self.last_face_landmarks = None
        self.missed_frames = 0
        self.track_confidence = 0.0
        self._last_timestamp_ms = -1

    def close(self):
        """Release the underlying landmarker resources."""
        if getattr(self, "landmarker", None) is not None:
            try:
                self.landmarker.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass


class MouthAligner:
    """
    Mouth alignment and normalization.

    Extracts mouth landmarks, computes mouth-centered coordinate frame,
    normalizes by inner-mouth-width scale, and generates appearance/context crops.
    """

    def __init__(self, config: TrackingConfig):
        self.config = config

    def extract_mouth_landmarks(
        self,
        landmarks: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, List[float], List[str]]:
        """
        Extract and normalize mouth landmarks from face landmarks.

        Args:
            landmarks: [468, 3] face landmarks (normalized 0-1)
            frame_shape: (height, width) of original frame

        Returns:
            Tuple of:
                - normalized_mouth_landmarks: [20, 2] mouth-centered,
                  scale-normalized coordinates in units of inner-mouth
                  width (corner-to-corner distance = ~1.0 by construction)
                - mouth_bbox: [x, y, w, h] normalized in original frame
                - quality_flags: List of quality issue strings
        """
        height, width = frame_shape

        # Get mouth landmarks
        mouth_indices = self.config.mouth_landmark_indices[:self.config.num_mouth_landmarks]
        mouth_lm = landmarks[mouth_indices]  # [20, 3]

        # Compute mouth bounding box in normalized coords
        min_x = mouth_lm[:, 0].min()
        max_x = mouth_lm[:, 0].max()
        min_y = mouth_lm[:, 1].min()
        max_y = mouth_lm[:, 1].max()

        mouth_width_raw = max_x - min_x
        mouth_height_raw = max_y - min_y

        # Inner mouth width (corner to corner) for scale normalization.
        # The lip corners are MediaPipe landmarks 61 (left) and 291 (right).
        # This matches the GeometryExtractor convention where subset positions
        # 8/9 hold landmarks 61/291.
        if 61 in mouth_indices and 291 in mouth_indices:
            left_corner_idx = mouth_indices.index(61)
            right_corner_idx = mouth_indices.index(291)
        else:
            # Fallback: canonical corner slots within the configured subset.
            left_corner_idx = min(8, len(mouth_indices) - 1)
            right_corner_idx = min(9, len(mouth_indices) - 1)
        inner_width = abs(float(mouth_lm[right_corner_idx, 0] - mouth_lm[left_corner_idx, 0]))
        inner_height = abs(float(mouth_lm[right_corner_idx, 1] - mouth_lm[left_corner_idx, 1]))

        # Scale = inner mouth width (make distances comparable across scales)
        if self.config.normalization_scale == "inner_mouth_width":
            scale = inner_width if inner_width > 1e-6 else 1.0
        elif self.config.normalization_scale == "inner_mouth_height":
            scale = inner_height if inner_height > 1e-6 else 1.0
        else:
            scale = 1.0

        # Mouth center (normalized)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # Normalize mouth landmarks to mouth-centered frame
        # Coordinates: (lm - center) / scale
        # Shape: [20, 2]
        normalized = np.zeros((len(mouth_indices), 2))
        for i in range(len(mouth_indices)):
            normalized[i, 0] = (mouth_lm[i, 0] - center_x) / scale
            normalized[i, 1] = (mouth_lm[i, 1] - center_y) / scale

        # Bounding box (normalized)
        # Add margin
        margin_x = mouth_width_raw * (self.config.crop_margin_multiplier - 1) / 2
        margin_y = mouth_height_raw * (self.config.crop_margin_multiplier - 1) / 2

        bbox_x = max(0.0, min_x - margin_x)
        bbox_y = max(0.0, min_y - margin_y)
        bbox_w = min(1.0 - bbox_x, mouth_width_raw + 2 * margin_x)
        bbox_h = min(1.0 - bbox_y, mouth_height_raw + 2 * margin_y)

        mouth_bbox = [bbox_x, bbox_y, bbox_w, bbox_h]

        # Quality flags
        quality_flags = []
        if inner_width < 1e-4 or inner_height < 1e-4:
            quality_flags.append(QualityFlag.SMALL_MOUTH.value)

        return normalized, mouth_bbox, quality_flags

    def generate_crops(
        self,
        frame: np.ndarray,
        mouth_bbox: List[float]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generate appearance and context crops from the original frame.

        Args:
            frame: BGR image (H, W, 3)
            mouth_bbox: [x, y, w, h] normalized in original frame

        Returns:
            Tuple of:
                - appearance_crop: [64, 64, 3] RGB mouth crop
                - context_crop: [112, 112, 1] grayscale full-face crop (if enabled)
        """
        height, width = frame.shape[:2]

        # Extract mouth bbox in pixel coordinates
        bx, by, bw, bh = mouth_bbox
        px1 = int(bx * width)
        py1 = int(by * height)
        px2 = int((bx + bw) * width)
        py2 = int((by + bh) * height)

        # Clamp to frame
        px1 = max(0, px1)
        py1 = max(0, py1)
        px2 = min(width, px2)
        py2 = min(height, py2)

        if px2 <= px1 or py2 <= py1:
            return None, None

        # Appearance crop (mouth region)
        mouth_region = frame[py1:py2, px1:px2]
        if mouth_region.size == 0:
            return None, None

        appearance_crop = cv2.resize(mouth_region, self.config.appearance_crop_size)
        appearance_crop = cv2.cvtColor(appearance_crop, cv2.COLOR_BGR2RGB)

        # Context crop (full face)
        # Use center of mouth as face center, crop 112x112 around it
        cx = int((px1 + px2) / 2)
        cy = int((py1 + py2) / 2)
        half_size = self.config.context_crop_size[0] // 2

        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(width, cx + half_size)
        y2 = min(height, cy + half_size)

        face_region = frame[y1:y2, x1:x2]
        if face_region.size == 0:
            context_crop = None
        else:
            context_crop = cv2.resize(face_region, self.config.context_crop_size)
            context_crop = cv2.cvtColor(context_crop, cv2.COLOR_BGR2GRAY)
            context_crop = context_crop[:, :, np.newaxis]  # Add channel dim

        return appearance_crop, context_crop


def estimate_head_pose(
    landmarks: np.ndarray,
    frame_shape: Tuple[int, int]
) -> List[float]:
    """
    Estimate head pose (pitch, yaw, roll) from landmarks.

    Args:
        landmarks: [468, 3] face landmarks (normalized)
        frame_shape: (height, width)

    Returns:
        [pitch, yaw, roll] in degrees
    """
    # 3D reference model in IMAGE coordinate convention
    # (x right, y DOWN, matching MediaPipe normalized landmark orientation).
    # Canonical anthropometric face model, nose tip at origin.
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (-225.0, -170.0, -135.0),    # Left eye left corner
        (225.0, -170.0, -135.0),     # Right eye right corner
        (-150.0, 150.0, -125.0),     # Left mouth corner
        (150.0, 150.0, -125.0),      # Right mouth corner
        (0.0, 330.0, -65.0),         # Chin (below nose in image coords)
        (0.0, -330.0, -65.0),        # Forehead (above nose in image coords)
    ], dtype=np.float64)

    # Image points
    image_points = np.array([
        landmarks[NOSE_TIP][:2] * [frame_shape[1], frame_shape[0]],
        landmarks[LEFT_EYE][:2] * [frame_shape[1], frame_shape[0]],
        landmarks[RIGHT_EYE][:2] * [frame_shape[1], frame_shape[0]],
        landmarks[LEFT_MOUTH][:2] * [frame_shape[1], frame_shape[0]],
        landmarks[RIGHT_MOUTH][:2] * [frame_shape[1], frame_shape[0]],
        landmarks[CHIN][:2] * [frame_shape[1], frame_shape[0]],
        landmarks[FOREHEAD][:2] * [frame_shape[1], frame_shape[0]],
    ], dtype=np.float64)

    # Camera matrix
    focal_length = frame_shape[1]
    center = (frame_shape[1] / 2, frame_shape[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    if not success:
        return [0.0, 0.0, 0.0]

    # Convert rotation vector to angles
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = 0.0
        y = 0.0
        z = 0.0

    pitch = float(np.degrees(x))
    yaw = float(np.degrees(y))
    roll = float(np.degrees(z))

    # Canonicalize to the principal branch (-90, +90]. solvePnP can converge
    # to an equivalent rotation whose Euler decomposition places a small
    # frontal pose near +-180 deg (e.g. roll = -178 instead of +2). For
    # dataset QC we only ever accept near-frontal poses, so map every angle
    # into the branch closest to zero.
    def _principal(angle_deg: float) -> float:
        a = (angle_deg + 180.0) % 360.0 - 180.0  # wrap to [-180, 180)
        if a > 90.0:
            a -= 180.0
        elif a < -90.0:
            a += 180.0
        return a

    pitch = _principal(pitch)
    yaw = _principal(yaw)
    roll = _principal(roll)

    return [pitch, yaw, roll]


class TrackingPipeline:
    """
    Full face tracking and mouth alignment pipeline.

    Processes a recording's video to generate:
    - Per-frame landmarks (488 total, 468 face + 20 mouth)
    - Mouth bounding boxes
    - Normalized mouth coordinates
    - Appearance crops
    - Context crops
    - Head pose
    - Quality flags
    """

    def __init__(self, config: TrackingConfig):
        self.config = config
        self.tracker = FaceTracker(config)
        self.aligner = MouthAligner(config)

    def process_recording(
        self,
        video_path: Path,
        recording_dir: Path,
        metadata: RecordingMetadata,
        max_frames: Optional[int] = None
    ) -> RecordingMetadata:
        """
        Process a recording's video and update metadata.

        Args:
            video_path: Path to recording video
            recording_dir: Recording directory (to save derived data)
            metadata: Existing metadata (will be updated)
            max_frames: Optional limit for testing

        Returns:
            Updated RecordingMetadata
        """
        import cv2

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        # Storage for derived data
        all_landmarks = []
        all_mouth_landmarks = []
        all_mouth_bboxes = []
        all_head_poses = []
        all_quality_flags = []
        all_appearance_crops = []
        all_context_crops = []

        # Statistics
        frames_with_face = 0
        frames_without_face = 0
        frames_with_mouth = 0
        missing_frames = 0
        max_consecutive_missing = 0
        current_missing = 0
        total_brightness = 0.0
        total_blur = 0.0
        total_mouth_width = 0.0
        total_mouth_height = 0.0
        total_pose_yaw = 0.0
        total_pose_pitch = 0.0
        total_pose_roll = 0.0
        min_mouth_area = float('inf')
        max_mouth_area = 0.0

        frame_idx = 0
        self.tracker.reset()

        try:
            while frame_idx < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Track face with REAL MediaPipe landmarks
                result = self.tracker.process_frame(
                    frame,
                    frame_index=frame_idx,
                    timestamp_ms=(frame_idx / fps * 1000.0) if fps > 0 else float(frame_idx),
                    speaker_id=metadata.speaker_id,
                )

                height, width = frame.shape[:2]

                if result.has_face and result.landmarks is not None:
                    frames_with_face += 1
                    current_missing = 0

                    # Extract mouth landmarks and bbox
                    mouth_lm, mouth_bbox, mouth_flags = self.aligner.extract_mouth_landmarks(
                        result.landmarks, (height, width)
                    )

                    if mouth_lm is not None:
                        frames_with_mouth += 1

                        # Pose estimation
                        pose = estimate_head_pose(result.landmarks, (height, width))
                        result.head_pose = pose

                        # Check pose angle limits
                        if any(abs(p) > self.config.max_pose_angle for p in pose):
                            result.quality_flags.append(QualityFlag.POSE_EXCEEDED.value)

                        # Generate crops
                        if self.config.save_frames or self.config.save_crops:
                            appearance_crop, context_crop = self.aligner.generate_crops(frame, mouth_bbox)
                            if appearance_crop is not None:
                                result.appearance_crop = appearance_crop
                            if context_crop is not None:
                                result.context_crop = context_crop

                        # Compute quality metrics
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
                        brightness = gray.mean()

                        total_blur += blur
                        total_brightness += brightness

                        if blur < self.config.blur_threshold:
                            result.quality_flags.append(QualityFlag.BLURRY.value)

                        if brightness < self.config.brightness_range[0]:
                            result.quality_flags.append(QualityFlag.DARK.value)
                        elif brightness > self.config.brightness_range[1]:
                            result.quality_flags.append(QualityFlag.BRIGHT.value)

                        # Mouth size
                        mouth_width = mouth_bbox[2] * width
                        mouth_height = mouth_bbox[3] * height
                        mouth_area = mouth_width * mouth_height

                        total_mouth_width += mouth_width
                        total_mouth_height += mouth_height
                        min_mouth_area = min(min_mouth_area, mouth_area)
                        max_mouth_area = max(max_mouth_area, mouth_area)

                        if mouth_area < self.config.min_mouth_size:
                            result.quality_flags.append(QualityFlag.SMALL_MOUTH.value)

                        # Pose accumulation
                        total_pose_yaw += pose[1]
                        total_pose_pitch += pose[0]
                        total_pose_roll += pose[2]

                        # Store data
                        all_landmarks.append(result.landmarks)
                        all_mouth_landmarks.append(mouth_lm)
                        all_mouth_bboxes.append(mouth_bbox)
                        all_head_poses.append(pose)
                        all_quality_flags.append(result.quality_flags)

                        if result.appearance_crop is not None:
                            all_appearance_crops.append(result.appearance_crop)
                        if result.context_crop is not None:
                            all_context_crops.append(result.context_crop)
                    else:
                        missing_frames += 1
                        current_missing += 1
                        max_consecutive_missing = max(max_consecutive_missing, current_missing)
                else:
                    frames_without_face += 1
                    missing_frames += 1
                    current_missing += 1
                    max_consecutive_missing = max(max_consecutive_missing, current_missing)

                frame_idx += 1

        finally:
            cap.release()
            self.tracker.close()

        # Update metadata
        valid_frames = frames_with_face
        metadata.face_tracking.frames_with_face = frames_with_face
        metadata.face_tracking.frames_without_face = frames_without_face
        metadata.face_tracking.max_consecutive_missing = max_consecutive_missing
        metadata.face_tracking.mean_confidence = self.tracker.track_confidence

        metadata.mouth.frames_with_mouth = frames_with_mouth
        metadata.mouth.missing_frames = missing_frames
        metadata.mouth.mean_mouth_width = total_mouth_width / max(1, frames_with_mouth)
        metadata.mouth.mean_mouth_height = total_mouth_height / max(1, frames_with_mouth)
        metadata.mouth.min_mouth_area = min_mouth_area if min_mouth_area != float('inf') else 0.0
        metadata.mouth.max_mouth_area = max_mouth_area

        metadata.quality.mean_brightness = total_brightness / max(1, valid_frames)
        metadata.quality.mean_blur = total_blur / max(1, valid_frames)
        metadata.quality.mean_mouth_size = (total_mouth_width * total_mouth_height) / max(1, frames_with_mouth ** 2)
        metadata.quality.mean_pose_yaw = total_pose_yaw / max(1, valid_frames)
        metadata.quality.mean_pose_pitch = total_pose_pitch / max(1, valid_frames)
        metadata.quality.mean_pose_roll = total_pose_roll / max(1, valid_frames)

        # Head pose per-frame storage
        metadata.head_pose = {
            "pitch": [p[0] for p in all_head_poses],
            "yaw": [p[1] for p in all_head_poses],
            "roll": [p[2] for p in all_head_poses],
        }

        # Save derived data
        self._save_derived_data(
            recording_dir, all_landmarks, all_mouth_landmarks,
            all_mouth_bboxes, all_head_poses, all_quality_flags,
            all_appearance_crops, all_context_crops
        )

        # Validation
        errors = validate_metadata(metadata)
        if errors:
            logger.warning(f"Metadata validation issues for {metadata.recording_id}: {errors}")

        return metadata

    def _save_derived_data(
        self,
        recording_dir: Path,
        landmarks: List[np.ndarray],
        mouth_landmarks: List[np.ndarray],
        mouth_bboxes: List[List[float]],
        head_poses: List[List[float]],
        quality_flags: List[List[str]],
        appearance_crops: List[np.ndarray],
        context_crops: List[np.ndarray]
    ) -> None:
        """Save derived data to recording directory."""

        # Save landmarks as .npz
        if self.config.save_landmarks_npz and landmarks:
            landmarks_dir = recording_dir / "landmarks"
            landmarks_dir.mkdir(exist_ok=True)

            # Save full face landmarks
            if landmarks:
                np.savez_compressed(
                    landmarks_dir / "face_landmarks.npz",
                    **{f"frame_{i:06d}": lm for i, lm in enumerate(landmarks)}
                )

            # Save mouth landmarks
            if mouth_landmarks:
                np.savez_compressed(
                    landmarks_dir / "mouth_landmarks.npz",
                    **{f"frame_{i:06d}": lm for i, lm in enumerate(mouth_landmarks)}
                )

            # Save mouth bboxes
            if mouth_bboxes:
                np.savez_compressed(
                    landmarks_dir / "mouth_bboxes.npz",
                    **{f"frame_{i:06d}": np.array(bbox) for i, bbox in enumerate(mouth_bboxes)}
                )

            # Save head poses
            if head_poses:
                np.savez_compressed(
                    landmarks_dir / "head_poses.npz",
                    **{f"frame_{i:06d}": np.array(pose) for i, pose in enumerate(head_poses)}
                )

        # Save crops as images (optional)
        if self.config.save_crops:
            crops_dir = recording_dir / "crops"
            crops_dir.mkdir(exist_ok=True)

            appearance_dir = crops_dir / "appearance"
            context_dir = crops_dir / "context"
            appearance_dir.mkdir(exist_ok=True)
            context_dir.mkdir(exist_ok=True)

            for i, crop in enumerate(appearance_crops):
                if crop is not None:
                    cv2.imwrite(str(appearance_dir / f"frame_{i:06d}.png"), crop)

            for i, crop in enumerate(context_crops):
                if crop is not None:
                    cv2.imwrite(str(context_dir / f"frame_{i:06d}.png"), crop)

        # Save frame data as JSON (lightweight, for inspection)
        frames_dir = recording_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        frame_data = []
        for i in range(len(quality_flags)):
            frame_data.append({
                "frame_index": i,
                "timestamp_ms": i * (metadata_fps_fallback := 33.33),  # Will be updated by caller
                "mouth_bbox": mouth_bboxes[i] if i < len(mouth_bboxes) else None,
                "head_pose": head_poses[i] if i < len(head_poses) else None,
                "quality_flags": quality_flags[i] if i < len(quality_flags) else [],
            })

        import json
        with open(frames_dir / "frame_data.json", 'w') as f:
            json.dump(frame_data, f, indent=2)


# Helper function to update frame timestamps in frame_data.json
def update_frame_timestamps(recording_dir: Path, fps: float) -> None:
    """Update timestamp field in frame_data.json with actual fps."""
    frames_dir = recording_dir / "frames"
    frame_data_path = frames_dir / "frame_data.json"
    if not frame_data_path.exists():
        return

    import json
    with open(frame_data_path) as f:
        frame_data = json.load(f)

    for fd in frame_data:
        fd["timestamp_ms"] = fd["frame_index"] / fps * 1000

    with open(frame_data_path, 'w') as f:
        json.dump(frame_data, f, indent=2)
