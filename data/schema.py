"""
Dataset schema definitions for the Alphabet Visual Speech dataset.

This module defines the data structures, validation schemas, and type annotations
for recordings, metadata, manifests, and quality assessments.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Literal
import json
from pathlib import Path


class Letter(str, Enum):
    """Supported alphabet letters."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"
    I = "I"
    J = "J"
    K = "K"
    L = "L"
    M = "M"
    N = "N"
    O = "O"
    P = "P"
    Q = "Q"
    R = "R"
    S = "S"
    T = "T"
    U = "U"
    V = "V"
    W = "W"
    X = "X"
    Y = "Y"
    Z = "Z"

    @classmethod
    def all(cls) -> List[str]:
        return [l.value for l in cls]

    @classmethod
    def is_valid(cls, letter: str) -> bool:
        """Return True for a single uppercase A-Z letter (canonical label).

        Lowercase input is NOT accepted here; normalization to uppercase is the
        caller's responsibility (e.g. ``RecordingMetadata.__post_init__``).
        """
        return isinstance(letter, str) and letter in cls.all()


class Split(str, Enum):
    """Dataset split identifiers."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


# Landmark provenance values (plain strings, not enum members, so that
# metadata.json stays forward-compatible).
#: Landmarks produced by the real MediaPipe FaceLandmarker. The ONLY value
#: accepted in production datasets.
LANDMARK_SOURCE_MEDIAPIPE = "mediapipe"
#: Landmarks synthesized inside isolated unit tests. NEVER valid in a
#: production dataset; validators must reject it.
LANDMARK_SOURCE_SYNTHETIC_TEST = "synthetic_test"

#: REAL DATA SOURCE VALVES (§6 Phase-6 DATA GATE)
DATA_SOURCE_REAL_CAMERA = "real_camera"
DATA_SOURCE_SYNTHETIC_TEST = "synthetic_test"

REAL_DATA_ONLY_SOURCES = {DATA_SOURCE_REAL_CAMERA}
TEST_SYNTHETIC_SOURCES = {DATA_SOURCE_SYNTHETIC_TEST}

#: Sentinel value used in the RecordingMetadata/ManifestEntry dataclass defaults
#: to FORCE production metadata to explicitly state its data_source. Anything
#: loaded with this sentinel will fail validation in :func:`validate_metadata`
#: / :func:`validate_manifest_entry` and the dataset-level validator, instead
#: of silently defaulting to "real_camera" (which previously let synthetic
#: data masquerade as real). The sentinel is intentionally NOT a member of
#: ``REAL_DATA_ONLY_SOURCES`` or ``TEST_SYNTHETIC_SOURCES``.
DATA_SOURCE_UNSPECIFIED = "__UNSPECIFIED__"


class SourceType(str, Enum):
    """Source type for the recording."""
    WEBCAM = "webcam"
    LOCAL_MP4 = "local_mp4"
    YOUTUBE = "youtube"  # Future extension


class QualityFlag(str, Enum):
    """Quality assessment flags for frames/recordings."""
    OK = "ok"
    BLURRY = "blurry"
    DARK = "dark"
    BRIGHT = "bright"
    NO_FACE = "no_face"
    NO_MOUTH = "no_mouth"
    POSE_EXCEEDED = "pose_exceeded"
    CORRUPTED = "corrupted"
    DUPLICATE = "duplicate"
    SMALL_MOUTH = "small_mouth"


@dataclass
class FaceTrackingInfo:
    """Face tracking statistics for a recording."""
    frames_with_face: int = 0
    frames_without_face: int = 0
    max_consecutive_missing: int = 0
    mean_confidence: float = 0.0
    min_confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FaceTrackingInfo":
        return cls(**data)


@dataclass
class MouthInfo:
    """Mouth detection statistics for a recording."""
    frames_with_mouth: int = 0
    missing_frames: int = 0
    mean_mouth_width: float = 0.0
    mean_mouth_height: float = 0.0
    min_mouth_area: float = 0.0
    max_mouth_area: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MouthInfo":
        return cls(**data)


@dataclass
class QualityMetrics:
    """Quality assessment metrics for a recording."""
    mean_brightness: float = 0.0
    std_brightness: float = 0.0
    mean_blur: float = 0.0  # Laplacian variance
    mean_mouth_size: float = 0.0  # pixels^2
    mean_pose_yaw: float = 0.0
    mean_pose_pitch: float = 0.0
    mean_pose_roll: float = 0.0
    duplicate_frame_count: int = 0
    quality_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityMetrics":
        return cls(**data)


@dataclass
class RecordingMetadata:
    """
    Complete metadata for a single recording.

    This is the primary metadata structure saved as metadata.json
    in each recording directory.
    """
    # Identity
    letter: str
    speaker_id: str
    session_id: str
    recording_id: str

    # Source information
    source: str  # SourceType value
    created_at: str  # ISO format timestamp

    # Video properties
    fps: float
    frame_count: int
    width: int
    height: int
    duration_ms: float

    # Tracking and detection statistics
    face_tracking: FaceTrackingInfo
    mouth: MouthInfo

    # Quality metrics
    quality: QualityMetrics

    # Processing information
    preprocessing_version: str
    config_version: str = "1.0"
    config_hash: str = ""
    schema_version: str = "1.0"

    # HARD DATA SOURCE GATE (§6 Phase 6 DATA GATE)
    # Production training/validation/test MUST be "real_camera".
    # Synthetic/test fixtures use "synthetic_test" and are rejected by validator.
    #
    # The default is the explicit sentinel ``DATA_SOURCE_UNSPECIFIED`` rather
    # than a permissive value like "real_camera". This forces every caller
    # (record, ingest_local_mp4, tests, the synthetic generator) to set the
    # data_source EXPLICITLY. Missing data_source => validation error.
    data_source: str = DATA_SOURCE_UNSPECIFIED

    # Provenance of face landmarks. Production datasets MUST use
    # LANDMARK_SOURCE_MEDIAPIPE; anything else is rejected by validation.
    landmark_source: str = LANDMARK_SOURCE_MEDIAPIPE

    # Optional extended fields
    head_pose: Optional[Dict[str, List[float]]] = None  # per-frame pitch, yaw, roll
    lighting_tag: Optional[str] = None
    accent_tag: Optional[str] = None
    notes: Optional[str] = None

    def __post_init__(self):
        """Normalize fields after initialization."""
        # Canonicalize the label to uppercase. Full validity is enforced by
        # validate_metadata, not at construction, so that malformed data read
        # from disk (or produced by corrupt files) can be surfaced as a
        # validation error instead of being silently dropped.
        if self.letter:
            self.letter = self.letter.upper()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert nested dataclasses
        data["face_tracking"] = self.face_tracking.to_dict()
        data["mouth"] = self.mouth.to_dict()
        data["quality"] = self.quality.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecordingMetadata":
        """Create from dictionary (JSON deserialization).

        If ``data_source`` is absent from the dictionary, the dataclass default
        (``DATA_SOURCE_UNSPECIFIED``) is used, which will cause validation to
        fail. This is intentional — missing data_source must NOT silently default
        to "real_camera".
        """
        # Convert nested dataclasses
        data = data.copy()
        data["face_tracking"] = FaceTrackingInfo.from_dict(data["face_tracking"])
        data["mouth"] = MouthInfo.from_dict(data["mouth"])
        data["quality"] = QualityMetrics.from_dict(data["quality"])
        return cls(**data)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "RecordingMetadata":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, path: Path) -> None:
        """Save metadata to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: Path) -> "RecordingMetadata":
        """Load metadata from file."""
        return cls.from_json(path.read_text())


@dataclass
class ManifestEntry:
    """
    Single entry in the dataset manifest (manifest.jsonl).

    One line per recording with essential indexing information.
    """
    recording_id: str
    speaker_id: str
    session_id: str
    letter: str
    path: str  # Relative path from dataset root
    frame_count: int
    duration_ms: float
    valid_frame_count: int
    quality_summary: Dict[str, Any]
    preprocessing_version: str
    config_version: str = "1.0"
    config_hash: str = ""
    # Landmark provenance; production datasets must be "mediapipe".
    landmark_source: str = LANDMARK_SOURCE_MEDIAPIPE
    # HARD DATA SOURCE GATE (§6 Phase 6 DATA GATE)
    # Must be set explicitly; __UNSPECIFIED__ sentinel fails validation.
    data_source: str = DATA_SOURCE_UNSPECIFIED
    split: Optional[str] = None  # train/val/test
    schema_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_jsonl(self) -> str:
        """Serialize to JSONL line."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_jsonl(cls, line: str) -> "ManifestEntry":
        """Deserialize from JSONL line."""
        return cls.from_dict(json.loads(line))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManifestEntry":
        return cls(**data)

    @classmethod
    def from_metadata(cls, metadata: RecordingMetadata, rel_path: str, split: Optional[str] = None) -> "ManifestEntry":
        """Create manifest entry from recording metadata."""
        return cls(
            recording_id=metadata.recording_id,
            speaker_id=metadata.speaker_id,
            session_id=metadata.session_id,
            letter=metadata.letter,
            path=rel_path,
            frame_count=metadata.frame_count,
            duration_ms=metadata.duration_ms,
            valid_frame_count=metadata.face_tracking.frames_with_face,
            quality_summary={
                "mean_brightness": metadata.quality.mean_brightness,
                "mean_blur": metadata.quality.mean_blur,
                "mean_mouth_size": metadata.quality.mean_mouth_size,
                "mean_pose_yaw": metadata.quality.mean_pose_yaw,
                "mean_pose_pitch": metadata.quality.mean_pose_pitch,
                "mean_pose_roll": metadata.quality.mean_pose_roll,
                "duplicate_frame_count": metadata.quality.duplicate_frame_count,
                "flags": metadata.quality.quality_flags,
            },
            preprocessing_version=metadata.preprocessing_version,
            config_version=metadata.config_version,
            config_hash=metadata.config_hash,
            landmark_source=metadata.landmark_source,
            data_source=metadata.data_source,
            split=split,
            schema_version=metadata.schema_version,
        )


@dataclass
class FrameData:
    """
    Per-frame data structure for derived data (landmarks, crops, motion).

    This represents the data for a single frame within a recording.
    """
    frame_index: int
    timestamp_ms: float

    # Face landmarks (normalized 0-1 coordinates, 468 points from MediaPipe)
    landmarks: Optional[List[List[float]]] = None  # [468, 3] - x, y, z

    # Mouth-specific landmarks (normalized to mouth-centered frame)
    mouth_landmarks: Optional[List[List[float]]] = None  # [20, 2] - x, y

    # Mouth bounding box in original frame coordinates
    mouth_bbox: Optional[List[float]] = None  # [x, y, w, h] normalized 0-1

    # Geometry scalars (20 canonical values)
    geometry_scalars: Optional[List[float]] = None

    # Motion features
    landmark_velocity: Optional[List[List[float]]] = None  # [20, 2]
    landmark_acceleration: Optional[List[List[float]]] = None  # [20, 2]
    geometry_velocity: Optional[List[float]] = None  # [20]
    geometry_acceleration: Optional[List[float]] = None  # [20]
    optical_flow: Optional[List[List[List[float]]]] = None  # [16, 16, 2]

    # Quality flags for this frame
    quality_flags: List[str] = field(default_factory=list)

    # Head pose (pitch, yaw, roll in degrees)
    head_pose: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameData":
        return cls(**data)


@dataclass
class SequenceData:
    """
    Complete sequence data for a recording (all frames).

    This is the structure used for model input after loading.
    """
    metadata: RecordingMetadata
    frames: List[FrameData]

    def __len__(self) -> int:
        return len(self.frames)

    def get_valid_frames(self) -> List[FrameData]:
        """Return frames without critical quality flags."""
        critical_flags = {QualityFlag.CORRUPTED, QualityFlag.NO_FACE, QualityFlag.NO_MOUTH}
        return [
            f for f in self.frames
            if not any(flag in critical_flags for flag in f.quality_flags)
        ]


@dataclass
class SpeakerSplit:
    """Speaker split assignment for dataset partitioning."""
    speaker_id: str
    split: Split
    sessions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeakerSplit":
        data = data.copy()
        data["split"] = Split(data["split"])
        return cls(**data)


@dataclass
class DatasetSplitInfo:
    """Information about dataset splits for validation."""
    train_speakers: List[str]
    val_speakers: List[str]
    test_speakers: List[str]
    train_sessions: List[str]
    val_sessions: List[str]
    test_sessions: List[str]
    split_seed: int
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetSplitInfo":
        return cls(**data)


# Validation functions
def validate_metadata(metadata: RecordingMetadata) -> List[str]:
    """
    Validate recording metadata and return list of errors (empty if valid).
    """
    errors = []

    # Check required fields
    if not metadata.letter or not Letter.is_valid(metadata.letter):
        errors.append(f"Invalid letter: {metadata.letter}")

    if not metadata.speaker_id:
        errors.append("Missing speaker_id")

    if not metadata.session_id:
        errors.append("Missing session_id")

    if not metadata.recording_id:
        errors.append("Missing recording_id")

    # Check video properties
    if metadata.fps <= 0:
        errors.append(f"Invalid fps: {metadata.fps}")

    if metadata.frame_count <= 0:
        errors.append(f"Invalid frame_count: {metadata.frame_count}")

    if metadata.width <= 0 or metadata.height <= 0:
        errors.append(f"Invalid resolution: {metadata.width}x{metadata.height}")

    if metadata.duration_ms <= 0:
        errors.append(f"Invalid duration: {metadata.duration_ms}")

    # Check tracking stats consistency
    ft = metadata.face_tracking
    if ft.frames_with_face + ft.frames_without_face != metadata.frame_count:
        errors.append(
            f"Face tracking frame count mismatch: "
            f"{ft.frames_with_face} + {ft.frames_without_face} != {metadata.frame_count}"
        )

    # Check mouth stats
    if metadata.mouth.frames_with_mouth + metadata.mouth.missing_frames > metadata.frame_count:
        errors.append("Mouth frame count exceeds total frames")

    # Check quality metrics
    q = metadata.quality
    if q.mean_blur < 0:
        errors.append(f"Negative blur metric: {q.mean_blur}")

    if q.mean_mouth_size < 0:
        errors.append(f"Negative mouth size: {q.mean_mouth_size}")

    # Landmark provenance: production data must come from the real MediaPipe
    # FaceLandmarker. Synthetic/test landmarks are rejected here so they can
    # never enter manifest.jsonl or training data.
    if metadata.landmark_source != LANDMARK_SOURCE_MEDIAPIPE:
        errors.append(
            f"landmark_source must be '{LANDMARK_SOURCE_MEDIAPIPE}' for "
            f"production datasets, got '{metadata.landmark_source}'"
        )

    # HARD DATA SOURCE GATE (§6 Phase 6 DATA GATE) — catch the
    # __UNSPECIFIED__ sentinel that replaces the old permissive "real_camera"
    # default. This ensures every recording must explicitly declare its source.
    if metadata.data_source == DATA_SOURCE_UNSPECIFIED:
        errors.append(
            "data_source is REQUIRED and was not specified "
            "(must be 'real_camera' for production, 'synthetic_test' for fixtures)"
        )
    elif metadata.data_source not in REAL_DATA_ONLY_SOURCES:
        errors.append(
            f"data_source must be '{DATA_SOURCE_REAL_CAMERA}' for production "
            f"datasets; got '{metadata.data_source}'"
        )

    return errors


def validate_manifest_entry(entry: ManifestEntry) -> List[str]:
    """Validate a manifest entry."""
    errors = []

    if not entry.recording_id:
        errors.append("Missing recording_id")

    if not entry.speaker_id:
        errors.append("Missing speaker_id")

    if not entry.session_id:
        errors.append("Missing session_id")

    if not entry.letter or not Letter.is_valid(entry.letter):
        errors.append(f"Invalid letter: {entry.letter}")

    if not entry.path:
        errors.append("Missing path")

    if entry.frame_count <= 0:
        errors.append(f"Invalid frame_count: {entry.frame_count}")

    if entry.duration_ms <= 0:
        errors.append(f"Invalid duration: {entry.duration_ms}")

    if entry.valid_frame_count < 0 or entry.valid_frame_count > entry.frame_count:
        errors.append(f"Invalid valid_frame_count: {entry.valid_frame_count}")

    if entry.split and entry.split not in [s.value for s in Split]:
        errors.append(f"Invalid split: {entry.split}")

    # Landmark provenance: synthetic/test landmarks are forbidden in
    # production datasets.
    if entry.landmark_source != LANDMARK_SOURCE_MEDIAPIPE:
        errors.append(
            f"landmark_source must be '{LANDMARK_SOURCE_MEDIAPIPE}' for "
            f"production datasets, got '{entry.landmark_source}'"
        )

    # HARD DATA SOURCE GATE (§6 Phase 6 DATA GATE)
    if entry.data_source == DATA_SOURCE_UNSPECIFIED:
        errors.append(
            "data_source is REQUIRED (must be 'real_camera' for production, "
            "'synthetic_test' for fixtures)"
        )
    elif entry.data_source not in REAL_DATA_ONLY_SOURCES:
        errors.append(
            f"data_source must be '{DATA_SOURCE_REAL_CAMERA}' for production "
            f"datasets; got '{entry.data_source}'"
        )

    return errors


def validate_frame_ordering(frames: List[FrameData]) -> List[str]:
    """Validate that frames are in correct chronological order."""
    errors = []

    if not frames:
        return errors

    # Check frame_index is sequential
    for i, frame in enumerate(frames):
        if frame.frame_index != i:
            errors.append(f"Frame index mismatch at position {i}: expected {i}, got {frame.frame_index}")

    # Check timestamps are monotonic
    for i in range(1, len(frames)):
        if frames[i].timestamp_ms < frames[i-1].timestamp_ms:
            errors.append(f"Timestamp not monotonic at frame {i}: {frames[i].timestamp_ms} < {frames[i-1].timestamp_ms}")

    return errors