import sys
sys.path.insert(0, r'C:\Users\ravik\OneDrive\Desktop\lip alphabet')

# Test imports
print("Testing imports...")

# Test schema
from data.schema import RecordingMetadata, FaceTrackingInfo, MouthInfo, QualityMetrics, ManifestEntry, Letter, Split, QualityFlag, DATA_SOURCE_SYNTHETIC_TEST
print("✓ schema imports OK")

# Test manifest
from data.manifest import ManifestManager, ManifestConfig, create_splits, save_split_info
print("✓ manifest imports OK")

# Test ingestion
from data.ingestion import VideoIngestor, IngestionConfig, get_video_info, validate_video
print("✓ ingestion imports OK")

# Test tracking
from tracking.aligner import FaceTracker, MouthAligner, TrackingPipeline, TrackingConfig
print("✓ tracking imports OK")

# Test appearance
from repr.appearance import ResNetStem, AppearanceExtractor, AppearanceConfig
print("✓ appearance imports OK")

# Test geometry
from repr.geometry import GeometryScalars, GeometryMLP, GeometryExtractor, GeometryConfig
print("✓ geometry imports OK")

# Test motion
from repr.motion import MotionFeatureExtractor, MotionMLP, MotionExtractor, MotionConfig
print("✓ motion imports OK")

# Test fusion
from repr.fusion import FeatureProjectionNetwork, StreamProjector, GatedFusion, GatedFusionWithResidual, FusionConfig
print("✓ fusion imports OK")

# Test validate
from data.validate import validate_dataset, DatasetValidator
print("✓ validate imports OK")

# Test inspect
from data.inspect import inspect_dataset, DatasetInspector
print("✓ inspect imports OK")

# Test preprocess
from data.preprocess import PreprocessingPipeline, PreprocessingConfig
print("✓ preprocess imports OK")

# Test CLI
from alphabet.cli import main, validate_cmd, inspect_cmd, ingest_cmd
print("✓ CLI imports OK")

print("\n✅ ALL IMPORTS SUCCESSFUL!")

# Now run a simple test
print("\nRunning simple schema test...")
metadata = RecordingMetadata(
letter="A",
    speaker_id="speaker_001",
    session_id="session_001",
    recording_id="recording_001",
                            data_source=DATA_SOURCE_SYNTHETIC_TEST,
    source="webcam",
    created_at="2026-01-01T00:00:00Z",
    fps=30.0,
    frame_count=60,
    width=1280,
    height=720,
    duration_ms=2000.0,
    face_tracking=FaceTrackingInfo(
        frames_with_face=55, frames_without_face=5,
        max_consecutive_missing=2, mean_confidence=0.9, min_confidence=0.7
    ),
    mouth=MouthInfo(
        frames_with_mouth=50, missing_frames=10,
        mean_mouth_width=100, mean_mouth_height=50,
        min_mouth_area=2000, max_mouth_area=8000
    ),
    quality=QualityMetrics(
        mean_brightness=128, std_brightness=20,
        mean_blur=150, mean_mouth_size=5000,
        mean_pose_yaw=0, mean_pose_pitch=0, mean_pose_roll=0,
        duplicate_frame_count=0, quality_flags=[]
    ),
    preprocessing_version="1.0"
)

errors = []
if not Letter.is_valid(metadata.letter):
    errors.append("Invalid letter")
print(f"Letter validation: {'PASS' if not errors else 'FAIL'}")

errors = []
if metadata.fps <= 0:
    errors.append("Invalid fps")
print(f"FPS validation: {'PASS' if not errors else 'FAIL'}")

print("\n✅ SIMPLE TESTS PASSED!")