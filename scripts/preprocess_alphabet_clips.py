"""Convert recorded alphabet videos into the classifier's fixed input clips.

The input videos and metadata are never modified. Outputs are written below a
separate directory as ``<letter>/clip_<recording_id>.npy`` and a JSONL
manifest containing speaker IDs for leakage-safe downstream splitting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow direct execution from the repository root or any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking.aligner import (
    FaceTracker,
    TrackingConfig,
    RealFaceLandmarksUnavailableError,
)
from src.video_training import resample_temporal
from tracking.mouth_canonicalization import aligned_mouth_crop

TARGET_FRAMES = 30
TARGET_HEIGHT = 140
TARGET_WIDTH = 46


def temporal_resample(frames: list[np.ndarray], target: int = TARGET_FRAMES) -> np.ndarray:
    """Linearly resample the complete clip to ``[target,H,W]`` uint8."""
    if not frames:
        raise ValueError("cannot resample an empty frame sequence")
    interpolated = resample_temporal(frames, target)
    return np.clip(np.rint(interpolated), 0, 255).astype(np.uint8)


def _read_metadata(path: Path) -> dict:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid metadata file {path}: {exc}") from exc
    required = ("letter", "speaker_id", "session_id", "recording_id")
    missing = [field for field in required if not metadata.get(field)]
    if missing:
        raise ValueError(f"{path} missing required fields: {', '.join(missing)}")
    letter = str(metadata["letter"]).upper()
    if len(letter) != 1 or not "A" <= letter <= "Z":
        raise ValueError(f"{path} has invalid letter {letter!r}")
    return metadata


def process_video(
    video_path: Path,
    *,
    model_path: Path,
    tracker_config: TrackingConfig | None = None,
) -> np.ndarray:
    """Extract a real mouth ROI clip with shape ``(30, 140, 46)``."""
    config = tracker_config or TrackingConfig(
        appearance_crop_size=(TARGET_WIDTH, TARGET_HEIGHT),
        save_crops=False,
        save_landmarks_npz=False,
    )
    tracker = FaceTracker(config, model_path=model_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        tracker.close()
        raise RuntimeError(f"cannot open video: {video_path}")

    frames: list[np.ndarray] = []
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = tracker.process_frame(
                frame,
                frame_index=index,
                timestamp_ms=index * 1000.0 / fps,
            )
            if result.has_face and result.landmarks is not None and "no_face" not in (result.quality_flags or []):
                aligned = aligned_mouth_crop(frame, result.landmarks, size=96)
                gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                frames.append(cv2.resize(gray, (TARGET_WIDTH, TARGET_HEIGHT)))
            index += 1
    finally:
        capture.release()
        tracker.close()
    if not frames:
        raise ValueError(f"no real mouth crops detected in {video_path}")
    return temporal_resample(frames)


def process_dataset(input_root: Path, output_root: Path, model_path: Path) -> int:
    records: list[dict] = []
    for metadata_path in sorted(input_root.glob("recordings/*/*/*/*/metadata.json")):
        metadata = _read_metadata(metadata_path)
        recording_dir = metadata_path.parent
        video_path = recording_dir / "video.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"video missing for {metadata_path}: {video_path}")
        clip = process_video(video_path, model_path=model_path)
        output_dir = output_root / metadata["letter"]
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"clip_{metadata['recording_id']}.npy"
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite existing clip: {output_path}")
        np.save(output_path, clip)
        records.append({
            **{key: metadata[key] for key in (
                "letter", "speaker_id", "session_id", "recording_id"
            )},
            "clip_path": str(output_path.relative_to(output_root)),
            "shape": list(clip.shape),
            "alignment": "measured-eye-axis-mouth-center-v1",
            "temporal_resampling": "linear",
            "data_source": metadata.get("data_source", "unverified"),
        })

    if not records:
        raise ValueError(f"no recording metadata found under {input_root}")
    manifest_path = output_root / "alphabet_manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {manifest_path}")
    manifest_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "frontend/public/models/face_landmarker.task",
    )
    args = parser.parse_args()
    count = process_dataset(args.input, args.output, args.model)
    print(f"Processed {count} alphabet recording(s) into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
