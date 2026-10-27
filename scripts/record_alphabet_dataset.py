"""Record labeled alphabet videos without modifying existing project assets.

Each recording is written below:
    <output>/recordings/<LETTER>/<SPEAKER>/<SESSION>/<RECORDING>/

This creates raw videos and metadata only. A separate, validated mouth-ROI
preprocessing step is still required before the CNN-BiLSTM can be trained.
Press SPACE to start/stop a clip and ESC to quit.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2


def record_clip(
    output_dir: Path,
    letter: str,
    speaker: str,
    session: str,
    recording_id: str,
    *,
    camera: int,
    fps: float,
    width: int,
    height: int,
    duration: float,
    preview: bool,
) -> Path | None:
    target = output_dir / "recordings" / letter / speaker / session / recording_id
    target.mkdir(parents=True, exist_ok=False)
    video_path = target / "video.mp4"
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        target.rmdir()
        raise RuntimeError(f"cannot open webcam {camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        video_path.unlink(missing_ok=True)
        target.rmdir()
        raise RuntimeError(f"cannot create video writer: {video_path}")

    frames = 0
    started = time.monotonic()
    try:
        while time.monotonic() - started < duration:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("webcam returned an unreadable frame")
            frames += 1
            writer.write(frame)
            if preview:
                cv2.imshow("Alphabet recording (SPACE/ESC)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    return None
    finally:
        writer.release()
        cap.release()
        if preview:
            cv2.destroyAllWindows()

    metadata = {
        "letter": letter,
        "speaker_id": speaker,
        "session_id": session,
        "recording_id": recording_id,
        "data_source": "real_camera",
        "source": "webcam",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fps": float(actual_fps),
        "frame_count": frames,
        "width": width,
        "height": height,
        "duration_ms": (frames / actual_fps * 1000.0) if actual_fps else 0.0,
        "preprocessing_status": "raw_video_only",
    }
    (target / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/external/alphabet_webcam_dataset"))
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--session", default="session_001")
    parser.add_argument("--letters", default="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    parser.add_argument("--samples-per-letter", type=int, default=5)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    if args.samples_per_letter <= 0 or args.duration <= 0:
        parser.error("samples-per-letter and duration must be positive")
    letters = "".join(dict.fromkeys(args.letters.upper()))
    if any(letter < "A" or letter > "Z" for letter in letters):
        parser.error("letters must contain only A-Z")

    for letter in letters:
        for index in range(1, args.samples_per_letter + 1):
            recording_id = f"recording_{index:03d}"
            print(f"Prepare to say letter {letter}, sample {index}/{args.samples_per_letter}.")
            input("Press Enter to record, or Ctrl+C to stop: ")
            path = record_clip(
                args.output, letter, args.speaker, args.session, recording_id,
                camera=args.camera, fps=args.fps, width=args.width,
                height=args.height, duration=args.duration,
                preview=not args.no_preview,
            )
            if path is None:
                print("Recording cancelled.")
                return 0
            print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
