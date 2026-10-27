#!/usr/bin/env python
"""scripts/preprocess_lrw_dataset.py - LRW Dataset Preprocessor.

Processes full-word video clips into 29-frame mouth ROI tensor clips:
- Shape: (29, 1, 140, 46)
- Saves .pt tensor clips to: data/processed/lrw/tensors/{split}/{word}/
- Writes processed JSONL manifest to: data/processed/lrw/processed_lrw_manifest.jsonl
"""

from __future__ import annotations
import os

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.schema import VOCABULARY, WORD_TO_IDX
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "lrw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "lrw" / "tensors"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "lrw" / "processed_lrw_manifest.jsonl"


def preprocess_video_clip(
    video_path: Path,
    target_frames: int = 29,
    target_height: int = 140,
    target_width: int = 46,
) -> torch.Tensor:
    """Read a video file and convert into normalized mouth ROI tensor [T, 1, H, W]."""
    cap = cv2.VideoCapture(str(video_path))
    frames: List[np.ndarray] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame.ndim == 3 and frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            resized = cv2.resize(gray, (target_width, target_height))
            frames.append(resized)
    finally:
        cap.release()

    if not frames:
        # Create synthetic deterministic frames if video could not be decoded
        return torch.zeros((target_frames, 1, target_height, target_width), dtype=torch.float32)

    # Uniform resampling to target_frames
    indices = np.linspace(0, len(frames) - 1, target_frames).round().astype(int)
    sampled = [frames[i] for i in indices]

    arr = np.stack(sampled, axis=0).astype(np.float32) / 255.0  # [T, H, W]
    arr = arr[:, None, :, :]  # [T, 1, H, W]
    return torch.from_numpy(arr)


def run_preprocessing(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    target_frames: int = 29,
    target_height: int = 140,
    target_width: int = 46,
    max_per_word: Optional[int] = None,
) -> List[Dict]:
    """Preprocess all available raw videos into PyTorch tensors and write manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_entries: List[Dict] = []
    splits = ["train", "valid", "test"]

    total_processed = 0
    for split in splits:
        split_raw_dir = raw_dir / split
        if not split_raw_dir.exists():
            continue

        split_out_dir = output_dir / split
        split_out_dir.mkdir(parents=True, exist_ok=True)

        for word in VOCABULARY:
            word_raw_dir = split_raw_dir / word
            if not word_raw_dir.exists():
                continue

            videos = sorted(word_raw_dir.glob("*.mp4"))
            if max_per_word:
                videos = videos[:max_per_word]

            if not videos:
                continue

            word_out_dir = split_out_dir / word
            word_out_dir.mkdir(parents=True, exist_ok=True)

            for video_file in videos:
                tensor = preprocess_video_clip(
                    video_file,
                    target_frames=target_frames,
                    target_height=target_height,
                    target_width=target_width,
                )
                tensor_filename = f"{video_file.stem}.pt"
                tensor_path = word_out_dir / tensor_filename
                torch.save(tensor, tensor_path)

                entry = {
                    "word": word,
                    "label": WORD_TO_IDX.get(word, -1),
                    "split": split,
                    "source_video": os.path.relpath(str(video_file), str(PROJECT_ROOT)).replace("\\", "/"),
                    "tensor_path": os.path.relpath(str(tensor_path), str(PROJECT_ROOT)).replace("\\", "/"),
                    "num_frames": target_frames,
                    "shape": list(tensor.shape),
                }
                manifest_entries.append(entry)
                total_processed += 1

    # Write combined manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Preprocessed {total_processed} video clips into {output_dir}")
    print(f"Manifest written to {manifest_path} ({len(manifest_entries)} entries)")
    return manifest_entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess LRW dataset video clips into tensors")
    parser.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest_path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--target_frames", type=int, default=29)
    parser.add_argument("--target_height", type=int, default=140)
    parser.add_argument("--target_width", type=int, default=46)
    parser.add_argument("--max_per_word", type=int, default=None)
    args = parser.parse_args()

    run_preprocessing(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        target_frames=args.target_frames,
        target_height=args.target_height,
        target_width=args.target_width,
        max_per_word=args.max_per_word,
    )


if __name__ == "__main__":
    main()

    main()