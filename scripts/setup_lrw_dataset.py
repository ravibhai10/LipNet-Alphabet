#!/usr/bin/env python
"""scripts/setup_lrw_dataset.py - LRW Dataset Setup and Mock Generator.

Creates the full 500-word directory structure for LRW (train, valid, test)
and generates development-ready sample fixtures when the 70GB full archive
is not present locally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LRW_DIR = DATA_DIR / "lrw"
VOCAB_PATH = DATA_DIR / "vocab.txt"
PROCESSED_DIR = DATA_DIR / "processed" / "lrw"
TRAIN_DIR = LRW_DIR / "train"
VALID_DIR = LRW_DIR / "valid"
TEST_DIR = LRW_DIR / "test"


def load_vocabulary() -> List[str]:
    """Load 500 word list from vocab.txt."""
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"Vocab file not found at {VOCAB_PATH}")
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        words = [line.strip().upper() for line in f if line.strip()]
    if len(words) != 500:
        print(f"Warning: Expected 500 words, found {len(words)}")
    return words


def create_directory_structure(words: List[str]) -> Dict[str, bool]:
    """Create train/valid/test directories for all 500 words."""
    created = {}
    splits = [("train", TRAIN_DIR), ("valid", VALID_DIR), ("test", TEST_DIR)]
    for split_name, split_dir in splits:
        try:
            split_dir.mkdir(parents=True, exist_ok=True)
            for word in words:
                (split_dir / word).mkdir(parents=True, exist_ok=True)
            created[split_name] = True
        except Exception as exc:
            print(f"Error creating {split_name} directories: {exc}")
            created[split_name] = False
    return created


def create_mock_video(path: Path, num_frames: int = 29, height: int = 140, width: int = 46) -> None:
    """Generate a valid MP4 test clip with specified dimensions and frame count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 25.0, (width, height))
    for i in range(num_frames):
        # Create a synthetic mouth ROI frame with subtle pattern
        frame = np.full((height, width, 3), 40 + (i % 30), dtype=np.uint8)
        # Add an inner lip-like ellipse
        cv2.ellipse(frame, (width // 2, height // 2), (width // 4, height // 6), 0, 0, 360, (180, 80, 100), -1)
        out.write(frame)
    out.release()


def populate_development_samples(words: List[str], samples_per_split: int = 10) -> int:
    """Populate sample MP4 videos for key words across splits to enable real I/O testing."""
    sample_words = words[:samples_per_split] + ["HELLO", "YOUNG"] if "HELLO" in words and "YOUNG" in words else words[:samples_per_split]
    sample_words = list(dict.fromkeys(sample_words))
    
    created_count = 0
    splits = [("train", TRAIN_DIR), ("valid", VALID_DIR), ("test", TEST_DIR)]
    for split_name, split_dir in splits:
        for word in sample_words:
            video_file = split_dir / word / f"{word.lower()}_00001.mp4"
            if not video_file.exists():
                create_mock_video(video_file, num_frames=29, height=140, width=46)
                created_count += 1
    return created_count


def generate_manifests(words: List[str]) -> Dict[str, Path]:
    """Generate reference manifests in data/processed/lrw/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest_paths = {}
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    splits = [("train", TRAIN_DIR), ("valid", VALID_DIR), ("test", TEST_DIR)]
    for split_name, split_dir in splits:
        manifest_file = PROCESSED_DIR / f"lrw_{split_name}_manifest.jsonl"
        entries = []
        
        # Scan for existing MP4s
        for word in words:
            word_folder = split_dir / word
            videos = list(word_folder.glob("*.mp4"))
            label = word_to_idx[word]
            if videos:
                for v in videos:
                    entries.append({
                        "word": word,
                        "label": label,
                        "split": split_name,
                        "video_name": v.name,
                        "file_path": str(v.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "num_frames": 29
                    })
            else:
                # Reference mock entry for downstream manifests
                entries.append({
                    "word": word,
                    "label": label,
                    "split": split_name,
                    "video_name": f"{word.lower()}_mock.mp4",
                    "file_path": f"data/lrw/{split_name}/{word}/{word.lower()}_mock.mp4",
                    "num_frames": 29
                })
                
        with open(manifest_file, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        manifest_paths[split_name] = manifest_file
        print(f"Generated {manifest_file.name} ({len(entries)} entries)")
        
    return manifest_paths


def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    words = load_vocabulary()
    print(f"Loaded vocabulary: {len(words)} words")
    
    dirs_created = create_directory_structure(words)
    print(f"Split directories validated: {dirs_created}")
    
    clips_created = populate_development_samples(words, samples_per_split=15)
    print(f"Development video clips created: {clips_created}")
    
    manifests = generate_manifests(words)
    print("Manifest generation complete.")


if __name__ == "__main__":
    main()
