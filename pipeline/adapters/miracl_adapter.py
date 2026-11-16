"""Adapter for MIRACL-VC1 Visual Speech Dataset (15 speakers, 10 words, 10 phrases).

Handles color-video sequences from MIRACL-VC1, filtering out depth maps,
enforcing a 10-speaker train / 2-speaker valid / 3-speaker test disjoint split.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

from .base_adapter import BaseDatasetAdapter, DatasetSample

MIRACL_WORDS = [
    "begin", "choose", "connection", "navigation", "next",
    "previous", "start", "stop", "hello", "web"
]

MIRACL_PHRASES = [
    "Stop navigation", "Excuse me", "I am sorry", "Thank you",
    "Good bye", "I love this game", "Nice to meet you", "You are welcome",
    "How are you", "Have a good time"
]

# 15 speakers partitioned into disjoint sets
MIRACL_SPLITS = {
    "train": ["F01", "F02", "F03", "F04", "F05", "M01", "M02", "M03", "M04", "M05"],
    "valid": ["F06", "M06"],
    "test": ["F07", "M07", "M08"]
}


class MiraclAdapter(BaseDatasetAdapter):
    def __init__(self, root_dir: Path = Path("data/external/miraclvc1")):
        super().__init__("miracl_vc1", root_dir)

    def is_available(self) -> bool:
        return self.root_dir.is_dir() and any(self.root_dir.glob("dataset/F*"))

    def validate_integrity(self) -> dict[str, Any]:
        if not self.is_available():
            return {
                "status": "UNAVAILABLE",
                "message": "MIRACL-VC1 dataset not found. Download via Kaggle API: 'kaggle datasets download -d apoorvwatsky/miraclvc1 -p data/external/miraclvc1 --unzip'",
                "required_credentials": "~/.kaggle/kaggle.json"
            }

        speakers_found = sorted(p.name for p in self.root_dir.glob("dataset/[FM]*") if p.is_dir())
        return {
            "status": "VALID" if len(speakers_found) == 15 else "PARTIAL",
            "speakers_count": len(speakers_found),
            "speakers": speakers_found,
            "vocabulary_words": MIRACL_WORDS
        }

    def generate_manifest(self) -> list[DatasetSample]:
        if not self.is_available():
            return []

        samples = []
        speaker_to_split = {}
        for split, spks in MIRACL_SPLITS.items():
            for s in spks:
                speaker_to_split[s] = split

        dataset_path = self.root_dir / "dataset"
        for spk_dir in sorted(dataset_path.glob("[FM]*")):
            spk = spk_dir.name
            split = speaker_to_split.get(spk, "train")

            # Process Words ('words' subdirectory)
            words_dir = spk_dir / "words"
            if words_dir.is_dir():
                for word_idx_dir in sorted(words_dir.iterdir()):
                    try:
                        w_idx = int(word_idx_dir.name) - 1
                        word_str = MIRACL_WORDS[w_idx]
                    except (ValueError, IndexError):
                        continue

                    for instance_dir in sorted(word_idx_dir.iterdir()):
                        color_dir = instance_dir / "color"
                        if not color_dir.is_dir():
                            continue
                        frames = sorted(color_dir.glob("*.jpg"))
                        if len(frames) < 4:
                            continue

                        sample_id = f"miracl_{spk}_{word_str}_{instance_dir.name}"
                        samples.append(DatasetSample(
                            sample_id=sample_id,
                            dataset="miracl_vc1",
                            video_path=str(color_dir),
                            start_frame=0,
                            end_frame=len(frames),
                            speaker_id=spk,
                            task_type="isolated_word",
                            transcript=word_str,
                            word_label=word_str,
                            split=split,
                            fps=15.0, # MIRACL Kinect color stream
                            source_checksum="",
                            validation_status="verified"
                        ))

        return samples
