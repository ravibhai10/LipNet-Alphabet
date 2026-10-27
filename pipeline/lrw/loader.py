#!/usr/bin/env python
"""pipeline/lrw/loader.py - LRW Dataset Loader for LipNet-Alphabet.

Provides PyTorch Dataset implementation for the 500-word LRW vocabulary.
Scans data/lrw/{split}/{word}/*.mp4 and preprocessed tensor clips.
Returns (X, Y) where X in R^{T x 1 x 140 x 46} and Y in {0, ..., 499}.
Supports spatial jitter, temporal masking, and photometric augmentations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.video_training import augment_video, resample_temporal

from data.schema import (
    VOCABULARY,
    WORD_TO_IDX,
    IDX_TO_WORD,
    NUM_CLASSES,
    get_word_idx,
    get_word,
)


def default_lrw_augmentation(tensor: torch.Tensor) -> torch.Tensor:
    """Apply spatial jitter, temporal masking, and brightness jitter to [T, 1, H, W]."""
    return augment_video(tensor)


class LRWSample(tuple):
    """Tuple (video, label) supporting tuple unpacking (x, y = sample),

    attribute access (sample.video, sample.label), and key access (sample['video']).
    """

    def __new__(
        cls,
        video: torch.Tensor,
        label: int,
        word: str,
        file_path: Optional[str] = None,
    ):
        return super().__new__(cls, (video, label))

    def __init__(
        self,
        video: torch.Tensor,
        label: int,
        word: str,
        file_path: Optional[str] = None,
    ):
        self.video = video
        self.label = label
        self.word = word
        self.file_path = file_path

    def __getitem__(self, item):
        if isinstance(item, str):
            if item == "video":
                return self.video
            if item == "label":
                return self.label
            if item == "word":
                return self.word
            if item == "file_path":
                return self.file_path
            raise KeyError(item)
        return super().__getitem__(item)

    def keys(self):
        return ["video", "label", "word", "file_path"]

    def __contains__(self, key):
        return key in ("video", "label", "word", "file_path")


class LRWDataset(Dataset):
    """LRW (Lip Reading in the Wild) Dataset for 500-word classification.

    Scans directory structure:
        data/lrw/{split}/{word}/*.mp4
        or preprocessed tensor clips:
        data/processed/lrw/tensors/{split}/{word}/*.pt

    Returns:
        - input tensor X in R^{T x 1 x 140 x 46} (T frames, 1 channel, mouth ROI)
        - label index Y in {0, 1, ..., 499}
    """

    def __init__(
        self,
        root_dir: str | Path = "data/lrw",
        split: str = "train",
        transform=None,
        augment: bool = False,
        max_frames: int = 29,
        target_height: int = 140,
        target_width: int = 46,
        use_mock: bool = False,
        preprocessed_dir: Optional[str | Path] = "data/processed/lrw/tensors",
        samples: Optional[List[Dict]] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        self.transform = transform
        self.augment = augment
        self.max_frames = max_frames
        self.target_height = target_height
        self.target_width = target_width
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None

        self.word_to_idx = WORD_TO_IDX
        self.idx_to_word = IDX_TO_WORD
        self.num_classes = NUM_CLASSES

        self.split_dir = self.root_dir / self.split
        self.samples: List[Dict[str, Union[str, int, Path]]] = []
        self.use_mock = use_mock
        if samples is None:
            self._load_samples(use_mock=use_mock)
        else:
            self.samples = list(samples)

    def _load_samples(self, use_mock: bool = False) -> None:
        """Scan clips; synthetic fixtures require explicit use_mock=True."""
        self.samples = []
        if use_mock:
            self.samples = [{"word": word, "label": self.word_to_idx[word], "file_path": None, "type": "mock"} for word in VOCABULARY]
            return
        seen = set()

        # 1. Search for preprocessed tensor clips if directory exists
        if self.preprocessed_dir and (self.preprocessed_dir / self.split).exists():
            tensor_split_dir = self.preprocessed_dir / self.split
            for word in VOCABULARY:
                word_dir = tensor_split_dir / word
                if word_dir.exists():
                    for pt_file in word_dir.glob("*.pt"):
                        self.samples.append({
                            "word": word,
                            "label": self.word_to_idx[word],
                            "file_path": pt_file,
                            "type": "tensor",
                        })
                        seen.add((word, pt_file.stem))

        # 2. Search for raw MP4 videos in root_dir/{split}/{word}/*.mp4
        if self.split_dir.exists():
            for word in VOCABULARY:
                word_dir = self.split_dir / word
                if word_dir.exists():
                    for mp4_file in word_dir.glob("*.mp4"):
                        if (word, mp4_file.stem) in seen:
                            continue
                        self.samples.append({
                            "word": word,
                            "label": self.word_to_idx[word],
                            "file_path": mp4_file,
                            "type": "video",
                        })

        # Missing data is an error; it must never become a synthetic training run.
        if not self.samples:
            raise FileNotFoundError("No LRW clips found. Fixture generation requires explicit use_mock=True.")

    def __len__(self) -> int:
        return len(self.samples)

    def _read_video_tensor(self, path: Path) -> torch.Tensor:
        """Decode MP4 video into mouth ROI grayscale tensor [T, 1, H, W]."""
        cap = cv2.VideoCapture(str(path))
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
                resized = cv2.resize(gray, (self.target_width, self.target_height))
                frames.append(resized)
        finally:
            cap.release()

        if not frames:
            raise ValueError(f"Cannot decode real video: {path}")

        arr = resample_temporal(frames, self.max_frames) / 255.0
        arr = arr[:, None, :, :]
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> LRWSample:
        sample_meta = self.samples[idx]
        word = str(sample_meta["word"])
        label = int(sample_meta["label"])
        file_path = sample_meta["file_path"]
        sample_type = sample_meta["type"]

        if sample_type == "tensor" and file_path and Path(file_path).exists():
            tensor = torch.load(file_path, map_location="cpu", weights_only=True)
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(1)
            elif tensor.ndim == 5:
                tensor = tensor.squeeze(0)
        elif sample_type == "video" and file_path and Path(file_path).exists():
            tensor = self._read_video_tensor(Path(file_path))
        elif sample_type == "mock" and self.use_mock:
            gen = torch.Generator().manual_seed(label + idx)
            tensor = torch.randn(
                self.max_frames, 1, self.target_height, self.target_width, generator=gen
            )
            tensor = tensor.sigmoid()
        else:
            raise FileNotFoundError(f"Training sample is missing: {file_path}")

        if tensor.shape[0] != self.max_frames:
            tensor = torch.from_numpy(resample_temporal(tensor.numpy(), self.max_frames))

        if self.augment:
            tensor = default_lrw_augmentation(tensor)

        if self.transform is not None:
            tensor = self.transform(tensor)

        return LRWSample(
            video=tensor,
            label=label,
            word=word,
            file_path=str(file_path) if file_path else None,
        )


def get_lrw_dataloader(
    root_dir: str | Path = "data/lrw",
    split: str = "train",
    batch_size: int = 2,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
    use_mock: bool = False,
    augment: Optional[bool] = None,
    **kwargs,
) -> Tuple[LRWDataset, DataLoader]:
    """Factory helper to create LRWDataset and DataLoader."""
    is_train = split.lower() == "train"
    do_augment = is_train if augment is None else augment
    dataset = LRWDataset(
        root_dir=root_dir,
        split=split,
        use_mock=use_mock,
        augment=do_augment,
        **kwargs,
    )
    do_shuffle = is_train if shuffle is None else shuffle
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=do_shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
    return dataset, loader
