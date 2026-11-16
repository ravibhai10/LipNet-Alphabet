"""PyTorch Dataset, caching, and data augmentation for word-level lip reading."""
from __future__ import annotations
import random
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset
from pipeline.word_recognition.preprocess import read_frames


def augment(tensor: torch.Tensor) -> torch.Tensor:
    """Apply non-destructive runtime spatial roll and contrast augmentations to [1, 24, 64, 64] tensor."""
    result = tensor.clone()
    _, T, H, W = result.shape

    # 1. Safe spatial circular shift (+-1 pixel): avoids zero-padded black boundary artifacts
    if random.random() < 0.5:
        dy = random.randint(-1, 1)
        dx = random.randint(-1, 1)
        if dx != 0 or dy != 0:
            result = torch.roll(result, shifts=(dy, dx), dims=(2, 3))

    # 2. Subtle contrast scaling (+-5%): preserves lip-viseme opening/closing motion
    if random.random() < 0.5:
        scale = 1.0 + (random.random() - 0.5) * 0.10
        result = result * scale

    return result.clamp(0.0, 1.0)


class WordDataset(Dataset):
    def __init__(self, run, rows, config, training: bool = False):
        self.run = run
        self.rows = rows
        self.config = config
        self.training = training
        if self.training:
            for row in rows:
                if row.get("split") != "train":
                    raise ValueError("All rows in a training WordDataset must belong to the training partition.")
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.max_cache=128  # uint8 clips: approximately 12 MiB, independent of corpus size.

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        row_id = row["id"]

        if row_id not in self.cache:
            clip = read_frames(self.run, row, self.config) # [24, 64, 64] uint8
            self.cache[row_id] = torch.from_numpy(clip)
            if len(self.cache)>self.max_cache:self.cache.popitem(last=False)
        self.cache.move_to_end(row_id)

        cached_uint8 = self.cache[row_id]
        tensor = cached_uint8.float().unsqueeze(0) / 255.0 # [1, 24, 64, 64] in [0, 1]

        if self.training:
            tensor = augment(tensor)

        label = row["label"]
        return tensor, label, row_id
