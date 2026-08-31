"""
Alphabet Lip-Reading Model — CNN-BiLSTM classifier (PyTorch 2.x).
NOTE: TensorFlow is unavailable for Python 3.14; this model uses PyTorch
with an identical architecture to the original TensorFlow 2.x specification.
"""

from __future__ import annotations

import logging
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

T, H, W = 30, 140, 46   # frames, height, width per GRID clip
NUM_CLASSES = 26          # A-Z
C1, C2, C3 = 16, 32, 64   # smaller channels — appropriate for 25-class, 800-sample train set
LSTM_HIDDEN = 96
LSTM_LAYERS = 1


class AlphabetCNNBiLSTM(nn.Module):
    """CNN-BiLSTM lip-reader. Input: (B, T, 1, H, W). Output: (B, 26)."""

    def __init__(self,
                 num_classes: int = NUM_CLASSES,
                 c1: int = C1, c2: int = C2, c3: int = C3,
                 lstm_hidden: int = LSTM_HIDDEN,
                 lstm_layers: int = LSTM_LAYERS,
                 dropout: float = 0.5) -> None:
        super().__init__()
        self.t, self.h, self.w = T, H, W
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1);   self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1); self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1); self.bn3 = nn.BatchNorm2d(c3)
        self.pool  = nn.AdaptiveAvgPool2d((4, 4))
        frame_feat = c3 * 4 * 4
        self.lstm  = nn.LSTM(frame_feat, lstm_hidden, lstm_layers,
                              batch_first=True, bidirectional=True,
                              dropout=dropout if lstm_layers > 1 else 0.0)
        self.drop  = nn.Dropout(dropout)
        self.fc1   = nn.Linear(lstm_hidden * 2, 128)
        self.fc2   = nn.Linear(128, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T_, C, H_, W_ = x.shape
        assert C == 1 and H_ == self.h and W_ == self.w
        x = x.view(B * T_, 1, H_, W_)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
        x = self.pool(x).view(B * T_, -1)
        x = x.view(B, T_, -1)
        x, _ = self.lstm(x)
        x = self.drop(F.relu(self.fc1(x[:, -1, :])))
        return self.fc2(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        return (f"AlphabetCNNBiLSTM\n"
                f"  Input:     (B, {self.t}, 1, {self.h}, {self.w})\n"
                f"  Output:    (B, {NUM_CLASSES})\n"
                f"  Params:    {self.num_parameters():,}\n"
                f"  Conv:      {C1}→{C2}→{C3}, AdaptivePool→(4,4)\n"
                f"  BiLSTM:    {LSTM_LAYERS}×{LSTM_HIDDEN}\n"
                f"  Head:      {LSTM_HIDDEN*2}→128→{NUM_CLASSES}")


if __name__ == "__main__":
    m = AlphabetCNNBiLSTM()
    print(m.summary())
    x = torch.randn(4, T, 1, H, W)
    print("Forward:", m(x).shape)
