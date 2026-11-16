"""New small 3D-CNN + two-layer bidirectional GRU word classifier.

This is a separately trained research model; it does not load or replace any
alphabet/LRW/AV-HuBERT checkpoint. Input [B,1,T,64,64], output [B,words].
"""
from __future__ import annotations
import torch
from torch import nn


class WordVideoNet(nn.Module):
    def __init__(self, num_classes, channels=(8, 16, 32), hidden_size=48, layers=2, dropout=.15):
        super().__init__()
        if num_classes < 2 or len(channels) != 3 or any(c < 1 for c in channels) or hidden_size < 1 or layers < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid word model configuration.")
        blocks, previous = [], 1
        for channels_out in channels:
            blocks.append(nn.Sequential(nn.Conv3d(previous, channels_out, 3, stride=(1, 2, 2), padding=1, bias=False),
                                        nn.GroupNorm(1, channels_out), nn.SiLU()))
            previous = channels_out
        self.frontend = nn.ModuleList(blocks)
        self.temporal = nn.GRU(previous, hidden_size, layers, batch_first=True, bidirectional=True,
                               dropout=dropout if layers > 1 else 0.)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size * 2, num_classes))

    def forward(self, x, return_shapes=False):
        if x.ndim != 5 or tuple(x.shape[1:]) != (1, 24, 64, 64):
            raise ValueError("Expected [B,1,24,64,64] real mouth video.")
        shapes = {"input": list(x.shape)}
        for i, block in enumerate(self.frontend):
            x = block(x)
            shapes[f"conv3d_{i+1}"] = list(x.shape)
        x = x.mean(dim=(-1, -2)).transpose(1, 2)
        shapes["frame_features"] = list(x.shape)
        x, _ = self.temporal(x)
        shapes["bidirectional_gru"] = list(x.shape)
        result = self.classifier(x.mean(dim=1))
        shapes["logits"] = list(result.shape)
        return (result, shapes) if return_shapes else result
