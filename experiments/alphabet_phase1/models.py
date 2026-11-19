"""Independent 26-output models. No serving registration or trained weights.

All candidates accept [B,30,1,140,46] in [0,1]. Masks are right-padded
valid-frame prefixes. Missing W data blocks full A-Z training, not the shape.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

LABELS = tuple(chr(65 + index) for index in range(26))
INPUT_SHAPE = (30, 1, 140, 46)


def validate(x, mask=None):
    if x.ndim != 5 or tuple(x.shape[1:]) != INPUT_SHAPE or not x.is_floating_point():
        raise ValueError('Expected floating [B,30,1,140,46]')
    if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError('Finite pixels in [0,1] required')
    if mask is None:
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    if mask.shape != x.shape[:2] or mask.dtype != torch.bool or mask.device != x.device:
        raise ValueError('Expected bool [B,30] mask on the input device')
    if not mask.any(dim=1).all() or ((~mask[:, :-1]) & mask[:, 1:]).any():
        raise ValueError('Each mask must be a nonempty right-padded valid prefix')
    return mask


class SpatialCNN(nn.Module):
    """Retain spatial grids until all three convolutions have run.

140x46 -> 70x23 -> 35x12 -> 18x6 -> 2x2; unlike the protected model,
there is no 1x1 bottleneck followed by repeated spatial upsampling.
GroupNorm avoids batch statistics from tiny or padded clip batches.
"""
    def __init__(self):
        super().__init__()
        blocks, channels = [], 1
        for output in (16, 32, 64):
            blocks.extend([nn.Conv2d(channels, output, 3, stride=2, padding=1, bias=False),
                           nn.GroupNorm(4, output), nn.ReLU()])
            channels = output
        self.layers = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())
        self.output_size = 256

    def forward(self, x, mask):
        batch, length = x.shape[:2]
        flat = x.reshape(batch * length, *x.shape[2:])
        valid = mask.reshape(-1)
        # Invalid frames never affect convolution/normalization or valid states.
        features = x.new_zeros((batch * length, self.output_size))
        features[valid] = self.layers(flat[valid])
        return features.reshape(batch, length, -1)


class ImprovedBiLSTM(nn.Module):
    def __init__(self, attention=False):
        super().__init__()
        self.spatial = SpatialCNN()
        self.temporal = nn.LSTM(256, 96, bidirectional=True, batch_first=True)
        self.attention = (nn.Sequential(nn.Linear(192, 64), nn.Tanh(), nn.Linear(64, 1))
                          if attention else None)
        self.classifier = nn.Sequential(nn.Linear(192, 128), nn.ReLU(), nn.Dropout(.3), nn.Linear(128, 26))

    def forward(self, x, mask=None, *, return_attention=False):
        mask = validate(x, mask)
        features = self.spatial(x, mask)
        packed = pack_padded_sequence(features, mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
        states, (hidden, _) = self.temporal(packed)
        weights = None
        if self.attention is None:
            # Both directions summarize their entire valid sequence.
            pooled = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        else:
            states, _ = pad_packed_sequence(states, batch_first=True, total_length=x.shape[1])
            scores = self.attention(states).squeeze(-1).masked_fill(~mask, float('-inf'))
            weights = scores.softmax(dim=1)
            pooled = (states * weights.unsqueeze(-1)).sum(1)
        logits = self.classifier(pooled)
        return (logits, weights) if return_attention else logits


class TemporalBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(96, 96, 3, padding=dilation, dilation=dilation) for _ in range(2)])
        self.dropout = nn.Dropout(.3)

    def forward(self, x, mask):
        residual = x
        for conv in self.convs:
            x = self.dropout(torch.relu(conv(x))) * mask
        return torch.relu(x + residual) * mask


class CNNTemporalConv(nn.Module):
    """Noncausal utterance TCN, receptive field 61; not a streaming model."""
    def __init__(self):
        super().__init__()
        self.spatial = SpatialCNN()
        self.project = nn.Conv1d(256, 96, 1)
        self.blocks = nn.ModuleList([TemporalBlock(d) for d in (1, 2, 4, 8)])
        self.classifier = nn.Linear(96, 26)

    def forward(self, x, mask=None):
        mask = validate(x, mask)
        weights = mask[:, None].to(x.dtype)
        states = self.project(self.spatial(x, mask).transpose(1, 2)) * weights
        for block in self.blocks:
            states = block(states, weights)
        return self.classifier(states.sum(2) / weights.sum(2))


class ExistingReference(nn.Module):
    """Reuse the exact protected definition; state dictionaries stay native."""
    def __init__(self):
        super().__init__()
        from models.alphabet_model import AlphabetCNNBiLSTM
        self.model = AlphabetCNNBiLSTM()

    def forward(self, x, mask=None):
        mask = validate(x, mask)
        if not mask.all():
            raise ValueError('Protected reference has no masked-sequence contract')
        return self.model(x)


def build(name):
    if name == 'reference':
        return ExistingReference()
    if name == 'bilstm':
        return ImprovedBiLSTM()
    if name == 'tcn':
        return CNNTemporalConv()
    if name == 'attention':
        return ImprovedBiLSTM(attention=True)
    raise ValueError('Unknown experimental architecture')
