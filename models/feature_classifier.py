"""Small supervised heads for a frozen pretrained temporal feature encoder."""
from __future__ import annotations
import torch
from torch import nn


class TemporalClassificationHead(nn.Module):
    def __init__(self, input_dim=768, num_classes=26, hidden_dim=0, viseme_classes=0):
        super().__init__()
        if min(input_dim, num_classes) < 1 or min(hidden_dim, viseme_classes) < 0:
            raise ValueError("Invalid classifier dimensions.")
        self.input_dim = input_dim
        self.encoder = (nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(.1))
                        if hidden_dim else nn.Identity())
        dimension = hidden_dim or input_dim
        self.classifier = nn.Linear(dimension, num_classes)
        self.viseme_head = nn.Linear(dimension, viseme_classes) if viseme_classes else None

    def pooled(self, features, lengths=None):
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError("Expected [batch,time,encoder_dim] features.")
        if lengths is None:
            return features.mean(1)
        lengths = torch.as_tensor(lengths, device=features.device)
        if lengths.shape != (features.shape[0],) or torch.any(lengths < 1) or torch.any(lengths > features.shape[1]):
            raise ValueError("Feature lengths must describe nonempty, unpadded sequences.")
        mask = torch.arange(features.shape[1], device=features.device)[None, :] < lengths[:, None]
        return (features * mask[..., None]).sum(1) / lengths[:, None]

    def forward_with_aux(self, features, lengths=None):
        encoded = self.encoder(self.pooled(features, lengths))
        return {"logits": self.classifier(encoded),
                "viseme_logits": self.viseme_head(encoded) if self.viseme_head is not None else None}

    def forward(self, features, lengths=None):
        return self.forward_with_aux(features, lengths)["logits"]
