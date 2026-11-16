"""Gradient inspection and smoke testing utilities."""
from __future__ import annotations
import math
import torch


def gradient_report(model: torch.nn.Module) -> dict:
    total_sq = 0.0
    stats = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            norm = p.grad.data.norm(2).item()
            total_sq += norm ** 2
            stats[name] = {"norm": norm}

    total_norm = math.sqrt(total_sq)
    return {
        "total_l2_norm": float(total_norm),
        "parameter_count": len(stats),
        "parameters": stats
    }
