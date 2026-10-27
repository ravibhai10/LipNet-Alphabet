"""Runtime controls for reproducible CPU execution."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CpuRuntime:
    """Effective CPU settings applied to a process."""

    threads: int
    interop_threads: int
    deterministic: bool


def configure_cpu_runtime(
    *,
    threads: int | None = None,
    deterministic: bool = False,
) -> CpuRuntime:
    """Apply bounded CPU settings without touching CUDA or global env files.

    ``LIPNET_TORCH_THREADS`` can override the default cap of four threads.
    Inter-op threads are set only when PyTorch still permits changing them.
    """
    requested = threads
    if requested is None:
        requested = int(os.environ.get("LIPNET_TORCH_THREADS", "4"))
    if requested <= 0:
        raise ValueError("CPU thread count must be positive")
    effective = min(requested, os.cpu_count() or 1)
    torch.set_num_threads(effective)
    interop = min(2, effective)
    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        # PyTorch disallows changing inter-op threads after parallel work starts.
        interop = torch.get_num_interop_threads()
    if deterministic:
        torch.use_deterministic_algorithms(True)
    return CpuRuntime(effective, interop, deterministic)
