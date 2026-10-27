# cpu_utils.py - CPU optimization utilities for LipNet-Alphabet
"""
CPU-only optimization utilities for the LipNet-Alphabet project.
Provides device management, thread configuration, memory monitoring,
and batch size calculation for CPU-only environments.
"""

import os
import torch
from typing import Optional


# ---------------------------------------------------------------------------
# Device configuration (CPU-only enforcement)
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")
"""Default device - always CPU for this environment."""

HAS_CUDA = False


# Optimal thread count based on physical CPU cores
TORCH_NUM_THREADS = min(os.cpu_count() or 1, 4)
"""Limit torch threads to prevent oversubscription on CPU."""


def set_threads(num: Optional[int] = None) -> int:
    """
    Set the number of threads for PyTorch operations.
    
    Args:
        num: Number of threads. If None, uses min(cpu_count, 4).
    
    Returns:
        The number of threads set.
    """
    global TORCH_NUM_THREADS
    if num is not None:
        if not isinstance(num, int) or num <= 0:
            raise ValueError("num must be a positive integer")
        TORCH_NUM_THREADS = max(1, min(num, os.cpu_count() or 1))
    torch.set_num_threads(TORCH_NUM_THREADS)
    return TORCH_NUM_THREADS


def get_device() -> torch.device:
    """Return the current computation device (always CPU in this config)."""
    return DEVICE


def is_cuda_available() -> bool:
    """Return whether CUDA is enabled for this CPU-only utility."""
    return HAS_CUDA


def configure_cpu_runtime(threads: int = 4) -> dict:
    """Configure bounded CPU execution without probing or selecting CUDA."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    configured = set_threads(threads)
    return {
        "device": torch.device("cpu"),
        "threads": configured,
        "cuda_available": False,
    }


def optimal_batch_size(
    model_params: int,
    target_memory_mb: int = 500,
    max_tokens: int = 4000
) -> int:
    """
    Calculate optimal batch size for CPU based on model size and memory.
    
    Args:
        model_params: Number of model parameters.
        target_memory_mb: Target memory per batch in MB (default 500MB for CPU).
        max_tokens: Maximum tokens per fairseq batch (default 4000 for CPU).
    
    Returns:
        Recommended batch size (1-8 typical for CPU).
    """
    # Rough estimate: each token ~ 100 bytes activation + 4 bytes grad
    # Plus model params overhead
    estimated_per_token = 100 + (model_params / 1_000_000) * 4  # MB per token
    max_by_memory = target_memory_mb // max(1, estimated_per_token)
    max_by_tokens = max_tokens  # fairseq limit
    
    batch_size = min(max_by_memory, max_by_tokens, 8)  # Cap at 8 for CPU
    return max(1, batch_size)


def log_environment_info() -> dict:
    """
    Log runtime environment information for reproducibility.
    
    Returns:
        Dictionary with environment specs.
    """
    import datetime
    return {
        "python_version": os.sys.version,
        "torch_version": torch.__version__,
        "device": str(DEVICE),
        "num_cpus": os.cpu_count() or 1,
        "torch_num_threads": TORCH_NUM_THREADS,
        "timestamp": datetime.datetime.now().isoformat(),
        "has_cuda": HAS_CUDA,
    }


# ---------------------------------------------------------------------------
# Memory management helpers
# ---------------------------------------------------------------------------
def empty_cache():
    """Clear PyTorch CPU cache (no-op on CPU but kept for compatibility)."""
    # On CPU, this is primarily for consistency with CUDA code paths
    # and to force garbage collection of intermediate tensors
    import gc
    gc.collect()


def monitor_memory(stage: str = "unknown") -> dict:
    """
    Get current memory usage statistics.
    
    Args:
        stage: Name of the pipeline stage being monitored.
    
    Returns:
        Dictionary with memory stats in MB.
    """
    try:
        # CPU doesn't have the same memory model as CUDA
        # but we can track tensor allocations
        allocated = torch.cpu.memory_reserved() / 1_024_000 if hasattr(torch.cpu, 'memory_reserved') else 0
        cached = torch.cpu.memory_cached() / 1_024_000 if hasattr(torch.cpu, 'memory_cached') else 0
    except Exception:
        allocated = 0
        cached = 0
    
    return {
        "stage": stage,
        "allocated_mb": round(allocated, 2),
        "cached_mb": round(cached, 2),
    }