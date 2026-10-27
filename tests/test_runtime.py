import pytest
import torch

from alphabet.runtime import configure_cpu_runtime


def test_cpu_runtime_applies_bounded_thread_count():
    previous = torch.get_num_threads()
    runtime = configure_cpu_runtime(threads=1)
    try:
        assert runtime.threads == 1
        assert runtime.interop_threads >= 1
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(previous)


def test_cpu_runtime_rejects_invalid_thread_count():
    with pytest.raises(ValueError, match="positive"):
        configure_cpu_runtime(threads=0)
