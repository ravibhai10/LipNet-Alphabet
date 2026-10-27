import torch
import pytest

from src.cpu_utils import configure_cpu_runtime, set_threads


def test_set_threads_updates_module_setting():
    previous = torch.get_num_threads()
    try:
        assert set_threads(1) == 1
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(previous)


def test_set_threads_rejects_invalid_values():
    with pytest.raises(ValueError):
        set_threads(0)


def test_configure_cpu_runtime_forces_cpu():
    result = configure_cpu_runtime(threads=1)
    assert result["device"] == torch.device("cpu")
    assert result["cuda_available"] is False
