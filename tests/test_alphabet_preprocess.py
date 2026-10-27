import numpy as np
import pytest

from scripts.preprocess_alphabet_clips import temporal_resample


def test_temporal_resample_produces_classifier_length():
    frames = [np.full((140, 46), index, dtype=np.uint8) for index in range(4)]
    clip = temporal_resample(frames, target=30)
    assert clip.shape == (30, 140, 46)
    assert clip.dtype == np.uint8
    assert int(clip[0, 0, 0]) == 0
    assert int(clip[-1, 0, 0]) == 3


def test_temporal_resample_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        temporal_resample([])
