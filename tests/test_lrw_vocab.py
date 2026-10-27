"""tests/test_lrw_vocab.py - LRW 500-Word Vocabulary & Dataset Regression Tests."""

from pathlib import Path
import pytest
import torch

from data.schema import (
    VOCABULARY,
    NUM_CLASSES,
    WORD_TO_IDX,
    IDX_TO_WORD,
    get_word_idx,
    get_word,
    is_valid_word,
    get_vocabulary,
    get_num_classes,
    letter_to_idx,
    idx_to_letter,
)
from pipeline.lrw.loader import LRWDataset, get_lrw_dataloader


def test_vocabulary_size():
    """Assert vocabulary contains exactly 500 unique non-empty words."""
    assert len(VOCABULARY) == 500, f"Expected 500 words, got {len(VOCABULARY)}"
    assert NUM_CLASSES == 500
    assert get_num_classes() == 500
    assert len(get_vocabulary()) == 500
    assert len(set(VOCABULARY)) == 500


def test_bidirectional_mapping():
    """Verify bidirectional index-to-word and word-to-index mappings for key sample words."""
    sample_words = ["ABOUT", "ACTION", "YOUNG", "YESTERDAY", "WEATHER", "WORDS"]
    for word in sample_words:
        assert word in WORD_TO_IDX, f"{word} missing from WORD_TO_IDX"
        idx = get_word_idx(word)
        assert isinstance(idx, int)
        assert 0 <= idx < 500

        recovered = get_word(idx)
        assert recovered == word
        assert IDX_TO_WORD[idx] == word
        assert is_valid_word(word)

    # Case-insensitivity support for get_word_idx
    assert get_word_idx("about") == get_word_idx("ABOUT")
    assert get_word_idx("Young") == get_word_idx("YOUNG")

    # Safety checks for out-of-vocabulary word
    with pytest.raises(ValueError, match="not in vocabulary"):
        get_word_idx("NONEXISTENT_WORD_XYZ")

    # Safety checks for out-of-bounds indices
    with pytest.raises(IndexError):
        get_word(-1)

    with pytest.raises(IndexError):
        get_word(500)


def test_backward_compatibility_alphabet():
    """Ensure legacy single-letter A-Z alphabet mappings are fully preserved."""
    assert letter_to_idx("A") == 0
    assert letter_to_idx("B") == 1
    assert letter_to_idx("Z") == 25
    assert letter_to_idx("a") == 0

    assert idx_to_letter(0) == "A"
    assert idx_to_letter(1) == "B"
    assert idx_to_letter(25) == "Z"
    assert idx_to_letter(999) == "?"


def test_lrw_dataset_instantiation_and_dimensions():
    """Verify LRWDataset instantiates cleanly and yields expected tensor dimensions [T, 1, 140, 46]."""
    dataset = LRWDataset(split="train")
    assert len(dataset) > 0, "LRWDataset should yield at least 1 sample"

    # Test indexing
    sample = dataset[0]
    assert "video" in sample
    assert "label" in sample
    assert "word" in sample

    # Test tuple unpacking support (x, y = dataset[0])
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, int)

    # Validate tensor shape: [T x 1 x 140 x 46]
    assert x.ndim == 4, f"Expected 4D tensor [T, C, H, W], got {x.ndim}D with shape {x.shape}"
    assert x.shape[0] == 29, f"Expected 29 frames, got {x.shape[0]}"
    assert x.shape[1] == 1, f"Expected 1 channel (grayscale), got {x.shape[1]}"
    assert x.shape[2] == 140, f"Expected height 140, got {x.shape[2]}"
    assert x.shape[3] == 46, f"Expected width 46, got {x.shape[3]}"
    assert 0 <= y < 500, f"Label {y} out of range [0, 500)"


def test_lrw_dataloader():
    """Verify DataLoader batches LRW samples correctly for CPU execution."""
    dataset, loader = get_lrw_dataloader(split="train", batch_size=2, num_workers=0)
    assert len(loader) > 0

    batch = next(iter(loader))
    # Batch is collated as (videos, labels)
    videos, labels = batch
    assert isinstance(videos, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    # Batch shape: [B, T, 1, H, W]
    assert videos.shape == torch.Size([2, 29, 1, 140, 46])
    assert labels.shape == torch.Size([2])
