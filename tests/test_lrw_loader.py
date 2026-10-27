#!/usr/bin/env python
"""pytest test for LRWDataset"""
import sys
sys.path.insert(0, ".")
import pytest
from pipeline.lrw.loader import LRWDataset


def test_lrwdataset_basic():
    """Test basic LRWDataset functionality with mock data."""
    dataset = LRWDataset(use_mock=True)
    assert len(dataset) > 0
    
    sample = dataset[0]
    assert hasattr(sample, 'video')
    assert hasattr(sample, 'label')
    assert hasattr(sample, 'word')
    assert hasattr(sample, 'file_path')
    
    # Verify dimensions
    assert sample.video.shape == (29, 1, 140, 46)
    assert 0 <= sample.label < 500


def test_lrwdataset_with_mock():
    """Test LRWDataset with mock data for all splits."""
    for split in ["train", "valid", "test"]:
        dataset = LRWDataset(split=split, use_mock=True)
        sample = dataset[0]
        assert sample.video.shape == (29, 1, 140, 46)
        assert 0 <= sample.label < 500