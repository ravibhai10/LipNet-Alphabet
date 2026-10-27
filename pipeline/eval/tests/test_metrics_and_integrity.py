from dataclasses import dataclass

import pytest

from pipeline.eval.integrity import validate_speaker_splits
from pipeline.eval.metrics import corpus_cer, corpus_wer


@dataclass
class Entry:
    recording_id: str
    speaker_id: str
    split: str


def test_corpus_wer_aggregates_edit_counts():
    result = corpus_wer(["bin blue", "at f"], ["bin green", "at"])
    assert result.errors == 2
    assert result.reference_units == 4
    assert result.rate == pytest.approx(0.5)
    assert result.samples == 2


def test_corpus_cer_handles_empty_reference():
    result = corpus_cer([""], ["prediction"])
    assert result.errors == len("prediction")
    assert result.reference_units == 0
    assert result.rate == 1.0


def test_speaker_split_integrity_detects_overlap_duplicates_and_missing():
    report = validate_speaker_splits([
        Entry("one", "s1", "train"),
        Entry("two", "s1", "val"),
        Entry("two", "s2", "test"),
        Entry("three", "s3", "unassigned"),
    ])
    assert report.overlaps == {"s1": ("train", "val")}
    assert report.duplicate_sample_ids == ("two",)
    assert report.missing_split_sample_ids == ("three",)
    assert report.valid is False


def test_speaker_split_integrity_accepts_disjoint_entries():
    report = validate_speaker_splits([
        Entry("one", "s1", "train"),
        Entry("two", "s2", "val"),
        Entry("three", "s3", "test"),
    ])
    assert report.valid is True
