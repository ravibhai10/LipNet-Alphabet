from pathlib import Path

import pytest

from scripts.train_alphabet import split_by_speaker


def test_split_by_speaker_is_disjoint_and_deterministic():
    samples = [
        (Path(f"{speaker}_{index}.npy"), 0, speaker)
        for speaker in ("s1", "s2", "s3", "s4", "s5")
        for index in range(2)
    ]
    first = split_by_speaker(samples, seed=42)
    second = split_by_speaker(samples, seed=42)
    assert first == second
    owners = {}
    for split, records in zip(("train", "val", "test"), first):
        for path, _ in records:
            speaker = path.name.split("_")[0]
            if speaker in owners:
                assert owners[speaker] == split
            else:
                owners[speaker] = split
    assert set(owners.values()) == {"train", "val", "test"}


def test_split_by_speaker_requires_three_speakers():
    samples = [(Path("one.npy"), 0, "s1"), (Path("two.npy"), 0, "s2")]
    with pytest.raises(ValueError, match="three speakers"):
        split_by_speaker(samples)
