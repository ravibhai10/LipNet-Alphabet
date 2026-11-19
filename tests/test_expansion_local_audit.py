import hashlib

import pytest

from scripts import audit_expansion_local as audit


def test_historical_test_speaker_cannot_become_training_speaker():
    values = [{'speaker_id': 'grid_T05', 'split': 'test', 'manifest': 'old'},
              {'speaker_id': 'grid_T05', 'split': 'train', 'manifest': 'new'},
              {'speaker_id': None, 'split': 'train', 'manifest': 'unknown'}]
    assert audit.role_conflicts(values) == {'grid_T05': {'test': ['old'], 'train': ['new']}}


def test_repeated_manifest_rows_are_not_speaker_leakage():
    row = {'speaker_id': 'F01', 'split': 'train', 'manifest': 'one'}
    assert audit.role_conflicts([row, row, {**row, 'manifest': 'two'}]) == {}


ANNOTATION = '0 1000 sil\n1000 2000 bin\n2000 3000 blue\n3000 4000 at\n4000 8000 a\n8000 11000 two\n11000 13000 now\n13000 14000 sil\n'


def letter():
    return {'fps': 25., 'alignment_timebase_hz': 25000, 'label': 'A',
            'start_units': 4000, 'end_units': 8000, 'start_frame': 4, 'end_frame': 8,
            'decoded_frames': 75, 'transcript': 'bin blue at a two now'}


def test_grid_units_are_not_milliseconds():
    assert audit.checked_letter(letter(), ANNOTATION) == 'A'
    with pytest.raises(ValueError, match='timestamp'):
        audit.checked_letter({**letter(), 'alignment_timebase_hz': 1000}, ANNOTATION)


@pytest.mark.parametrize('changed', [{'label': 'B'}, {'end_frame': 9}, {'decoded_frames': 7}])
def test_false_label_or_bad_frame_bounds_are_rejected(changed):
    with pytest.raises(ValueError):
        audit.checked_letter({**letter(), **changed}, ANNOTATION)


def test_unknown_w_is_not_accepted():
    with pytest.raises(ValueError, match='grammar'):
        audit.checked_letter({**letter(), 'label': 'W'}, ANNOTATION.replace('8000 a', '8000 w'))


def test_sequence_hash_verifies_order_bytes_and_source_root(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    directory = tmp_path / 'data/external/miracl/original/F01'
    directory.mkdir(parents=True)
    first, second = directory / 'color_001.jpg', directory / 'color_002.jpg'
    first.write_bytes(b'first source'); second.write_bytes(b'second source')
    before = audit.sequence_digest([first, second])
    assert before != audit.sequence_digest([second, first])
    second.write_bytes(b'changed source')
    assert before != audit.sequence_digest([first, second])
    with pytest.raises(ValueError, match='outside'):
        audit.sequence_digest([tmp_path / 'private.jpg'])


def test_namespaced_speaker_is_compared_without_inventing_unknown_identity():
    assert audit.miracl_speaker('miracl:F01') == 'F01'
    assert audit.miracl_speaker('F01') == 'F01'
    assert audit.miracl_speaker(None) is None
    assert audit.miracl_speaker('different:F01') == 'different:F01'


def test_historical_hash_recipe_is_explicit_and_preserves_raw_speaker():
    row = {'speaker_id': 'miracl:F01', 'source_sha256': 'example'}
    result = audit.exposure(row, 'runs/word_accuracy/avhubert_combined20_v1/features.jsonl')
    assert result['speaker_id'] == 'F01'
    assert result['original_speaker_id'] == 'miracl:F01'
    assert result['hash_recipe'] == 'miracl_ordered_basenames_binary_hashes'
    result = audit.exposure(row, 'runs/lrw500/available_samples_v1/manifest.jsonl')
    assert result['hash_recipe'] == 'lrw_json_ordered_file_hashes'
