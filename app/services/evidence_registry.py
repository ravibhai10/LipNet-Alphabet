"""Artifact-bound historical evidence. Reading a report never creates a new result."""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def fingerprint(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def resolve_source(value, manifest, root=ROOT):
    """Relocate legacy developer paths without editing historical manifests."""
    text = str(value).replace('\\', '/')
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = Path(manifest).parent / candidate
        if not candidate.is_file():
            candidate = root / text
    elif not candidate.is_file():
        for marker in ('/data/', '/runs/', '/models/'):
            if marker in text:
                candidate = root / text.split(marker, 1)[1]
                candidate = root / marker.strip('/') / text.split(marker, 1)[1]
                break
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('Source path escapes the configured repository root.')
    return resolved


def dataset_checks(path, root=ROOT):
    from data.validate import ValidationReport, ValidationResult
    path = Path(path); report = ValidationReport(results=[])
    def add(name, passed, message, severity='error'):
        report.add_result(ValidationResult(name, passed, message, severity))
    if not path.is_file():
        add('manifest_exists', False, 'Manifest unavailable; no data verification was performed.')
        return report
    try:
        data = rows(path)
        if not data or any(not isinstance(row, dict) for row in data):
            raise ValueError('Manifest must contain nonempty object records.')
    except (OSError, ValueError):
        add('manifest_entries', False, 'Manifest missing, empty or malformed.')
        return report
    add('manifest_entries', True, f'{len(data)} records parsed; not a media-decode result.')
    ids = [r.get('id', r.get('recording_id')) for r in data]
    add('duplicate_recording_ids', None not in ids and len(set(ids)) == len(ids), 'Record identifiers checked.')
    missing = invalid = 0; speakers = defaultdict(set); sources = defaultdict(set); labels = Counter()
    for row in data:
        split = row.get('split'); speaker = row.get('speaker_id', row.get('speaker'))
        if speaker is not None:
            speakers[str(speaker)].add(split)
        digest = row.get('video_sha256', row.get('source_sha256'))
        if digest:
            sources[digest].add(split)
        label = row.get('label', row.get('letter', row.get('word_label')))
        if label is not None:
            labels[str(label)] += 1
        paths = row.get('frames') or [row.get('video', row.get('file_path'))]
        for source in paths:
            if source is None:
                invalid += 1; continue
            try:
                missing += not resolve_source(source, path, root).is_file()
            except ValueError:
                invalid += 1
        if 'start_frame' in row and 'end_frame' in row:
            invalid += not (isinstance(row['start_frame'], int) and isinstance(row['end_frame'], int) and
                            0 <= row['start_frame'] < row['end_frame'])
    add('missing_files', missing == 0 and invalid == 0, f'{missing} missing paths; {invalid} invalid paths/bounds.')
    add('train_test_leakage', all(len(s) == 1 for s in speakers.values()), 'Declared participant assignments checked.')
    add('source_leakage', all(len(s) == 1 for s in sources.values()), 'Declared source-hash groups checked; bytes not rehashed.')
    add('letters', set(labels) >= set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
        'Missing A-Z labels: ' + ','.join(sorted(set('ABCDEFGHIJKLMNOPQRSTUVWXYZ') - set(labels))), 'warning')
    add('video_integrity', False, 'Not measured: structural validation does not decode media.', 'warning')
    return report


SPECS = {
    'avhubert_combined20_v1': ('runs/word_accuracy/avhubert_combined20_v1/evaluation.json',
                               'runs/word_accuracy/avhubert_combined20_v1/head.pt',
                               'Frozen AV-HuBERT layer 9 + mean/std StandardScaler + 20-word supervised classifier', 'isolated_word'),
    'avhubert_miracl8_v1': ('runs/word_accuracy/avhubert_miracl8_v1/evaluation.json',
                           'runs/word_accuracy/avhubert_miracl8_v1/head.pt',
                           'Frozen AV-HuBERT layer 9 + mean/std StandardScaler + logistic regression', 'isolated_word'),
    'word_recognition_v1': ('runs/word_recognition_v1/evaluation_summary.json',
                            'runs/word_recognition_v1/checkpoints/best_word_model.pt',
                            'Three Conv3D stages + two-layer bidirectional GRU + linear classifier', 'isolated_word'),
    'alphabet_cnn_bilstm': ('models/training_summary.json', 'models/alphabet_classifier.pt',
                           'Three framewise Conv2D stages + one-layer BiLSTM + linear classifier', 'isolated_letter'),
}


def evaluation(identifier, root=ROOT):
    if identifier not in SPECS:
        raise LookupError('Unknown evaluation ID.')
    report_name, checkpoint_name, architecture, task = SPECS[identifier]
    report, checkpoint = root / report_name, root / checkpoint_name
    value = {'id': identifier, 'name': identifier, 'architecture': architecture, 'task': task,
             'status': 'unavailable', 'scope': 'historical_offline', 'webcam_accuracy_validated': False,
             'checkpoint_path': checkpoint_name, 'evaluated_at': None, 'test_samples': None,
             'top1_accuracy': None, 'macro_f1': None, 'top3': None, 'top5_accuracy': None,
             'source_artifacts': [], 'limitations': []}
    if not report.is_file() or not checkpoint.is_file():
        return {**value, 'reason': 'Evaluation report or checkpoint unavailable.'}
    try:
        data = json.loads(report.read_text(encoding='utf-8'))
        actual = fingerprint(checkpoint)
        expected = data.get('head_sha256', data.get('checkpoint_sha256'))
        value['source_artifacts'] = [{'path': report_name, 'sha256': fingerprint(report)},
                                     {'path': checkpoint_name, 'sha256': actual}]
        if expected and expected != actual:
            return {**value, 'reason': 'Report/checkpoint integrity mismatch.'}
        if not expected:
            return {**value, 'status': 'unbound_historical_report',
                    'reason': 'Historical report has no checkpoint fingerprint; metrics are not an authenticated model evaluation.'}
        value.update(status='available', checkpoint_sha256=actual, vocabulary=data.get('vocabulary'),
                     test_samples=data.get('test_samples', data.get('total_samples')),
                     correct=data.get('correct'), top1_accuracy=data.get('top1_accuracy', data.get('word_accuracy')),
                     macro_f1=data.get('macro_f1', data.get('classification_report', {}).get('macro avg', {}).get('f1-score')),
                     top5_accuracy=data.get('top5_accuracy'), wer=data.get('single_word_wer', data.get('wer')),
                     per_word=data.get('per_word'), confusion_matrix=data.get('confusion_matrix'),
                     evaluated_at=data.get('evaluated_at'), rejection_policy=data.get('rejection_policy'))
        if identifier in ('avhubert_miracl8_v1', 'avhubert_combined20_v1'):
            feature_manifest = root / f'runs/word_accuracy/{identifier}/features.jsonl'
            if feature_manifest.is_file():
                records = rows(feature_manifest)
                spks = sorted({str(r['speaker_id']) for r in records if r.get('split') == 'test' and r.get('role', 'known_word') == 'known_word'})
                value['speakers'] = spks
                value['speaker_protocol'] = 'Held-out test partition: ' + (', '.join(spks) if spks else 'speaker/clip disjoint')
                value['limitations'] = [data.get('test_policy', 'Held-out test evaluation.'), 'Unknown isolated-word and physical-webcam performance unverified.']
                value['source_artifacts'].append({'path': str(feature_manifest.relative_to(root)), 'sha256': fingerprint(feature_manifest)})
        if identifier == 'avhubert_miracl8_v1':
            fresh = root / '.qa/reliability_20261109/final_frozen_evaluation.json'
            if fresh.is_file():
                part = json.loads(fresh.read_text())['partitions']['test']
                if part['head_sha256'] == actual:
                    rejection = part['after']
                    value['rejection_metrics'] = {**rejection, 'coverage': rejection['accepted'] / rejection['known_samples'],
                                                  'raw_accuracy': part['top1_accuracy'], 'scope': 'historical_cached_feature_replay'}
                    value['source_artifacts'].append({'path': str(fresh.relative_to(root)), 'sha256': fingerprint(fresh)})
        return value
    except (OSError, ValueError, KeyError, TypeError):
        return {**value, 'status': 'unavailable', 'reason': 'Evidence is malformed or incomplete.'}


def evaluations(root=ROOT):
    return {'schema_version': 2, 'evaluations': [evaluation(name, root) for name in SPECS],
            'archived_diagnostics': [{'id': 'continuous_sequence_v2t', 'status': 'diagnostic_only',
                                     'reason': 'Constant-hypothesis legacy diagnostic; no verified sentence decoder.'}]}
