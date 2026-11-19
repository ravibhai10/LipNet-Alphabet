"""Read-only GRID/MIRACL audit. Writes exclusively to a new report directory.

No dataset download, crop extraction, split publication or training is performed.
Historical held-out roles are compared across versions, not merely within one file.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import re
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def role_conflicts(records):
    groups = defaultdict(lambda: defaultdict(set))
    for row in records:
        speaker = row.get('speaker_id')
        if speaker and row.get('split') in {'train', 'valid', 'test'}:
            groups[speaker][row['split']].add(row['manifest'])
    return {speaker: {role: sorted(paths) for role, paths in roles.items()}
            for speaker, roles in groups.items() if len(roles) > 1}


def sequence_digest(frames):
    """Existing MIRACL contract: ordered basenames followed by binary file hashes."""
    value = hashlib.sha256()
    content_only = hashlib.sha256()
    frame_hashes = []
    for name in frames:
        path = Path(name)
        if not path.resolve().is_relative_to((ROOT / 'data/external/miracl/original').resolve()):
            raise ValueError('MIRACL frame outside eligible local source')
        digest = bytes.fromhex(sha256(path))
        value.update(path.name.encode())
        value.update(digest)
        content_only.update(digest)
        frame_hashes.append(digest.hex())
    lrw_digest = (frame_hashes[0] if len(frame_hashes) == 1 else
                  hashlib.sha256(json.dumps(frame_hashes, separators=(',', ':')).encode()).hexdigest())
    return value.hexdigest(), content_only.hexdigest(), lrw_digest


def miracl_speaker(value):
    if value is None:
        return None
    match = re.fullmatch(r'(?:miracl:)?([FM][0-9]{2})', value)
    return match.group(1) if match else value


def exposure(row, manifest):
    # These two historical files use the hash recipe implemented in
    # pipeline/lrw_tcn/evaluate.py:load_source, not the MIRACL basename recipe.
    lrw = manifest in {'runs/lrw500/available_samples_v1/manifest.jsonl',
                       'runs/lrw500/available_samples_v1/evaluation/predictions.jsonl'}
    return {'manifest': manifest, 'split': row.get('split'),
            'source_sha256': row.get('source_sha256'),
            'hash_recipe': 'lrw_json_ordered_file_hashes' if lrw else 'miracl_ordered_basenames_binary_hashes',
            'speaker_id': miracl_speaker(row.get('speaker_id')),
            'original_speaker_id': row.get('speaker_id')}


def checked_letter(row, annotation):
    tokens = []
    previous = 0
    for line in annotation.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError('Alignment must contain start/end/token triples')
        start, end = int(parts[0]), int(parts[1])
        if start < previous or end <= start:
            raise ValueError('Nonmonotonic annotation')
        previous = end
        if parts[2] not in {'sil', 'sp'}:
            tokens.append((start, end, parts[2]))
    allowed = [{'bin', 'lay', 'place', 'set'}, {'blue', 'green', 'red', 'white'},
               {'at', 'by', 'in', 'with'}, set('abcdefghijklmnopqrstuvxyz'),
               {'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'},
               {'again', 'now', 'please', 'soon'}]
    if len(tokens) != 6 or any(t[2] not in vocabulary for t, vocabulary in zip(tokens, allowed)):
        raise ValueError('Ambiguous or unexpected GRID grammar/letter')
    start, end, label = tokens[3]
    fps = row['fps']
    if not math.isfinite(fps) or fps <= 0 or row['alignment_timebase_hz'] != 25000:
        raise ValueError('Unverified timestamp contract')
    first, last = math.floor(start / 25000 * fps), math.ceil(end / 25000 * fps)
    if (row['label'], row['start_units'], row['end_units'], row['start_frame'], row['end_frame']) != (label.upper(), start, end, first, last):
        raise ValueError('Annotation and manifest label/boundary disagree')
    if not 0 <= first < last <= row['decoded_frames'] or last - first < 2:
        raise ValueError('Frame boundary outside verified decoding receipt')
    if previous / 25000 > row['decoded_frames'] / fps + 1 / fps:
        raise ValueError('Annotation exceeds verified duration')
    if row['transcript'] != ' '.join(t[2] for t in tokens):
        raise ValueError('Transcript mismatch')
    return label.upper()


def audit(output):
    if output.exists():
        raise FileExistsError('Choose a new audit directory; existing reports are protected')
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise OSError('20 GiB disk reserve breached')
    output.mkdir(parents=True)
    started = time.monotonic()

    def save(name, value):
        with (output / name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')

    def jsonl(name, values):
        with (output / name).open('x', encoding='utf-8') as stream:
            for value in values:
                stream.write(json.dumps(value) + '\n')

    # Inventory all known historical JSONL manifests, including historical .qa
    # manifests. Reports/predictions are counted as exposures, never as training.
    paths = set()
    for name in ['expansion_preflight_inventory.jsonl', 'expansion_preflight_historical.jsonl']:
        for item in rows(ROOT / '.qa' / name):
            path = ROOT / item['path']
            if path.suffix == '.jsonl' and 'expansion_v1' not in item['path']:
                paths.add(path)
    for base in ['data/processed', 'runs']:
        paths.update(p for p in (ROOT / base).rglob('*.jsonl') if 'expansion_v1' not in str(p))
    inputs = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths) if p.is_file()}
    exposures = defaultdict(list)
    grid_roles = []
    manifest_summary = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        name = path.relative_to(ROOT).as_posix()
        found = 0
        for row in rows(path):
            identity = row.get('id', '')
            if isinstance(identity, str) and identity.startswith('miracl:'):
                # Canonical normalized inventory defines availability, not use.
                if name != 'data/processed/multidataset/miracl_v1.jsonl':
                    exposures[identity].append(exposure(row, name))
                found += 1
            if (row.get('speaker_id') or '').startswith('grid_') and row.get('split'):
                grid_roles.append({'id': identity, 'speaker_id': row['speaker_id'],
                                   'split': row['split'], 'manifest': name})
        manifest_summary.append({'path': name, 'miracl_rows': found})
    save('inputs.json', {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         'manifest_sha256': inputs, 'manifest_scan': manifest_summary,
         'scope': 'All JSONL files in current processed/runs directories and both V1 protected inventories. Canonical inventory is availability, not prior use.'})

    grid_root = ROOT / 'data/external/grid'
    inventory = rows(ROOT / 'data/processed/grid_multispeaker_v1/video_inventory.jsonl')
    assignments = {r['id']: r for r in rows(ROOT / 'data/processed/grid_multispeaker_v1/avhubert_letters.jsonl')}
    conflicts = role_conflicts(grid_roles)
    by_letter, by_speaker, by_split = Counter(), Counter(), Counter()
    joint = Counter()
    exclusions, candidates = [], []
    content_groups = defaultdict(list)
    for index, row in enumerate(inventory, 1):
        identity = row['speaker'] + '/' + row['utterance_id']
        if row['status'] != 'valid':
            exclusions.append({'id': identity, 'reason': 'historical_decode_invalid', 'detail': row.get('error')})
            continue
        try:
            annotation = grid_root / row['alignment']
            annotation_bytes = annotation.read_bytes()
            if hashlib.sha256(annotation_bytes).hexdigest() != row['alignment_sha256']:
                raise ValueError('Original alignment hash changed')
            label = checked_letter(row, annotation_bytes.decode('utf-8'))
            manifest = assignments[identity]
            for key in ['label', 'speaker_id', 'start_frame', 'end_frame', 'video_sha256', 'alignment_sha256']:
                if manifest[key] != row[key]:
                    raise ValueError('Canonical manifest/receipt mismatch: ' + key)
            video = grid_root / row['video']
            if not video.is_file() or video.stat().st_size != row['bytes']:
                raise ValueError('Original video missing or size changed')
            by_letter[label] += 1
            by_speaker[row['speaker_id']] += 1
            by_split[manifest['split']] += 1
            joint[label, row['speaker_id'], manifest['split']] += 1
            content_groups[row['video_sha256']].append(identity)
            candidates.append({**manifest, 'source_video': video.relative_to(ROOT).as_posix(),
                'fps': row['fps'], 'source_decoded_frames_historical': row['decoded_frames'],
                'transcript': row['transcript'], 'status': 'annotation_verified_not_crop_validated',
                'historical_speaker_conflict': row['speaker_id'] in conflicts})
        except (OSError, ValueError, KeyError) as exc:
            exclusions.append({'id': identity, 'reason': str(exc)})
        if index % 2000 == 0:
            print(f'GRID annotations {index}/{len(inventory)}', flush=True)
    jsonl('grid_letters_audit_only.jsonl', candidates)
    jsonl('grid_exclusions.jsonl', exclusions)
    save('grid_audit.json', {'original_pairs': len(inventory), 'annotation_verified': len(candidates),
        'fresh_video_hashes_or_decoding': False, 'historical_validity_is_not_fresh_decoding': True,
        'missing_letters': ['W'], 'complete_A_Z': False, 'by_letter': dict(sorted(by_letter.items())),
        'by_speaker': dict(sorted(by_speaker.items())), 'historical_split_counts': dict(by_split),
        'counts_by_letter_speaker_split': [{'letter': k[0], 'speaker': k[1], 'split': k[2], 'count': n} for k, n in sorted(joint.items())],
        'speaker_role_conflicts_across_all_historical_manifests': conflicts,
        'source_sha256_duplicate_groups_historical': [v for v in content_groups.values() if len(v) > 1],
        'exclusions': len(exclusions), 'timestamp_units_per_second': 25000,
        'boundary_policy': 'floor(start_units*fps/25000), exclusive ceil(end_units*fps/25000)',
        'stage_status': 'STOPPED_SPEAKER_LEAKAGE' if conflicts else 'annotations_verified_pending_crops',
        'published_train_valid_test_manifests': False})
    print(f'GRID audit saved; conflicts: {len(conflicts)} speakers', flush=True)

    miracl = rows(ROOT / 'data/processed/multidataset/miracl_v1.jsonl')
    hashes = defaultdict(set)
    for identity, uses in exposures.items():
        for use in uses:
            if use['source_sha256']:
                hashes[use['source_sha256']].add(identity)
    actual_ids, actual_hashes = set(), defaultdict(list)
    audited, errors = [], []
    for index, row in enumerate(miracl, 1):
        try:
            if row['id'] in actual_ids:
                raise ValueError('Duplicate canonical source ID')
            actual_ids.add(row['id'])
            if row['task'] == 'isolated_word' and row['class_id'] in {'01', '10'}:
                raise ValueError('Disputed publisher word class included')
            if row['source_frames'] != len(row['frames']) or row['end_frame'] != len(row['frames']) or row['start_frame'] != 0:
                raise ValueError('Ordered frame bounds inconsistent')
            current, content_only, lrw_digest = sequence_digest(row['frames'])
            if current != row['source_sha256']:
                raise ValueError('Original MIRACL sequence SHA-256 changed')
            actual_hashes[content_only].append(row['id'])
            used = exposures.get(row['id'], [])
            for use in used:
                expected = lrw_digest if use['hash_recipe'] == 'lrw_json_ordered_file_hashes' else current
                if use['source_sha256'] and use['source_sha256'] != expected:
                    raise ValueError('Historical ID has a different source hash: ' + use['manifest'])
                if use['speaker_id'] and use['speaker_id'] != row['speaker_id']:
                    raise ValueError('Historical ID has a different speaker: ' + use['manifest'])
                if use['split'] and use['split'] != row['split']:
                    raise ValueError('Historical ID has a different split: ' + use['manifest'])
            other_ids = sorted((hashes.get(current, set()) | hashes.get(lrw_digest, set())) - {row['id']})
            audited.append({'id': row['id'], 'task': row['task'], 'word': row['word_label'],
                'transcript': row['raw_transcript'], 'speaker_id': row['speaker_id'], 'split': row['split'],
                'source_frames': len(row['frames']), 'source_sha256': current,
                'ordered_content_hash': content_only, 'lrw_diagnostic_hash': lrw_digest,
                'prior_exposures': used, 'matching_other_historical_ids': other_ids,
                'genuinely_unused': not used and not other_ids,
                'fresh_content_hash_verified': True})
        except (OSError, ValueError, KeyError) as exc:
            errors.append({'id': row['id'], 'error': str(exc)})
        if index % 150 == 0:
            print(f'MIRACL source hashes {index}/{len(miracl)}', flush=True)
    duplicate_groups = [v for v in actual_hashes.values() if len(v) > 1]
    duplicate_ids = {identity for group in duplicate_groups for identity in group}
    for row in audited:
        if row['id'] in duplicate_ids:
            row['genuinely_unused'] = False
    speaker_records = [{**r, 'manifest': 'canonical MIRACL'} for r in miracl]
    speaker_records.extend(use for uses in exposures.values() for use in uses)
    miracl_conflicts = role_conflicts(speaker_records)
    counts = defaultdict(lambda: Counter(total=0, previously_used=0, genuinely_unused=0))
    for row in audited:
        key = row['task'], row['word'] or row['transcript'], row['speaker_id'], row['split']
        counts[key]['total'] += 1
        counts[key]['genuinely_unused' if row['genuinely_unused'] else 'previously_used'] += 1
    jsonl('miracl_sequence_audit.jsonl', audited)
    jsonl('miracl_errors.jsonl', errors)
    save('miracl_audit.json', {'sequences': len(miracl), 'content_verified': len(audited), 'errors': errors,
        'task_counts': dict(Counter(r['task'] for r in audited)),
        'genuinely_unused_by_task': dict(Counter(r['task'] for r in audited if r['genuinely_unused'])),
        'counts_by_task_word_speaker_split': [{'task': k[0], 'label': k[1], 'speaker': k[2], 'split': k[3], **v} for k, v in sorted(counts.items())],
        'speaker_partitions': {s: sorted({r['speaker_id'] for r in miracl if r['split'] == s}) for s in ['train', 'valid', 'test']},
        'speaker_conflicts': miracl_conflicts, 'exact_ordered_frame_hash_duplicate_groups': duplicate_groups,
        'disputed_word_sequences_excluded': 300, 'phrases_are_not_isolated_eight_word_examples': True,
        'unknown_fps_policy': 'null; ordered source frame indices, no fabricated seconds',
        'stage_status': 'STOPPED_INTEGRITY_ERROR' if errors or miracl_conflicts else 'PASS_MANIFEST_AND_SOURCE_HASH_AUDIT',
        'crop_geometry_and_near_duplicates': 'not yet evaluated'})
    after = {name: sha256(ROOT / name) for name in inputs}
    save('input_preservation.json', {'manifest_count': len(inputs),
         'changed': [name for name in inputs if inputs[name] != after[name]]})
    save('summary.json', {'grid_extraction_training': 'stopped_historical_speaker_leakage' if conflicts else 'pending',
         'miracl_audit': 'failed' if errors or miracl_conflicts else 'passed',
         'new_isolated_word_sequences': sum(r['task'] == 'isolated_word' and r['genuinely_unused'] for r in audited),
         'training_started': False, 'sources_downloaded': False, 'seconds': time.monotonic() - started,
         'stop_conditions_are_findings_not_a_successful_training_claim': True})
    save('report_hashes.json', [{'path': p.name, 'sha256': sha256(p), 'bytes': p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()])
    print('Audit complete: findings saved; training has not started.', flush=True)
    return 2 if conflicts or errors or miracl_conflicts else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(audit(args.output.resolve()))
