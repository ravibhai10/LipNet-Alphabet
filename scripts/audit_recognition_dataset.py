"""Read-only dataset audit. Counts every row; optional full source decoding.

Image sequences, annotated video segments, and unrelated tasks stay separate.
No automatic label correction, file deletion, or data splitting is performed.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(manifest, *, decode=False, limit=None):
    manifest = Path(manifest).resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    classes, speakers, splits, tasks = Counter(), Counter(), Counter(), Counter()
    owners, hashes, ids = defaultdict(set), defaultdict(list), Counter()
    records, durations, lengths, fps_values, resolutions = [], [], [], Counter(), Counter()
    start = time.perf_counter()
    for index, row in enumerate(rows):
        label = row.get('word_label') or row.get('label') or row.get('letter')
        task = row.get('task', 'unspecified')
        split = row.get('split', 'unknown')
        speaker = row.get('speaker_id') or row.get('speaker')
        key = f'{task}/{label}'
        classes[key] += 1; speakers[str(speaker)] += 1; splits[str(split)] += 1; tasks[task] += 1
        if speaker: owners[str(speaker)].add(split)
        identity = row.get('id', str(index)); ids[identity] += 1
        checksum = row.get('source_sha256') or row.get('video_sha256')
        if checksum: hashes[checksum].append({'id': identity, 'split': split})
        item = {'id': identity, 'label': label, 'task': task, 'speaker': speaker,
                'split': split, 'errors': [], 'source_decoded': False,
                'face_visibility': 'not_measured', 'mouth_occlusion': 'not_measured'}
        if not label and task != 'phrase': item['errors'].append('missing_label')
        paths = row.get('frames')
        path = None
        if paths is not None:
            paths = [Path(p) if Path(p).is_absolute() else manifest.parent / p for p in paths]
            item['frame_count'] = len(paths); lengths.append(len(paths))
            if any(not p.is_file() for p in paths): item['errors'].append('missing_frame')
            if row.get('source_frames') not in (None, len(paths)): item['errors'].append('frame_count_mismatch')
            item['fps'] = row.get('fps')  # Never infer unknown MIRACL FPS.
        else:
            raw = row.get('video') or row.get('video_path') or row.get('source_video')
            if raw:
                path = Path(raw) if Path(raw).is_absolute() else manifest.parent / raw
                if not path.is_file(): item['errors'].append('missing_video')
            else: item['errors'].append('missing_source')
        if decode and (limit is None or index < limit) and not item['errors']:
            shapes, brightness, contrasts = Counter(), [], []
            count = 0
            def inspect(image):
                nonlocal count
                if image is None or image.size == 0:
                    item['errors'].append('corrupted_frame'); return
                count += 1; shapes[str(list(image.shape[:2]))] += 1
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
                small = cv2.resize(gray, (32, 32))
                brightness.append(float(small.mean())); contrasts.append(float(small.std()))
            if paths is not None:
                actual_hash = hashlib.sha256()
                for p in paths:
                    inspect(cv2.imread(str(p)))
                    actual_hash.update(p.name.encode()); actual_hash.update(bytes.fromhex(digest(p)))
                if checksum and actual_hash.hexdigest() != checksum: item['errors'].append('source_hash_mismatch')
            elif path:
                cap = cv2.VideoCapture(str(path))
                native_fps = cap.get(cv2.CAP_PROP_FPS)
                expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if not cap.isOpened(): item['errors'].append('unreadable_video')
                while True:
                    ok, frame = cap.read()
                    if not ok: break
                    inspect(frame)
                cap.release()
                if count != expected: item['errors'].append('decoded_frame_count_mismatch')
                item.update(native_frames=count, fps=native_fps, reported_frames=expected)
                first, last = row.get('start_frame', 0), row.get('end_frame', count)
                if not 0 <= first < last <= count: item['errors'].append('invalid_segment_bounds')
                item['frame_count'] = last - first; lengths.append(last-first)
                if native_fps > 0:
                    duration = (last-first) / native_fps; durations.append(duration); item['duration_seconds'] = duration
                if checksum and digest(path) != checksum: item['errors'].append('source_hash_mismatch')
            if count == 0: item['errors'].append('no_decoded_frames')
            item.update(source_decoded=True, decoded_frames=count, resolutions=dict(shapes),
                        brightness_mean=float(np.mean(brightness)) if brightness else None,
                        contrast_mean=float(np.mean(contrasts)) if contrasts else None)
            resolutions.update(shapes); fps_values[str(item.get('fps'))] += 1
        records.append(item)
        if (index+1) % 500 == 0: print(f'audited {index+1}/{len(rows)}', file=sys.stderr, flush=True)
    def distribution(values):
        return dict(zip(('min','p50','p95','max'), map(float,np.percentile(values,[0,50,95,100])))) if values else None
    duplicate_groups = [v for v in hashes.values() if len(v)>1]
    return {'manifest': str(manifest), 'manifest_sha256': digest(manifest), 'samples': len(rows),
            'tasks': dict(tasks), 'samples_per_class': dict(classes), 'samples_per_speaker': dict(speakers),
            'samples_per_split': dict(splits), 'speakers': len(owners),
            'speaker_leakage': {s:sorted(v) for s,v in owners.items() if len(v)>1},
            'missing_speaker_samples': speakers['None'], 'duplicate_ids': {k:v for k,v in ids.items() if v>1},
            'duplicate_source_groups': duplicate_groups,
            'cross_split_source_duplicates': [g for g in duplicate_groups if len({r['split'] for r in g})>1],
            'corrupted_or_invalid': sum(bool(r['errors']) for r in records),
            'errors': [r for r in records if r['errors']],
            'source_decoded_samples': sum(r['source_decoded'] for r in records),
            'fps_distribution': dict(fps_values), 'resolution_frames': dict(resolutions),
            'duration_seconds_distribution': distribution(durations), 'frame_count_distribution': distribution(lengths),
            'seconds': time.perf_counter()-start, 'records': records,
            'limitations': 'Publisher labels are checked structurally, not independently relabelled. '
                          'Full-frame brightness is not mouth visibility. Face/occlusion require separate measured tracking. '
                          'Unknown FPS remains unknown. Duplicate source bytes do not prove absence of re-encoded duplicates.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--decode', action='store_true'); parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError('Audit outputs are immutable; choose a new --out.')
    if args.limit is not None and args.limit < 1: parser.error('--limit must be positive')
    result = audit(args.manifest, decode=args.decode, limit=args.limit)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in {'records','errors','samples_per_class','samples_per_speaker','duplicate_source_groups'}},indent=2))


if __name__ == '__main__': main()
