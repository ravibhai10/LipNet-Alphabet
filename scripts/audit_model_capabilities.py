"""Read-only artifact/CPU/HTTP audit. Never trains, saves weights, or selects a backend.

Use .venv with --in-process for ASGI checks, and .venv-avhubert-cpu
with --encoder-only for official encoder checks (separate dependency stacks).
Results from generated fixtures and out-of-domain clips are not benchmarks.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inventory():
    artifacts, tensors = [], []
    excluded = {'.git', 'node_modules', '.qa', '__pycache__', '.pytest_cache'}
    for directory, folders, files in os.walk(ROOT):
        folders[:] = [name for name in folders if name not in excluded and not name.startswith('.venv')]
        for name in files:
            path = Path(directory) / name
            if path.suffix.lower() not in {'.pt', '.pth', '.ckpt', '.safetensors', '.onnx'}:
                continue
            relative = path.relative_to(ROOT).as_posix()
            (tensors if relative.startswith('data/processed/') else artifacts).append(relative)
    return sorted(artifacts), sorted(tensors)


def http(base, path, payload=None):
    if base == 'in-process':
        response = TEST_CLIENT.get(path) if payload is None else TEST_CLIENT.post(path, json=payload)
        return {'status': response.status_code, 'body': response.json()}
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(base + path, data=body, headers={'Content-Type': 'application/json'})
    try:
        response = urlopen(request, timeout=45)
    except HTTPError as error:
        response = error
    with response:
        raw = response.read().decode()
        return {'status': response.status, 'body': json.loads(raw)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--http-base', default='http://127.0.0.1:8000')
    parser.add_argument('--encoder-smoke', action='store_true')
    parser.add_argument('--encoder-only', action='store_true', help='Inspect and smoke the official encoder without any web framework test dependencies.')
    parser.add_argument('--in-process', action='store_true', help='Exercise ASGI routes locally without starting or changing any running server.')
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / '.qa' / ('model-capabilities-encoder.json' if args.encoder_only else 'model-capabilities-audit.json')
    import cv2
    import numpy as np
    import torch
    from models.alphabet_model import AlphabetCNNBiLSTM
    from data.schema import VOCABULARY
    from app.main import app
    from src.video_training import resample_temporal
    from tracking.mouth_canonicalization import read_aligned_mouth_video

    if args.encoder_only:
        from pipeline.avhubert.loader import AVHubertLoader
        path = ROOT / 'models/external/avhubert/base_vox_iter5.pt'
        video = ROOT / 'data/external/grid/s1/corpus/T01/video/s1/bbaf2n.mpg'
        started = time.perf_counter()
        loader = AVHubertLoader(path).load()
        features = loader.encode(video, target_frames=29)
        model = loader.models[0]
        result = {'command': subprocess_command(), 'checkpoint': str(path.relative_to(ROOT)),
            'sha256': sha256(path), 'bytes': path.stat().st_size, 'python': platform.python_version(),
            'torch': torch.__version__, 'model_type': type(model).__name__, 'embedding_dim': loader.d_out,
            'parameters': sum(p.numel() for p in model.parameters()), 'feature_shape': list(features.shape),
            'finite_features': bool(np.isfinite(features).all()), 'preprocessing': loader.preprocessing(),
            'seconds_including_load_tracking_encoding': round(time.perf_counter()-started, 3),
            'cluster_projection_shapes': {key:list(value.shape) for key,value in model.state_dict().items()
                if key.startswith(('final_proj', 'label_embs_concat'))},
            'classification_performed': False, 'class_mapping': 'Pretraining cluster targets; no supervised A-Z/LRW/sentence mapping'}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2))
        return

    if args.in_process:
        from fastapi.testclient import TestClient
        global TEST_CLIENT
        TEST_CLIENT = TestClient(app)
        args.http_base = 'in-process'

    torch.set_num_threads(4)
    words = [line.strip() for line in (ROOT / 'data/vocab.txt').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    candidates, tensor_files = inventory()
    report = {
        'audit_date': datetime.now(timezone.utc).date().isoformat(), 'python': platform.python_version(), 'torch': torch.__version__,
        'endpoint_transport': 'in-process ASGI TestClient; not a deployed-server availability claim' if args.in_process else args.http_base,
        'command': subprocess_command(), 'recognition_accuracy_measured': False,
        'vocabulary': {'nonempty': len(words), 'unique': len(set(words)),
            'duplicates': {word: count for word, count in Counter(words).items() if count > 1},
            'sha256': sha256(ROOT / 'data/vocab.txt'), 'index_order': words,
            'schema_matches_uppercase_file_order': VOCABULARY == [word.upper() for word in words]},
        'checkpoint_inventory': candidates, 'dataset_tensor_files': tensor_files,
        'checkpoints': [], 'routes': sorted([
            {'path': route.path, 'methods': sorted(getattr(route, 'methods', []) or [])}
            for route in app.routes], key=lambda row: row['path']),
    }
    report['archived_alphabet_training'] = {
        'summary': json.loads((ROOT / 'models/training_summary.json').read_text()),
        'curve': json.loads((ROOT / 'models/training_curve.json').read_text()),
        'reproduced': False,
        'limitations': 'No original split IDs, evaluated clips, per-class sample counts, per-sample predictions, or checkpoint fingerprint are recorded in these JSON files.',
    }
    report['auxiliary_tracking_models'] = [
        {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': sha256(path),
         'kind': 'MediaPipe facial landmark bundle; not a speech classifier'}
        for path in [ROOT / 'models/face_landmarker.task', ROOT / 'frontend/public/models/face_landmarker.task']
    ]
    # Capture service selection without mutating environment or issuing start/stop/load requests.
    report['health_before'] = http(args.http_base, '/health')
    report['availability'] = http(args.http_base, '/api/model/availability')
    report['evidence'] = http(args.http_base, '/api/evidence')
    report['openapi_paths'] = sorted(http(args.http_base, '/openapi.json')['body']['paths'])

    grid_root = ROOT / 'data/external/grid/s1'
    corpus = grid_root / 'corpus'
    manifest = grid_root / 'grid_manifest.jsonl'
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    positions = [set() for _ in range(6)]
    issues = []
    pause_rows = 0
    for row in rows:
        raw_tokens = row['transcript'].split()
        pause_rows += int('sp' in raw_tokens)
        tokens = [token for token in raw_tokens if token not in {'sil', 'sp'}]
        if len(tokens) != 6:
            issues.append({'utterance': row['utterance_id'], 'issue': 'not six tokens'})
        else:
            for position, word in zip(positions, tokens):
                position.add(word)
        for key in ['video_path', 'audio_path', 'alignment_path']:
            if not (corpus / row[key]).is_file():
                issues.append({'utterance': row['utterance_id'], 'issue': 'missing ' + key})
    report['grid'] = {'manifest_entries': len(rows), 'speaker_ids': sorted({row['speaker_id'] for row in rows}),
        'videos': len(list(corpus.rglob('*.mpg'))), 'audio': len(list(corpus.rglob('*.wav'))),
        'alignments': len(list(corpus.rglob('*.align'))), 'status_counts': dict(Counter(row['validation_status'] for row in rows)),
        'manifest_rows_with_sp_pause_token': pause_rows,
        'grammar_positions': [sorted(position) for position in positions],
        'observed_word_vocabulary': sorted(set().union(*positions)), 'issues': issues}
    lrw_clips = sorted((ROOT / 'data/lrw').glob('*/*/*.mp4'))
    hashes = Counter(sha256(path) for path in lrw_clips)
    report['lrw_data'] = {'clips': len(lrw_clips), 'unique_video_hashes': len(hashes),
        'hash_multiplicity': dict(hashes), 'classes': sorted({path.parent.name for path in lrw_clips}),
        'split_counts': dict(Counter(path.parts[-3] for path in lrw_clips)),
        'provenance': 'generated development clips; scripts/setup_lrw_dataset.py:create_mock_video',
        'training_summary': json.loads((ROOT / 'runs/finetune/lrw_500words/training_summary.json').read_text())}

    # A genuine GRID sentence is valid visual input, but not an isolated A-Z benchmark.
    source = next(row for row in rows if row['video_path'].endswith('bbaf2n.mpg'))
    video = corpus / source['video_path']
    rois, fps = read_aligned_mouth_video(video)
    if len(rois) != source['frame_count']:
        raise ValueError('Tracking dropped frames; alignment indices cannot be safely applied to the compressed ROI array.')
    alignment = []
    for line in (corpus / source['alignment_path']).read_text().splitlines():
        start, end, token = line.split()
        if token not in {'sil', 'sp'}:
            alignment.append((int(start), int(end), token))
    start, end, label = alignment[3]
    first, last = math.floor(start / 25000 * fps), math.ceil(end / 25000 * fps)
    segment = rois[first:last]
    if len(segment) < 1:
        raise ValueError('The real GRID letter segment is empty.')
    # /api/predict needs >=8 frames; temporal interpolation does not create labels.
    crops = resample_temporal(segment, max(8, len(segment))).clip(0, 255).astype(np.uint8)
    encoded = []
    for crop in crops:
        ok, jpeg = cv2.imencode('.jpg', crop)
        if not ok:
            raise ValueError('JPEG encoding failed.')
        encoded.append(base64.b64encode(jpeg).decode())
    report['real_input'] = {'video': video.relative_to(ROOT).as_posix(), 'sha256': sha256(video),
        'transcript': source['transcript'], 'fps': fps, 'tracked_frames': len(rois),
        'alignment_file': (corpus / source['alignment_path']).relative_to(ROOT).as_posix(),
        'letter_token': label.upper(), 'interval_units': [start, end], 'units_per_second': 25000,
        'frame_interval': [first, last], 'segment_frames': len(segment), 'http_frames': len(encoded),
        'interpretation': 'One alignment-derived, coarticulated GRID letter segment; training-speaker overlap unknown. Smoke only, not held-out accuracy.'}
    full_hashes_before = {name: sha256(ROOT / name) for name in candidates}
    previous_state = None
    for name in candidates:
        path = ROOT / name
        result = {'path': name, 'bytes': path.stat().st_size, 'sha256': full_hashes_before[name]}
        if name.startswith('models/external/avhubert/'):
            result['kind'] = 'self-supervised representation encoder, not an LRW or alphabet classifier'
            if args.encoder_smoke:
                from pipeline.avhubert.loader import AVHubertLoader
                began = time.perf_counter()
                encoder = AVHubertLoader(path).load()
                model = encoder.models[0]
                features = encoder.encode_frames(rois, source_fps=fps, target_frames=29)
                result.update(strict_loader='official Fairseq ensemble loader', embedding_dim=encoder.d_out,
                    feature_shape=list(features.shape), finite_features=bool(np.isfinite(features).all()),
                    parameters=sum(p.numel() for p in model.parameters()), preprocessing=encoder.preprocessing(),
                    model_type=type(model).__name__, seconds_including_load=round(time.perf_counter()-began, 3),
                    cluster_projection_shapes={key: list(value.shape) for key, value in model.state_dict().items()
                        if key.startswith(('final_proj', 'label_embs_concat'))})
                del encoder, model
        else:
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            state = checkpoint.get('model_state_dict', checkpoint.get('state_dict'))
            if state is None or 'fc2.weight' not in state:
                raise ValueError('Unrecognized checkpoint requires manual inspection: ' + name)
            count = state['fc2.weight'].shape[0]
            model = AlphabetCNNBiLSTM(num_classes=count).eval()
            model.load_state_dict(state, strict=True)
            frames = 30 if count == 26 else 29
            sequence = np.stack([cv2.resize(crop, (46, 140)) for crop in rois])
            tensor = torch.from_numpy(resample_temporal(sequence, frames) / 255.0)[None, :, None]
            with torch.inference_mode():
                logits = model(tensor)
                probabilities = logits.softmax(-1)[0]
            result.update(model_type=type(model).__name__, strict_load=True,
                batch_norm_training_batches=int(state['bn1.num_batches_tracked']),
                metadata={key: value for key, value in checkpoint.items() if key not in {'model_state_dict', 'state_dict', 'optimizer_state_dict'}},
                parameters=sum(p.numel() for p in model.parameters()), input_shape=list(tensor.shape),
                output_shape=list(logits.shape), output_classes=count, final_weight_shape=list(state['fc2.weight'].shape),
                finite_logits=bool(torch.isfinite(logits).all()), probability_sum=probabilities.sum().item(),
                class_mapping='A-Z index 0-25 in browser adapter' if count == 26 else 'External data/schema.py file-order mapping; vocabulary not embedded in checkpoint',
                smoke_top_index=probabilities.argmax().item(), smoke_score=probabilities.max().item(),
                smoke_label=chr(65+probabilities.argmax().item()) if count==26 else words[probabilities.argmax().item()],
                smoke_input='Full GRID sentence (out of task); execution only, no correctness claim')
            if count == 500:
                if previous_state is not None:
                    result['same_weights_as_first_500_checkpoint'] = all(torch.equal(state[key], previous_state[key]) for key in state)
                else:
                    previous_state = state
        report['checkpoints'].append(result)
        print('Inspected:', name, flush=True)

    report['real_http_prediction'] = http(args.http_base, '/api/predict', {'frames': encoded})
    report['health_after'] = http(args.http_base, '/health')
    report['checkpoint_hashes_unchanged'] = all(sha256(ROOT/name)==digest for name,digest in full_hashes_before.items())
    report['active_backend_unchanged'] = report['health_before']['body']['capabilities']['active_backend'] == report['health_after']['body']['capabilities']['active_backend']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'vocabulary_unique': len(set(words)),
        'checkpoints': len(candidates), 'grid_entries': len(rows), 'lrw_unique_video_hashes': len(hashes),
        'real_http_status': report['real_http_prediction']['status'],
        'checkpoint_hashes_unchanged': report['checkpoint_hashes_unchanged'],
        'active_backend_unchanged': report['active_backend_unchanged']}, indent=2))


def subprocess_command():
    import subprocess
    return subprocess.list2cmdline([sys.executable, *sys.argv])


if __name__ == '__main__':
    main()
