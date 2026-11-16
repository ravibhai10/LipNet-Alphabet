"""Real saved-input API execution and checkpoint preservation receipt, not accuracy."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:8001')
    parser.add_argument('--stage', choices=['before', 'after'], required=True)
    args = parser.parse_args()
    output = ROOT / '.qa' / f'live-assistant-real-{args.stage}.json'
    if output.exists():
        raise SystemExit(f'Refusing to overwrite {output.name}')
    manifest = ROOT / 'runs/word_accuracy/avhubert_miracl8_v1/prepared.jsonl'
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    row = next(r for r in rows if r['split'] == 'test' and r.get('word_label') == 'connection' and r['usable'])
    rois = np.load(row['roi'], allow_pickle=False)
    frames = []
    for roi in rois:
        ok, encoded = cv2.imencode('.jpg', roi, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        frames.append(base64.b64encode(encoded).decode())
    with urlopen(args.base + '/api/word/health', timeout=180) as source:
        health = json.load(source)
    if not health['inference_ready']:
        raise RuntimeError('Existing word backend is not ready')
    request = Request(args.base + '/api/predict/word', data=json.dumps({'frames': frames}).encode(),
                      headers={'Content-Type': 'application/json'})
    start = time.perf_counter()
    with urlopen(request, timeout=90) as source:
        result = json.load(source)
        status = source.status
    round_trip_ms = (time.perf_counter() - start) * 1000
    assert result['task'] == 'isolated_word' and result['predicted_word'] in health['vocabulary']
    assert len(result['all_probabilities']) == 8
    assert np.isfinite([p['prob'] for p in result['all_probabilities']]).all()
    checkpoints = ['models/alphabet_classifier.pt', 'models/external/avhubert/base_vox_iter5.pt',
                   'runs/word_accuracy/avhubert_miracl8_v1/head.pt', 'models/external/lrw_tcn/resnet18_dctcn_video.pth',
                   'runs/multidataset/miracl8_cpu_v2/best.pt']
    hashes = {p: fingerprint(ROOT / p) for p in checkpoints}
    receipt = {'stage': args.stage, 'scope': 'Same preserved running backend, one genuine saved clip per stage. '
               'Not a controlled UI performance benchmark, recognition evaluation or physical webcam test.',
               'source_id': row['id'], 'source_sha256': row['source_sha256'], 'label': row['word_label'],
               'roi_sha256': fingerprint(Path(row['roi'])), 'frames': len(frames), 'roi_shape': list(rois.shape),
               'http_status': status, 'round_trip_ms': round(round_trip_ms, 2), 'result': result,
               'matches_label': result['predicted_word'] == row['word_label'], 'checkpoint_sha256': hashes}
    if args.stage == 'after':
        previous = json.loads((output.parent / 'live-assistant-real-before.json').read_text())
        receipt['checkpoints_unchanged'] = hashes == previous['checkpoint_sha256']
        receipt['same_source_and_probability_vector'] = previous['source_id'] == row['id'] and previous['result']['all_probabilities'] == result['all_probabilities']
        assert receipt['checkpoints_unchanged']
    output.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({k: receipt[k] for k in ['stage', 'source_id', 'http_status', 'round_trip_ms', 'matches_label']}) +
          f" server_ms={result.get('latency_ms')} uncertain={result['uncertain']}")


if __name__ == '__main__':
    main()
