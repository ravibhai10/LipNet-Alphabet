"""Acquire only overlapping upstream word clips as provenance-bound reference data.

Upstream mouth geometry and speaker metadata are incompatible with our training
contract. These files are quarantined, not silently relabelled or used to train.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
import cv2

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'allenye66/Computer-Vision-Lip-Reading-2.0'
COMMIT = 'e88f8a3e2c14cd25b44a0c180e1ebf3510790822'
WORDS = {'choose','connection','hello','navigation','next','previous','start','stop'}


def fetch(url):
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'LipNet-word-data-audit'}), timeout=40) as source:
                return source.read()
        except OSError:
            if attempt == 3:
                raise
            time.sleep(min(2 ** attempt, 4))


def main():
    tree = json.loads(fetch(f'https://api.github.com/repos/{REPOSITORY}/git/trees/{COMMIT}?recursive=1'))
    if tree.get('truncated'):
        raise ValueError('Upstream tree is truncated; refuse an incomplete inventory.')
    commit = COMMIT
    output = ROOT / 'data/external/allenye66_lipreading2_reference'
    output.mkdir(parents=True, exist_ok=True)
    candidates = [x for x in tree['tree'] if x['type'] == 'blob' and x['path'].endswith('/video.mp4')
                  and x['path'].startswith('collected_data/') and x['path'].split('/')[1].rsplit('_', 1)[0] in WORDS]

    def acquire(item):
        clip = item['path'].split('/')[1]
        target = output / (clip + '.mp4')
        raw = target.read_bytes() if target.exists() else fetch(f'https://raw.githubusercontent.com/{REPOSITORY}/{commit}/{item["path"]}')
        blob = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
        if blob != item['sha'] or len(raw) != item['size']:
            raise ValueError('Upstream Git blob fingerprint/size mismatch: ' + clip)
        if not target.exists():
            temporary = target.with_suffix('.mp4.part')
            temporary.write_bytes(raw)
            temporary.replace(target)
        cap = cv2.VideoCapture(str(target));count = 0;shapes = set()
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                count += 1;shapes.add(tuple(frame.shape))
        finally:
            cap.release()
        if count != 22 or shapes != {(80,112,3)}:
            raise ValueError(f'Unexpected decoded upstream geometry for {clip}: {count}, {shapes}')
        return {'id': f'allenye66:{commit}:{clip}', 'label': clip.rsplit('_',1)[0], 'file': target.name,
                'repository': f'https://github.com/{REPOSITORY}', 'commit': commit, 'upstream_path': item['path'],
                'git_blob_sha1': blob, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                'decoded_frames': count, 'shape': [22,80,112,3], 'container_fps': fps,
                'speaker_id': None, 'split': 'quarantine', 'eligible_for_eight_word_training': False,
                'reason': 'Only hello overlaps; 80x112 prefiltered mouth crops lack our eye-axis 96px convention and speaker/session IDs.',
                'license_status': 'No repository license or participant reuse/consent metadata found in inspected tree; no redistribution grant inferred.'}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as workers:
        rows = sorted(workers.map(acquire, candidates), key=lambda r:r['id'])
    (output / 'inventory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    summary = {'repository': REPOSITORY, 'commit': commit, 'clips': len(rows), 'labels': sorted({r['label'] for r in rows}),
               'bytes': sum(r['bytes'] for r in rows), 'training_clips_added': 0,
               'policy': 'Local reference acquisition only; no relabelling, no inferred identity, no preprocessing equivalence or permission claim.'}
    (output / 'acquisition.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
