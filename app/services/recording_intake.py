"""Validate real recordings and save immutable bundles without changing models."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import uuid

from pipeline.word_recognition.common import sha256
from pipeline.word_recognition.personal_data import validate_clip, WORDS


def save_bundle(root: Path, clips: list[dict], vocabulary: list[str], *, owner: str | None = None) -> dict:
    if not clips:
        raise ValueError('No recorded clips provided. Record and confirm the spoken label first.')
    if len(clips) > 1000:
        raise ValueError('Export at most 1,000 recordings per bundle.')
    labels = {c.get('label') for c in clips}
    configured = vocabulary or sorted(labels)
    if (any(not isinstance(w, str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,31}', w) for w in configured)
            or len(set(configured)) != len(configured) or not labels <= set(configured)):
        raise ValueError('Provide unique configured words and a matching confirmed label for every clip.')
    landmark = sha256(root / 'frontend/public/models/face_landmarker.task')
    ids, pixels, sessions = set(), set(), {}
    for clip in clips:
        image = validate_clip(clip, landmark, allowed_words=configured)
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        group = (clip['speaker_id'], clip['session_id'])
        if clip['id'] in ids or digest in pixels:
            raise ValueError('Duplicate recording or decoded video content. Record a new utterance.')
        if group in sessions and sessions[group] != clip['split']:
            raise ValueError('Session leakage: a participant/session crosses dataset partitions.')
        ids.add(clip['id']); pixels.add(digest); sessions[group] = clip['split']
    # An expanded collection is data only. The existing eight-word trainer must
    # reject it until an explicit expanded-vocabulary training protocol is used.
    words = list(WORDS) if set(configured) == set(WORDS) else list(configured)
    bundle = {'format': 'lipnet-word-collection-v1', 'protocol': 'personal_cross_session',
              'vocabulary': words, 'clips': clips}
    payload = json.dumps(bundle, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    if len(payload) > 250 * 1024 * 1024:
        raise ValueError('Bundle exceeds 250 MiB. Export smaller sessions.')
    bundle_id = uuid.uuid4().hex
    directory = root / 'data/personal_recordings'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / (bundle_id + '.json')).open('xb') as stream:
        stream.write(payload)
    if owner is not None:
        from pipeline.adaptive.store import FeedbackStore
        from pipeline.adaptive.lifecycle import register_file
        try: register_file(FeedbackStore(root), bundle_id, directory / (bundle_id + '.json'), owner)
        except BaseException:
            (directory / (bundle_id + '.json')).unlink(missing_ok=True)
            raise
    return {'status': 'validation_required', 'success': False, 'model_activated': False,
            'bundle_id': bundle_id, 'bundle_sha256': hashlib.sha256(payload).hexdigest(),
            'clips_processed': len(clips), 'vocabulary': words,
            'recorded_words': sorted(labels), 'missing_recordings': sorted(set(words) - labels),
            'counts': dict(Counter(c['label'] for c in clips)),
            'message': 'Recordings saved. Training and held-out comparison are required before model activation. '
                       'Export the collection and run the documented separate training workflow.'}
