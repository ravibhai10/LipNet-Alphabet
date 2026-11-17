"""Verified browser recordings -> frozen layer-9 features; no test-driven fitting."""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime
import base64
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time
import cv2
import numpy as np
from .common import ROOT, atomic_json, fingerprint, read_rows, sha256, write_rows
from .pretrained import ENCODER, PREPROCESS

WORDS = ['choose','connection','hello','navigation','next','previous','start','stop']
QUOTAS = {'train':15, 'valid':5, 'test':13}
BASE_RUN = ROOT / 'runs/word_accuracy/avhubert_miracl8_v1'
IDENTIFIER = re.compile(r'^[a-zA-Z0-9_-]{1,80}$')


def validate_clip(row, landmarker, allowed_words=None):
    if any(not isinstance(row.get(k),str) or not IDENTIFIER.fullmatch(row[k]) for k in ('id','speaker_id','session_id')):
        raise ValueError('Invalid recording/participant/session identifier.')
    target_vocab = WORDS if allowed_words is None else allowed_words
    if row.get('label') not in target_vocab:
        raise ValueError(f"Word '{row.get('label')}' is not in configured vocabulary.")
    if row.get('split') not in QUOTAS:
        raise ValueError('Only train/valid/test partitions are accepted.')
    if row.get('consent') is not True or row.get('label_confirmed') is not True or row.get('source') != 'user_camera':
        raise ValueError('Explicit recording consent and user-confirmed word labels are required.')
    if row.get('preprocessing') != 'browser-eyeaxis96-jpeg-v1' or row.get('landmarker_sha256') != landmarker:
        raise ValueError('Browser crop convention or measured landmarker fingerprint mismatch.')
    if not isinstance(row.get('recorded_at'),str) or datetime.fromisoformat(row['recorded_at'].replace('Z','+00:00')).tzinfo is None:
        raise ValueError('An actual timezone-aware recording date is required.')
    frames, times = row.get('frames'), row.get('timestamps_ms')
    if not isinstance(frames,list) or not 8 <= len(frames) <= 76 or not isinstance(times,list) or len(times) != len(frames):
        raise ValueError('Record 8–76 frames with matching timestamps.')
    if any(type(t) not in (int,float) or not math.isfinite(t) or t < 0 or i and t <= times[i-1] for i,t in enumerate(times)):
        raise ValueError('Timestamps must be finite, nonnegative and strictly increasing.')
    duration = row.get('duration_ms')
    if type(duration) not in (int,float) or not math.isfinite(duration) or not 0 <= duration <= 5000 or abs(duration-times[-1]) > 1:
        raise ValueError('Invalid measured word duration.')
    digest = hashlib.sha256(json.dumps(frames,separators=(',',':')).encode()).hexdigest()
    if row.get('frames_sha256') != digest:
        raise ValueError('JPEG frame fingerprint mismatch.')
    images = []
    for frame in frames:
        if not isinstance(frame,str) or not frame.startswith('data:image/jpeg;base64,') or len(frame) > 200000:
            raise ValueError('Only bounded JPEG mouth crops are accepted.')
        try:
            raw = base64.b64decode(frame.split(',',1)[1], validate=True)
        except ValueError as error:
            raise ValueError('Invalid base64 JPEG.') from error
        if not raw.startswith(b'\xff\xd8\xff'):
            raise ValueError('Frame is not a JPEG.')
        image = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (96,96):
            raise ValueError('Expected measured 96×96 mouth crops, not upstream 80×112 images or full faces.')
        images.append(image)
    return np.stack(images)


def audit_partitions(rows, quotas=None):
    quotas = quotas or QUOTAS
    if not rows:
        raise ValueError('No labelled recordings are present.')
    sessions, content, ids = {}, set(), set()
    people = defaultdict(set);counts = Counter()
    for r in rows:
        group = (r['speaker_id'],r['session_id'])
        if group in sessions and sessions[group] != r['split']:
            raise ValueError('Session leakage: one participant/session crosses partitions.')
        if r['roi_sha256'] in content or r['id'] in ids:
            raise ValueError('Duplicate recording/content; rerecord instead of copying clips across partitions.')
        sessions[group] = r['split'];content.add(r['roi_sha256']);ids.add(r['id'])
        people[r['split']].add(r['speaker_id']);counts[(r['speaker_id'],r['split'],r['label'])] += 1
    if set(people) != set(QUOTAS) or not (people['train'] == people['valid'] == people['test']):
        raise ValueError('Personal cross-session evaluation requires the same participant(s) in all three partitions, using different sessions.')
    missing = {f'{person}/{split}/{word}':max(0,quota-counts[person,split,word])
               for person in people['train'] for split,quota in quotas.items() for word in WORDS
               if counts[person,split,word] < quota}
    if missing:
        raise ValueError('More confirmed recordings required: ' + json.dumps(missing,sort_keys=True))
    return {'protocol':'personal_cross_session', 'speaker_independent':False, 'participants':sorted(people['train']),
            'sessions':len(sessions), 'counts':{s:{w:sum(r['split']==s and r['label']==w for r in rows) for w in WORDS} for s in QUOTAS}}


def ingest(bundles, run):
    run = Path(run)
    if run.exists() and any(run.iterdir()):
        raise FileExistsError('Choose a new personal run. Existing recordings/results are protected.')
    landmark = sha256(ROOT/'frontend/public/models/face_landmarker.task')
    clips = {}; receipts = []
    for name in bundles:
        path = Path(name)
        if path.stat().st_size > 250*1024*1024:
            raise ValueError('A collection export exceeds 250 MiB. Export smaller sessions.')
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if data.get('qa_fixture') or data.get('format') != 'lipnet-word-collection-v1' or data.get('protocol') != 'personal_cross_session' or list(data.get('vocabulary',[])) != WORDS:
            raise ValueError('Use an actual eight-word browser collection export.')
        for clip in data['clips']:
            if clip['id'] in clips and clips[clip['id']] != clip:
                raise ValueError('Conflicting copies of the same recording ID.')
            clips[clip['id']] = clip
        receipts.append({'file':str(path.resolve()),'sha256':sha256(path)})
    decoded = {}; rows = []
    for clip in clips.values():
        image = validate_clip(clip,landmark)
        # Decoded pixel identity prevents trivial re-encoding duplicates.
        pixels = hashlib.sha256(image.tobytes()).hexdigest()
        decoded[clip['id']] = image
        rows.append({k:clip[k] for k in ('id','label','speaker_id','session_id','split','recorded_at','frames_sha256','duration_ms')})
        rows[-1].update(roi_sha256=pixels,source='user_camera',preprocessing='browser-eyeaxis96-jpeg-v1')
    audit = audit_partitions(rows)
    if not (BASE_RUN/'head.pt').is_file():
        raise ValueError('The preserved evaluated eight-word baseline is missing.')
    if shutil.disk_usage(run.parent if run.parent.exists() else ROOT).free < 2*1024**3:
        raise RuntimeError('At least 2 GiB free disk is required before extraction.')
    run.mkdir(parents=True,exist_ok=True);(run/'rois').mkdir()
    for row in rows:
        path = run/'rois'/(row['id']+'.npy')
        with path.open('xb') as source:
            np.save(source,decoded[row['id']],allow_pickle=False)
        row['roi'] = str(path.resolve());row['roi_file_sha256'] = sha256(path)
    write_rows(run/'recordings.jsonl',rows)
    contract = {'format':'lipnet-personal-eight-words-v1','vocabulary':WORDS,'preprocessing':PREPROCESS,
                'encoder_sha256':sha256(ENCODER),'landmarker_sha256':landmark,
                'baseline_head':str((BASE_RUN/'head.pt').resolve()),'baseline_head_sha256':sha256(BASE_RUN/'head.pt'),
                'baseline_policy_sha256':sha256(BASE_RUN/'rejection_policy.json'),
                'base_training_features':str((BASE_RUN/'features.jsonl').resolve()),'base_training_manifest_sha256':sha256(BASE_RUN/'features.jsonl'),
                'recordings_sha256':sha256(run/'recordings.jsonl'),'bundles':receipts,'audit':audit,
                'feature_layer':9,'pooling':'mean_std','selection':'Validation sessions only; test never selects a candidate.',
                'limits':'Personalized same-participant/new-session evaluation; not unseen-speaker or general-English accuracy.'}
    atomic_json(run/'personal_protocol.json',contract)
    return contract


def load_contract(run):
    run = Path(run);contract = json.loads((run/'personal_protocol.json').read_text())
    checks = [(run/'recordings.jsonl',contract['recordings_sha256']),
              (contract['baseline_head'],contract['baseline_head_sha256']),
              (Path(contract['baseline_head']).parent/'rejection_policy.json',contract['baseline_policy_sha256']),
              (contract['base_training_features'],contract['base_training_manifest_sha256'])]
    if contract['format'] != 'lipnet-personal-eight-words-v1' or contract['preprocessing'] != PREPROCESS or contract['vocabulary'] != WORDS or any(sha256(p) != digest for p,digest in checks):
        raise ValueError('Personal experiment inputs/recordings/baseline changed.')
    rows = read_rows(run/'recordings.jsonl');audit_partitions(rows)
    return contract,rows


def extract(run, minutes=30):
    from pipeline.avhubert.loader import AVHubertLoader
    run = Path(run);contract,rows = load_contract(run)
    if sha256(ENCODER) != contract['encoder_sha256'] or sha256(ROOT/'frontend/public/models/face_landmarker.task') != contract['landmarker_sha256']:
        raise ValueError('Encoder or landmark model changed.')
    output = run/'features';output.mkdir(exist_ok=True)
    loader = AVHubertLoader(threads=1,low_memory=True).load();started = time.perf_counter();result=[];new=0
    for index,row in enumerate(rows):
        if sha256(row['roi']) != row['roi_file_sha256']:
            raise ValueError('Recording pixels changed.')
        key = fingerprint({'roi':row['roi_file_sha256'],'encoder':contract['encoder_sha256'],'layer':9,'preprocessing':PREPROCESS})
        target = output/(key+'.npy');receipt = output/(key+'.json')
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved['key'] != key or sha256(target) != saved['sha256']:
                raise ValueError('Cached personal features changed.')
        else:
            if time.perf_counter()-started > minutes*60:
                raise TimeoutError('Feature time budget reached; completed feature receipts are resumable.')
            feature = loader.encode_frames(np.load(row['roi'],allow_pickle=False),source_fps=None,target_frames=29,output_layer=9)
            if target.exists():
                raise FileExistsError('Unidentified feature file is protected.')
            temporary = target.with_suffix('.npy.part')
            with temporary.open('xb') as stream:
                np.save(stream,feature,allow_pickle=False)
            temporary.replace(target);atomic_json(receipt,{'key':key,'sha256':sha256(target)});new+=1
        feature = np.load(target,allow_pickle=False)
        if feature.shape != (29,768) or not np.isfinite(feature).all():
            raise ValueError('Expected finite layer-9 [29,768] encoder features.')
        result.append({**row,'feature':str(target.resolve()),'feature_sha256':sha256(target)})
        if (index+1)%16 == 0:
            print(f'Encoded {index+1}/{len(rows)} personal recordings',flush=True)
    write_rows(run/'features.jsonl',result)
    atomic_json(run/'extraction.json',{'new_clips':new,'total':len(result),'seconds':time.perf_counter()-started,
                                     'threads':1,'device':'cpu','encoder_frozen':True})
