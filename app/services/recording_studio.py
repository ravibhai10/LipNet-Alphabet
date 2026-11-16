"""Immutable, task-separated recording store with global speaker assignments.

Client camera/tracking provenance is a declaration, not device attestation.
Server decoding, hashes, lighting statistics and split constraints are checked.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime
import base64
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import uuid
from contextlib import contextmanager

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IDENTIFIER = re.compile(r'^[a-zA-Z0-9_-]{1,80}$')
SPLITS = ('train', 'valid', 'test')


def configuration(root=ROOT):
    value=json.loads((root/'configs/recording_studio.json').read_text(encoding='utf-8'))
    if value['protocol']!='speaker_independent': raise ValueError('Unsupported studio split protocol.')
    for task,spec in value['tasks'].items():
        labels=spec['labels']
        if not labels or len(set(labels))!=len(labels) or any(not isinstance(s,str) or not s or len(s)>32 or not s.isalpha() for s in labels):
            raise ValueError('Configured labels must be unique, explicit single tokens.')
    value['configuration_sha256']=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    asset=root/'frontend/public/models/face_landmarker.task'
    with asset.open('rb') as stream:value['landmarker_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
    return value


def validate_recording(row, config):
    for key in ('id','speaker_id','session_id'):
        if not isinstance(row.get(key),str) or not IDENTIFIER.fullmatch(row[key]): raise ValueError('Invalid recording, speaker or session identifier.')
    task=row.get('task');label=row.get('label')
    if task not in config['tasks'] or label not in config['tasks'][task]['labels']:raise ValueError('Label does not belong to the configured recording task.')
    if row.get('split') not in SPLITS: raise ValueError('Choose train, valid or test.')
    if row.get('consent') is not True or row.get('label_confirmed') is not True: raise ValueError('Recording consent and explicit spoken-label confirmation are required.')
    if row.get('qa_fixture') or row.get('source')!='user_camera' or row.get('capture_kind') not in {'physical_webcam','webcam_replay'}: raise ValueError('Declare genuine camera input; software fixtures are not dataset recordings.')
    if row.get('preprocessing')!=config['preprocessing'] or row.get('landmarker_sha256')!=config['landmarker_sha256']:raise ValueError('Mouth preprocessing or landmarker identity mismatch.')
    try:
        if datetime.fromisoformat(row['recorded_at'].replace('Z','+00:00')).tzinfo is None:raise ValueError()
    except (KeyError,TypeError,ValueError,AttributeError):raise ValueError('A timezone-aware recording timestamp is required.')
    frames=row.get('frames');times=row.get('timestamps_ms')
    if not isinstance(frames,list) or not config['minimum_frames']<=len(frames)<=config['maximum_frames'] or not isinstance(times,list) or len(times)!=len(frames):raise ValueError('Capture 8–76 chronological frames with matching timestamps.')
    if any(type(t) not in (float,int) or not np.isfinite(t) or t<0 or i and t<=times[i-1] for i,t in enumerate(times)):raise ValueError('Frame timestamps must be finite, nonnegative and increasing.')
    duration=times[-1]-times[0]
    if not 0<duration<=config['maximum_duration_ms'] or max(np.diff(times))>config['maximum_gap_ms']:raise ValueError('Capture is too long or contains a video gap. Record again.')
    fps=(len(times)-1)*1000/duration
    camera=row.get('camera',{})
    if any(type(camera.get(k)) is not int or not 96<=camera[k]<=8192 for k in ('width','height')):raise ValueError('Measured camera resolution is required.')
    measured=camera.get('captured_fps')
    if type(measured) not in (float,int) or not np.isfinite(measured) or abs(measured-fps)>.05 or fps<config['minimum_capture_fps']:raise ValueError('Measured capture FPS is invalid or too low.')
    if row.get('frame_count')!=len(frames):raise ValueError('Frame count does not match the recording.')
    quality=row.get('quality',{})
    if quality.get('status')!='passed' or quality.get('face_counts')!=[1]*len(frames) or quality.get('mouth_valid')!=[True]*len(frames):raise ValueError('Exactly one usable face and visible mouth are required in every frame.')
    if task=='negative' and label=='unknown_word':
        utterance=row.get('spoken_text','').strip().lower()
        if not re.fullmatch(r'[a-z]{1,32}',utterance) or utterance in config['tasks']['isolated_word']['labels']:raise ValueError('Confirm an actual isolated negative word outside the configured vocabulary.')
    encoded_hash=hashlib.sha256(json.dumps(frames,separators=(',',':')).encode()).hexdigest()
    if row.get('frames_sha256')!=encoded_hash:raise ValueError('Source JPEG sequence hash mismatch.')
    decoded=[];brightness=[];contrast=[];blur=[]
    for frame in frames:
        if not isinstance(frame,str) or len(frame)>200000 or not frame.startswith('data:image/jpeg;base64,'):raise ValueError('Only bounded JPEG mouth frames are accepted.')
        try:raw=base64.b64decode(frame.split(',',1)[1],validate=True)
        except (ValueError,TypeError):raise ValueError('Invalid base64 JPEG.')
        if not raw.startswith(b'\xff\xd8\xff'):raise ValueError('Invalid JPEG file type.')
        image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape!=(96,96):raise ValueError('Every saved mouth frame must decode at 96×96.')
        mean=float(image.mean());std=float(image.std())
        if not 15<=mean<=240 or std<8:raise ValueError('Decoded mouth lighting/contrast failed quality validation.')
        decoded.append(image);brightness.append(mean);contrast.append(std);blur.append(float(cv2.Laplacian(image,cv2.CV_64F).var()))
    sequence=np.stack(decoded)
    motion=float(np.abs(np.diff(sequence.astype(np.float32),axis=0)).mean())
    return {**row,'source_sha256':encoded_hash,'pixels_sha256':hashlib.sha256(sequence.tobytes()).hexdigest(),
            'validation_status':'validated_pending_dataset','measured':{'fps':fps,'duration_ms':duration,'frame_count':len(frames),
            'mouth_resolution':[96,96],'brightness_mean':float(np.mean(brightness)),'contrast_mean':float(np.mean(contrast)),
            'laplacian_variance_mean':float(np.mean(blur)),'motion_mean':motion,
            'warnings':['Low measured motion; review the utterance.'] if motion<.5 and task!='negative' else [],
            'tracking_evidence':'browser declared; JPEG geometry and photometric quality independently checked',
            'device_attestation':False}}


class RecordingStudio:
    def __init__(self,root=ROOT):self.root=Path(root);self.directory=self.root/'data/studio_v2'
    @contextmanager
    def _db(self):
        self.directory.mkdir(parents=True,exist_ok=True)
        db=sqlite3.connect(self.directory/'recordings.sqlite3',timeout=10)
        db.execute('PRAGMA foreign_keys=ON')
        db.executescript('''CREATE TABLE IF NOT EXISTS speakers(id TEXT PRIMARY KEY, split TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS recordings(id TEXT PRIMARY KEY, task TEXT NOT NULL, label TEXT NOT NULL,
        speaker TEXT NOT NULL REFERENCES speakers(id), split TEXT NOT NULL, pixels TEXT NOT NULL UNIQUE,
        payload TEXT NOT NULL, bundle TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS observations(id TEXT PRIMARY KEY, scope TEXT NOT NULL, payload TEXT NOT NULL);''')
        try:
            with db:yield db
        finally:db.close()
    def save(self,clips,configuration_sha256):
        config=configuration(self.root)
        if configuration_sha256!=config['configuration_sha256']:raise ValueError('Studio configuration changed. Refresh before saving.')
        if not isinstance(clips,list) or not 1<=len(clips)<=32:raise ValueError('Save between one and 32 clips per bundle.')
        checked=[validate_recording(r,config) for r in clips]
        bundle=uuid.uuid4().hex
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                for row in checked:
                    prior=db.execute('SELECT split FROM speakers WHERE id=?',(row['speaker_id'],)).fetchone()
                    if prior and prior[0]!=row['split']:raise ValueError('Speaker leakage: this participant is already assigned to '+prior[0]+'. Use a genuinely new participant for another split.')
                    db.execute('INSERT OR IGNORE INTO speakers VALUES(?,?)',(row['speaker_id'],row['split']))
                    db.execute('INSERT INTO recordings VALUES(?,?,?,?,?,?,?,?)',(row['id'],row['task'],row['label'],row['speaker_id'],row['split'],row['pixels_sha256'],json.dumps(row,separators=(',',':')),bundle))
            except sqlite3.IntegrityError as error:raise ValueError('Duplicate recording ID or decoded content; existing recordings were preserved.') from error
        return {'status':'validated_pending_dataset','bundle_id':bundle,'saved':len(checked),'model_activated':False,
                'recordings':[{'id':r['id'],'task':r['task'],'label':r['label'],'measured':r['measured']} for r in checked]}
    def records(self,task=None,include_frames=False):
        with self._db() as db:
            values=db.execute('SELECT payload FROM recordings'+(' WHERE task=?' if task else ''),((task,) if task else ())).fetchall()
        rows=[json.loads(v[0]) for v in values]
        if not include_frames:
            for row in rows:row.pop('frames',None)
        return rows
    def inventory(self):
        config=configuration(self.root);rows=self.records();speakers=defaultdict(set);counts=Counter()
        for row in rows:counts[row['task'],row['label'],row['split']]+=1;speakers[row['task'],row['label']].add(row['speaker_id'])
        tasks={}
        for task,spec in config['tasks'].items():
            entries=[]
            for label in spec['labels']:
                by_split={s:counts[task,label,s] for s in SPLITS}
                entries.append({'label':label,'samples':sum(by_split.values()),'speakers':len(speakers[task,label]),'splits':by_split,
                    'missing_splits':[s for s,n in by_split.items() if not n],
                    'recommendation':'Prioritize a new speaker in missing partitions; determine further counts from learning curves.',
                    'diversity_status':'not_established' if len(speakers[task,label])<3 else 'requires_validation_learning_curve',
                    'quality_warnings':sum(bool(r['measured']['warnings']) for r in rows if r['task']==task and r['label']==label)})
            missing=[{'label':e['label'],'split':s} for e in entries for s in e['missing_splits']]
            eligible=not missing and task!='negative'
            tasks[task]={'labels':entries,'samples':sum(e['samples'] for e in entries),'structural_training_eligible':eligible,
                'training_authorized':False,'missing_class_splits':missing,
                'reason':'Structural coverage only; review diversity, identity, provenance and freeze a protocol before research fitting.' if eligible else 'Real class coverage in independent partitions is incomplete.'}
        return {'dataset_version':config['dataset_version'],'protocol':'speaker_independent','configuration_sha256':config['configuration_sha256'],
                'samples':len(rows),'unique_speakers':len({r['speaker_id'] for r in rows}),'tasks':tasks,'model_activated':False}
    def export(self,task):
        config=configuration(self.root)
        if task not in config['tasks']:raise ValueError('Unknown recording task.')
        rows=self.records(task,include_frames=True)
        return {'format':'lipnet-studio-v2','task':task,'protocol':'speaker_independent','configuration':config,
                'vocabulary':config['tasks'][task]['labels'],'recordings':rows,
                'model_activated':False,'inventory':self.inventory()['tasks'][task]}


studio=RecordingStudio()
