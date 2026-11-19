"""Transactional private media store. Original evidence and label events are separate."""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import io
import json
from pathlib import Path
import sqlite3
import uuid
import threading

from .quality import canonical, digest

STATUSES = {'PENDING_REVIEW', 'APPROVED', 'REJECTED', 'DUPLICATE', 'QUARANTINED', 'WITHDRAWN', 'TRAINING_INCLUDED'}
ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


class FeedbackStore:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        from app.settings import settings
        self.directory = settings().private_root if self.root.resolve() == ROOT.resolve() else self.root/'data/adaptive_private'
        self._schema_lock = threading.Lock()
        self._initialized = False

    @contextmanager
    def db(self, write=True):
        self.directory.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.directory/'feedback.sqlite3', timeout=5)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA secure_delete=ON')
        with self._schema_lock:
            if not self._initialized:
                self._initialize(db)
                self._initialized = True
        try:
            db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            yield db
            db.commit()
        except BaseException:
            db.rollback(); raise
        finally:
            db.close()

    @staticmethod
    def _initialize(db):
        db.executescript('''
        CREATE TABLE IF NOT EXISTS captures(id TEXT PRIMARY KEY, owner TEXT NOT NULL, request_hash TEXT NOT NULL,
          task TEXT NOT NULL, speaker TEXT NOT NULL, session TEXT NOT NULL, source TEXT NOT NULL, pixels TEXT NOT NULL,
          metadata TEXT NOT NULL, media BLOB, rois BLOB, prediction TEXT, verdict TEXT, label TEXT,
          status TEXT NOT NULL, created TEXT NOT NULL, reason TEXT);
        CREATE INDEX IF NOT EXISTS captures_pixels ON captures(pixels);
        CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, record TEXT NOT NULL, actor TEXT NOT NULL,
          action TEXT NOT NULL, payload TEXT NOT NULL, at TEXT NOT NULL, previous TEXT NOT NULL, hash TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS versions(id TEXT PRIMARY KEY, task TEXT NOT NULL, hash TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS assignments(speaker TEXT PRIMARY KEY, split TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, version TEXT NOT NULL, status TEXT NOT NULL,
          created TEXT NOT NULL, result TEXT, error TEXT);
        CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY, task TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS champions(task TEXT PRIMARY KEY, model TEXT NOT NULL, previous TEXT);
        CREATE TABLE IF NOT EXISTS test_locks(record TEXT PRIMARY KEY, version TEXT NOT NULL, at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS evaluation_speakers(speaker TEXT PRIMARY KEY, version TEXT NOT NULL, at TEXT NOT NULL);
        ''')
        from .lifecycle import initialize
        initialize(db)

    def event(self, db, record, actor, action, payload):
        prior = db.execute('SELECT hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
        previous = prior[0] if prior else '0'*64
        stamp = now()
        value = {'record': record, 'actor': actor, 'action': action, 'payload': payload, 'at': stamp, 'previous': previous}
        db.execute('INSERT INTO events(record,actor,action,payload,at,previous,hash) VALUES(?,?,?,?,?,?,?)',
                   (record, actor, action, json.dumps(payload), stamp, previous, digest(canonical(value))))

    def row(self, db, identifier, owner=None):
        row = db.execute('SELECT * FROM captures WHERE id=?', (identifier,)).fetchone()
        if row is None or (owner is not None and row['owner'] != owner):
            raise LookupError('Recording not found.')
        expiration = db.execute('SELECT expires FROM retention WHERE record=?', (identifier,)).fetchone()
        if expiration and expiration[0] <= now() and row['status'] != 'WITHDRAWN':
            raise LookupError('Recording expired; access blocked pending scheduled cleanup.')
        return row

    def save_capture(self, identifier, owner, request_hash, meta, media, rois):
        from app.settings import settings
        from .lifecycle import expiry, private_file_bytes
        with self.db() as db:
            used = db.execute('SELECT COALESCE(SUM(length(media)+length(rois)),0) FROM captures').fetchone()[0]
            if used + private_file_bytes(self, db) + len(media) + len(rois) > settings().private_quota_bytes:
                raise ValueError('Private media quota exceeded. Expire or withdraw existing recordings first.')
            prior = db.execute('SELECT * FROM captures WHERE id=?', (identifier,)).fetchone()
            if prior:
                if prior['owner'] != owner or prior['request_hash'] != request_hash:
                    raise ValueError('Capture ID already belongs to another request.')
                return self.public(prior)
            duplicate = db.execute('SELECT id FROM captures WHERE source=? OR pixels=?',
                                   (meta['source_sha256'], meta['pixels_sha256'])).fetchone()
            exposed=db.execute('SELECT 1 FROM evaluation_speakers WHERE speaker=?',(meta['speaker_id'],)).fetchone()
            from .quality_policy import near_duplicate
            signature=meta.get('similarity_signature',[])
            for candidate in db.execute("SELECT metadata FROM captures WHERE owner=? AND status!='WITHDRAWN'",(owner,)):
                if signature and near_duplicate(signature,json.loads(candidate[0]).get('similarity_signature',[])) and not duplicate:
                    meta['warnings']=sorted(set(meta['warnings']+['near_duplicate_review_required']))
                    break
            status = 'DUPLICATE' if duplicate else 'QUARANTINED' if meta['warnings'] or exposed else 'PENDING_REVIEW'
            db.execute('INSERT INTO captures VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       (identifier, owner, request_hash, meta['task'], meta['speaker_id'], meta['session_id'],
                        meta['source_sha256'], meta['pixels_sha256'], json.dumps(meta), media, rois,
                        None, None, None, status, now(), 'duplicate_content' if duplicate else None))
            db.execute('INSERT INTO retention VALUES(?,?,NULL)', (identifier, expiry()))
            self.event(db, identifier, owner, 'capture', {'status': status, 'source_sha256': meta['source_sha256']})
            return self.public(self.row(db, identifier))

    @staticmethod
    def public(row, detail=False):
        meta = json.loads(row['metadata'])
        value = {k: row[k] for k in ('id', 'task', 'speaker', 'session', 'verdict', 'label', 'status', 'created', 'reason')}
        value.update(prediction=json.loads(row['prediction']) if row['prediction'] else None,
                     source_sha256=row['source'], warnings=meta['warnings'], capture_kind=meta['capture_kind'],
                     frame_count=meta['frame_count'], captured_fps=meta['captured_fps'],
                     original_video_available=bool(meta.get('original_video')) and row['status']!='WITHDRAWN',
                     duration_ms=meta['duration_ms'], media_available=bool(row['has_media']) if 'has_media' in row.keys() else row['media'] is not None)
        if detail:
            value['metadata'] = meta
        return value

    def inspect(self, identifier, owner=None):
        with self.db(write=False) as db:
            row = self.row(db, identifier, owner)
            value = self.public(row, True)
            value['audit'] = [dict(e) for e in db.execute('SELECT actor,action,payload,at,previous,hash FROM events WHERE record=? ORDER BY seq', (identifier,))]
            return value

    def blob(self, identifier, kind, owner=None):
        if kind not in {'media', 'rois'}:
            raise ValueError('Unknown media kind.')
        with self.db(write=False) as db:
            row = self.row(db, identifier, owner)
            if row[kind] is None or row['status'] == 'WITHDRAWN':
                raise LookupError('Recording media is no longer available.')
            return row[kind]

    def prediction(self, identifier, prediction, actor):
        with self.db() as db:
            row = self.row(db, identifier, actor)
            if row['status'] == 'WITHDRAWN':
                raise ValueError('Recording was withdrawn.')
            if row['prediction']:
                return self.public(row)
            db.execute('UPDATE captures SET prediction=? WHERE id=?', (json.dumps(prediction), identifier))
            self.event(db, identifier, actor, 'prediction', {'model_id': prediction['model_id'], 'model_version': prediction['model_version']})
            return self.public(self.row(db, identifier))

    def feedback(self, identifier, actor, verdict, label, labels):
        with self.db() as db:
            row = self.row(db, identifier, actor)
            if row['status'] == 'WITHDRAWN':
                raise ValueError('Withdrawn feedback cannot be resubmitted.')
            if not row['prediction']:
                raise ValueError('Run actual inference before confirming or correcting.')
            prediction = json.loads(row['prediction'])
            if verdict == 'correct':
                if prediction['prediction'] is None:
                    raise ValueError('An uncertain result requires an explicit correction label.')
                if label != prediction['prediction']:
                    raise ValueError('Confirm the actual accepted prediction explicitly.')
            elif verdict != 'incorrect' or not label:
                raise ValueError('Incorrect feedback requires a corrected label.')
            if label not in labels and label != 'UNKNOWN':
                raise ValueError('Corrected label does not belong to this task vocabulary.')
            if verdict == 'incorrect' and label == prediction['prediction']:
                raise ValueError('Choose a corrected label different from the accepted prediction.')
            if row['verdict']:
                if row['verdict'] == verdict and row['label'] == label:
                    return self.public(row)
                raise ValueError('Feedback already exists. Withdraw it before recording a new utterance.')
            status = row['status']
            if label == 'UNKNOWN' and status != 'DUPLICATE':
                status = 'QUARANTINED'
            db.execute('UPDATE captures SET verdict=?, label=?, status=? WHERE id=?', (verdict, label, status, identifier))
            self.event(db, identifier, actor, 'feedback', {'verdict': verdict, 'label': label, 'status': status})
            return self.public(self.row(db, identifier))

    def review(self, identifier, actor, decision, reason):
        if decision not in {'APPROVED', 'REJECTED', 'QUARANTINED'} or not reason.strip():
            raise ValueError('Provide a review decision and its reason.')
        with self.db() as db:
            row = self.row(db, identifier)
            if row['status'] in {'WITHDRAWN', 'TRAINING_INCLUDED', 'DUPLICATE'}:
                raise ValueError('This record cannot be relabelled or approved.')
            meta = json.loads(row['metadata'])
            if decision=='APPROVED' and db.execute('SELECT 1 FROM evaluation_speakers WHERE speaker=?',(row['speaker'],)).fetchone():
                raise ValueError('A previously evaluated test speaker remains quarantined from the independent feedback protocol.')
            if decision == 'APPROVED' and (not row['verdict'] or row['verdict'] == 'unknown'):
                raise ValueError('Approval requires a verified label and passing recording quality. UNKNOWN records are negative evaluation only. Rerecord failed quality.')
            if decision == 'APPROVED' and meta.get('warnings'):
                if len(reason.strip()) < 20:
                    raise ValueError('Resolving quality quarantine requires a detailed reviewer rationale.')
                meta['resolved_quality_warnings'] = {'warnings': meta['warnings'], 'reviewer': actor, 'reason': reason}
                meta['warnings'] = []
            if decision == 'REJECTED':
                meta = {'warnings': [], 'capture_kind': 'deleted', 'frame_count': 0,
                        'captured_fps': None, 'duration_ms': None, 'original_video': None,
                        'deletion_reason': 'reviewer_rejected', 'deleted_at': now()}
                db.execute('UPDATE captures SET media=NULL,rois=NULL,prediction=NULL,verdict=NULL,label=NULL WHERE id=?', (identifier,))
            db.execute('UPDATE captures SET status=?,reason=?,metadata=? WHERE id=?', (decision, reason, json.dumps(meta), identifier))
            self.event(db, identifier, actor, 'review', {'decision': decision, 'reason': reason})
            return self.public(self.row(db, identifier))

    def withdraw(self, identifier, actor, reviewer=False):
        with self.db() as db:
            row = db.execute('SELECT * FROM captures WHERE id=?', (identifier,)).fetchone()
            if row is None or (not reviewer and row['owner'] != actor):
                raise LookupError('Recording not found.')
            included = any(identifier in json.loads(v[0])['record_ids'] for v in db.execute('SELECT payload FROM versions'))
            if row['status'] != 'WITHDRAWN':
                # Secure deletion removes the only stored source/ROI BLOBs. Hash receipts and
                # immutable training lineage remain; no frame or landmark copy stays in metadata.
                from .lifecycle import invalidate
                affected = invalidate(db, self, identifier)
                meta = {'warnings': [], 'capture_kind': 'deleted', 'frame_count': 0,
                        'captured_fps': None, 'duration_ms': None, 'original_video': None,
                        'deletion_reason': 'withdrawal_or_expiry', 'deleted_at': now()}
                db.execute("UPDATE captures SET status='WITHDRAWN',media=NULL,rois=NULL,prediction=NULL,verdict=NULL,label=NULL,metadata=?,source=?,pixels=? WHERE id=?",
                           (json.dumps(meta), 'deleted:'+identifier, 'deleted:'+identifier, identifier))
                self.event(db, identifier, actor, 'withdraw', {'included_in_manifest': included, 'requires_model_review': included})
            return {'id': identifier, 'status': 'WITHDRAWN', 'media_deleted': True, 'requires_model_review': included}

    def list(self, owner=None, limit=1000, **filters):
        allowed = {'task', 'label', 'speaker', 'session', 'status', 'verdict'}
        sql = 'SELECT id,owner,task,speaker,session,metadata,prediction,verdict,label,status,created,reason,source,media IS NOT NULL AS has_media FROM captures WHERE 1=1'; values = []
        if owner:
            sql += ' AND owner=?'; values.append(owner)
        for key, value in filters.items():
            if key in allowed and value:
                sql += f' AND {key}=?'; values.append(value)
        with self.db(write=False) as db:
            return [self.public(r) for r in db.execute(sql+' ORDER BY created DESC'+(' LIMIT ?' if limit else ''), values+([limit] if limit else []))]

    def stats(self, owner=None):
        rows = self.list(owner,limit=None)
        with self.db(write=False) as db:
            versions = [json.loads(v[0]) for v in db.execute('SELECT payload FROM versions')]
        return {'total': len(rows), 'statuses': dict(Counter(r['status'] for r in rows)),
                'confirmed': sum(r['verdict']=='correct' for r in rows), 'corrected': sum(r['verdict']=='incorrect' for r in rows),
                'unique_speakers': len({r['speaker'] for r in rows}),
                'classes': dict(Counter(f"{r['task']}:{r['label']}" for r in rows if r['label'])),
                'growth': dict(sorted(Counter(r['created'][:10] for r in rows).items())),
                'versions': [{k:v[k] for k in ('id','task','created','manifest_sha256')} for v in versions] if owner is None else [],
                'training_eligible': sum(r['status']=='APPROVED' for r in rows)}

    def hard_examples(self, owner=None):
        rows = self.list(owner)
        mistakes = Counter(r['prediction']['candidate'] for r in rows if r['verdict']=='incorrect' and r['prediction'])
        values = []
        for row in rows:
            p = row['prediction'] or {}; reasons = list(row['warnings'])
            if row['verdict']=='incorrect':
                reasons += ['human_corrected', 'speaker_specific_failure']
                scores = p.get('top_k', [])
                if len(scores)>1 and scores[0]['score']-scores[1]['score']>.5:
                    reasons.append('large_margin_error')
            if p.get('confidence', 1)<.6: reasons.append('low_confidence_review_only')
            if mistakes[p.get('candidate')]>1: reasons.append('repeated_class_failure')
            if reasons:
                values.append({**row, 'review_reasons': sorted(set(reasons)), 'training_weight': None})
        return values

    def freeze(self, task, labels, assignments, actor, minimum_per_class=2):
        if set(assignments.values()) != {'train', 'valid', 'test'}:
            raise ValueError('Assign genuinely independent speakers to train, valid and test.')
        with self.db() as db:
            rows = db.execute("SELECT * FROM captures WHERE task=? AND status IN ('APPROVED','TRAINING_INCLUDED')", (task,)).fetchall()
            rows = [r for r in rows if r['speaker'] in assignments]
            if not rows:
                raise ValueError('No approved feedback is available.')
            counts = Counter(); receipts = []; negatives = []; seen = set()
            # Existing evaluation speakers are forbidden from all new feedback versions.
            # Identity aliases cannot be detected by software; reviewers must verify IDs.
            forbidden = self.historical_evaluation_speakers()
            for row in rows:
                split = assignments[row['speaker']]
                if row['speaker'] in forbidden or db.execute('SELECT 1 FROM evaluation_speakers WHERE speaker=?',(row['speaker'],)).fetchone():
                    raise ValueError('Historical evaluation speaker must be quarantined from new independent protocols.')
                negative=row['label']=='UNKNOWN'
                if row['label'] not in labels and not negative:
                    raise ValueError('Expanded and unknown labels require a separate training protocol.')
                if negative and split=='train':
                    raise ValueError('UNKNOWN negatives may only enter independent validation/test evaluation, never classifier training.')
                if row['pixels'] in seen:
                    raise ValueError('Duplicate decoded sequence in dataset.')
                seen.add(row['pixels'])
                prior = db.execute('SELECT split FROM assignments WHERE speaker=?', (row['speaker'],)).fetchone()
                if prior and prior[0] != split:
                    raise ValueError('Speaker leakage across dataset versions.')
                if not negative:counts[split,row['label']] += 1
                meta = json.loads(row['metadata'])
                if meta['consent'] is not True or meta['warnings'] or row['rois'] is None:
                    raise ValueError('Source consent or quality no longer allows training.')
                (negatives if negative else receipts).append({'id':row['id'], 'speaker':row['speaker'], 'session':row['session'], 'label':row['label'],
                                 'split':split, 'source_sha256':row['source'], 'pixels_sha256':row['pixels'],
                                 'roi_sha256':meta['roi_sha256'], 'preprocessing':meta['preprocessing']})
            if any(counts[s,l]<minimum_per_class for s in ('train','valid','test') for l in labels):
                raise ValueError('Every class requires at least two approved utterances in each independent partition.')
            for split in ('train','valid','test'):
                ns = [counts[split,l] for l in labels]
                if max(ns)>5*min(ns): raise ValueError('Class imbalance exceeds the fixed 5:1 limit.')
            identifier = uuid.uuid4().hex
            value = {'id': identifier, 'task':task, 'labels':labels, 'created':now(), 'protocol':'speaker_independent',
                     'records':receipts, 'negative_records':negatives, 'record_ids':[r['id'] for r in receipts+negatives], 'minimum_per_class':minimum_per_class,
                     'limits':'Structural checks only; identities are reviewer verified, sample sufficiency remains unproven.'}
            value['manifest_sha256'] = digest(canonical(value))
            db.execute('INSERT INTO versions VALUES(?,?,?,?)', (identifier, task, value['manifest_sha256'], json.dumps(value)))
            for row in rows:
                db.execute('INSERT OR IGNORE INTO assignments VALUES(?,?)', (row['speaker'],assignments[row['speaker']]))
                db.execute("UPDATE captures SET status='TRAINING_INCLUDED' WHERE id=?", (row['id'],))
            self.event(db, identifier, actor, 'freeze_dataset', {'manifest_sha256':value['manifest_sha256'], 'task':task})
            return value

    def historical_evaluation_speakers(self):
        values = set()
        for path in (self.root/'data/processed/multidataset/miracl_v1.jsonl',
                     self.root/'data/processed/grid_multispeaker_v1/avhubert_letters.jsonl'):
            if path.is_file():
                for line in path.read_text(encoding='utf-8').splitlines():
                    r = json.loads(line)
                    if r.get('split') in {'valid','test'}:
                        value = r.get('speaker_id',r.get('speaker'))
                        if value: values.add(str(value))
        return values

    def version(self, identifier):
        with self.db(write=False) as db:
            row = db.execute('SELECT payload FROM versions WHERE id=?',(identifier,)).fetchone()
            if not row: raise LookupError('Dataset version not found.')
            value = json.loads(row[0]); expected = value.pop('manifest_sha256')
            if digest(canonical(value)) != expected: raise ValueError('Dataset manifest integrity failed.')
            value['manifest_sha256'] = expected
            for record in value['records']+value.get('negative_records',[]):
                r = self.row(db,record['id'])
                if r['status']=='WITHDRAWN' or r['rois'] is None:
                    raise ValueError('A dataset recording was withdrawn. Freeze a new reviewed version.')
            return value

    def claim_test(self,identifier,actor,resume_model=None):
        version=self.version(identifier)
        records=[r for r in version['records']+version.get('negative_records',[]) if r['split']=='test']
        if not records:raise ValueError('A complete held-out partition is required.')
        with self.db() as db:
            def identity(row):
                return ('personal:'+version['owner']+':'+row['speaker']+':'+row['session']) if version.get('owner') else row['speaker']
            for row in records:
                locked=db.execute('SELECT version FROM test_locks WHERE record=?',(row['id'],)).fetchone()
                resumable=resume_model is not None and db.execute('SELECT 1 FROM attempts WHERE model=?',(resume_model,)).fetchone() is not None
                claimant=db.execute('SELECT model FROM test_claims WHERE record=?',(row['id'],)).fetchone()
                if locked and not (resumable and locked[0]==identifier and claimant and claimant[0]==resume_model):
                    raise FileExistsError('The held-out recording was already accessed by another final evaluation.')
                speaker=db.execute('SELECT version FROM evaluation_speakers WHERE speaker=?',(identity(row),)).fetchone()
                if speaker and speaker[0]!=identifier:
                    raise FileExistsError('This held-out speaker was already evaluated in another version.')
            for row in records:
                db.execute('INSERT OR IGNORE INTO test_claims VALUES(?,?)',(row['id'],resume_model or 'legacy:'+identifier))
                db.execute('INSERT OR IGNORE INTO test_locks VALUES(?,?,?)',(row['id'],identifier,now()))
                db.execute('INSERT OR IGNORE INTO evaluation_speakers VALUES(?,?,?)',(identity(row),identifier,now()))
            self.event(db,identifier,actor,'claim_final_test',{'recordings':len(records),'purpose':'one final report; not selection'})
        return {'claimed':len(records),'dataset_version':identifier}

    def expire_uncommitted(self, actor='retention', days=None):
        # Compatibility entry point delegates to the single validated retention policy.
        from app.settings import settings
        if days is not None and days != settings().retention_days:
            raise ValueError('Set LIPNET_RETENTION_DAYS; per-call retention overrides are unsupported.')
        from .lifecycle import maintenance
        return maintenance(self)
