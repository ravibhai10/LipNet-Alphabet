"""Private retention, verified participants and owner-scoped dataset contracts.

No private media is kept in deletion receipts. Only artifacts in managed private
candidate directories can be deleted; preserved reference artifacts are excluded.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import uuid
from .quality import canonical, digest
from .store import now


def initialize(db):
    db.executescript('''
      CREATE TABLE IF NOT EXISTS retention(record TEXT PRIMARY KEY, expires TEXT NOT NULL, renewed TEXT);
      CREATE TABLE IF NOT EXISTS participants(owner TEXT NOT NULL, id TEXT NOT NULL, verified_by TEXT NOT NULL,
        permission TEXT NOT NULL, PRIMARY KEY(owner,id));
      CREATE TABLE IF NOT EXISTS cleanup(id TEXT PRIMARY KEY, reason TEXT NOT NULL, at TEXT NOT NULL,
        status TEXT NOT NULL, error TEXT);
      CREATE TABLE IF NOT EXISTS attempts(model TEXT PRIMARY KEY, binding TEXT NOT NULL,
        state TEXT NOT NULL, updated TEXT NOT NULL);
    ''')


def expiry(days=None):
    from app.settings import settings
    return (datetime.now(timezone.utc) + timedelta(days=days or settings().retention_days)).isoformat()


def verify_participant(store, owner, identifier, reviewer, permission):
    if not owner or not identifier or not permission.strip():
        raise ValueError('Stable owner, participant and permission record required.')
    with store.db() as db:
        db.execute('INSERT OR REPLACE INTO participants VALUES(?,?,?,?)', (owner, identifier, reviewer, permission))
        store.event(db, identifier, reviewer, 'participant_verified', {'owner': owner, 'permission_recorded': True})
    return {'owner': owner, 'participant_id': identifier, 'verified': True}


def freeze_personal(store, task, labels, owner, assignments, actor, minimum_per_class=2):
    if set(assignments.values()) != {'train', 'valid', 'test'}:
        raise ValueError('Personal adaptation requires distinct held-out sessions in every partition.')
    with store.db() as db:
        records = db.execute("SELECT * FROM captures WHERE owner=? AND task=? AND status='APPROVED'", (owner, task)).fetchall()
        counts = Counter(); selected = []; negatives = []
        speakers = set(); fingerprints = set()
        for row in records:
            if row['session'] not in assignments:
                continue
            if not db.execute('SELECT 1 FROM participants WHERE owner=? AND id=?', (owner, row['speaker'])).fetchone():
                raise ValueError('Reviewer-verified participant identity and permission are required.')
            if row['speaker'] in store.historical_evaluation_speakers():
                raise ValueError('Previously exposed historical evaluation participants are ineligible.')
            meta = json.loads(row['metadata'])
            if meta.get('warnings') or not meta.get('original_video') or row['rois'] is None:
                raise ValueError('Only passing source-video-bound recordings can enter personal training.')
            split = assignments[row['session']]
            identity='personal:'+owner+':'+row['speaker']+':'+row['session']
            if db.execute('SELECT 1 FROM evaluation_speakers WHERE speaker=?',(identity,)).fetchone():
                raise ValueError('This personal final-test session was already exposed.')
            if row['label'] not in labels and row['label'] != 'UNKNOWN':
                raise ValueError('Label outside the fixed adaptation vocabulary.')
            if row['pixels'] in fingerprints:
                raise ValueError('Duplicate content cannot cross sessions.')
            fingerprints.add(row['pixels']); speakers.add(row['speaker'])
            negative = row['label'] == 'UNKNOWN'
            if negative and split == 'train':
                raise ValueError('UNKNOWN recordings are independent evaluation only.')
            # Prior frozen sessions and test exposure cannot be repurposed.
            for previous in db.execute('SELECT payload FROM versions'):
                prior = json.loads(previous[0])
                for old in prior.get('records', []) + prior.get('negative_records', []):
                    if prior.get('owner') == owner and old['session'] == row['session'] and old['split'] != split:
                        raise ValueError('Session assignment changed across personal dataset versions.')
            if db.execute('SELECT 1 FROM test_locks WHERE record=?', (row['id'],)).fetchone():
                raise ValueError('An exposed final-test recording cannot be reused.')
            receipt = {'id': row['id'], 'speaker': row['speaker'], 'session': row['session'], 'label': row['label'],
                       'split': split, 'source_sha256': row['source'], 'pixels_sha256': row['pixels'],
                       'roi_sha256': meta['roi_sha256'], 'preprocessing': meta['preprocessing']}
            (negatives if negative else selected).append(receipt)
            if not negative:
                counts[split, row['label']] += 1
        if len(speakers) != 1:
            raise ValueError('A personal dataset must contain exactly one verified participant.')
        if any(counts[s, label] < minimum_per_class for s in ('train', 'valid', 'test') for label in labels):
            raise ValueError('Every class needs at least two approved examples in each held-out session partition; sufficiency is unproven.')
        identifier = uuid.uuid4().hex
        value = {'id': identifier, 'task': task, 'owner': owner, 'labels': labels, 'created': now(),
                 'protocol': 'personal_cross_session_v2', 'speaker_independent': False, 'records': selected,
                 'negative_records': negatives, 'record_ids': [r['id'] for r in selected + negatives],
                 'limits': 'One participant, held-out sessions; no population or webcam generalization claim.'}
        value['manifest_sha256'] = digest(canonical(value))
        db.execute('INSERT INTO versions VALUES(?,?,?,?)', (identifier, task, value['manifest_sha256'], json.dumps(value)))
        for record in selected + negatives:
            db.execute("UPDATE captures SET status='TRAINING_INCLUDED' WHERE id=?", (record['id'],))
        store.event(db, identifier, actor, 'freeze_personal_dataset', {'owner': owner, 'manifest_sha256': value['manifest_sha256']})
        return value


def renew(store, identifier, owner, consent):
    if consent is not True:
        raise ValueError('Explicit renewed consent required.')
    with store.db() as db:
        row = store.row(db, identifier, owner)
        if row['status'] == 'WITHDRAWN':
            raise ValueError('Deleted media cannot be renewed or restored.')
        stamp = expiry()
        db.execute('UPDATE retention SET expires=?,renewed=? WHERE record=?', (stamp, now(), identifier))
        store.event(db, identifier, owner, 'consent_renewed', {'expires': stamp})
    return {'id': identifier, 'expires_at': stamp}


def invalidate(db, store, identifier):
    versions = {r[0] for r in db.execute('SELECT id,payload FROM versions') if identifier in json.loads(r[1])['record_ids']}
    affected = []
    for row in db.execute('SELECT id,task,payload FROM models').fetchall():
        model = json.loads(row['payload'])
        if model.get('training_dataset_version') not in versions:
            continue
        model['invalidated'] = True; model['invalidation_reason'] = 'source_withdrawal_or_expiry'
        db.execute('UPDATE models SET payload=? WHERE id=?', (json.dumps(model), row['id']))
        from .models import BASE_WORD, BASE_ALPHABET
        baseline = BASE_WORD if row['task'] == 'isolated_word' else BASE_ALPHABET
        db.execute('UPDATE champions SET model=?,previous=NULL WHERE model=?', (baseline, row['id']))
        db.execute('INSERT OR REPLACE INTO cleanup VALUES(?,?,?,?,NULL)', (row['id'], 'private_derivatives', now(), 'PENDING'))
        affected.append(row['id'])
    return affected


def cleanup(store):
    """Idempotent deletion restricted to managed private candidate directories."""
    with store.db(write=False) as db:
        pending = [dict(row) for row in db.execute("SELECT * FROM cleanup WHERE status!='DELETED'")]
    for item in pending:
        try:
            with store.db(write=False) as db:
                record = db.execute('SELECT payload FROM models WHERE id=?', (item['id'],)).fetchone()
            if record:
                model = json.loads(record[0]); path = (store.root / model['checkpoint']).resolve()
                allowed = [(store.directory / 'derivatives').resolve(), (store.root / 'runs/adaptive').resolve()]
                if not any(path.is_relative_to(base) for base in allowed):
                    raise ValueError('Deletion target is outside managed private derivatives; reference files protected.')
                directory = path.parent
                # Refuse symlink escapes; delete only regular contents of this model's directory.
                for child in directory.rglob('*') if directory.exists() else []:
                    if child.is_symlink() or not child.resolve().is_relative_to(directory):
                        raise ValueError('Private derivative symlink requires operator review.')
                    if child.is_file():
                        child.unlink()
            with store.db() as db:
                db.execute("UPDATE cleanup SET status='DELETED',error=NULL WHERE id=?", (item['id'],))
                store.event(db, item['id'], 'retention-worker', 'private_derivatives_deleted', {})
        except Exception:
            with store.db() as db:
                db.execute("UPDATE cleanup SET status='RETRY',error='Cleanup failed; operator review required.' WHERE id=?", (item['id'],))


def maintenance(store):
    with store.db(write=False) as db:
        identifiers = [r[0] for r in db.execute("SELECT c.id FROM captures c JOIN retention r ON c.id=r.record WHERE r.expires<=? AND c.status!='WITHDRAWN'", (now(),))]
    for identifier in identifiers:
        store.withdraw(identifier, 'retention-worker', reviewer=True)
    cleanup(store)
    with store.db(write=False) as db:
        pending = db.execute("SELECT COUNT(*) FROM cleanup WHERE status!='DELETED'").fetchone()[0]
        retained = db.execute("SELECT COUNT(*),COALESCE(SUM(length(media)+length(rois)),0) FROM captures WHERE status!='WITHDRAWN'").fetchone()
    return {'expired': len(identifiers), 'cleanup_pending': pending, 'retained_records': retained[0],
            'media_bytes': retained[1], 'backup_policy': 'Only explicitly managed backups can be erased; unmanaged copies require operator deletion.',
            'browser_policy': 'Client must expire its own copies; offline browser erasure cannot be attested by server.'}


def recover(store):
    with store.db() as db:
        db.execute("UPDATE jobs SET status='INTERRUPTED',error='Process stopped; immutable attempt can be retried.' WHERE status IN ('RUNNING','QUEUED')")
        db.execute("UPDATE attempts SET state='INTERRUPTED',updated=? WHERE state='RUNNING'", (now(),))
