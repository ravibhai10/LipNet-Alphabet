"""Security/lifecycle software fixtures; no recognition or participant evidence."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.security import access
from pipeline.adaptive import lifecycle
from pipeline.adaptive.models import BASE_WORD
from pipeline.adaptive.quality import canonical, digest
from pipeline.adaptive.store import FeedbackStore
from tests.test_adaptive_feedback import source, store, save, predict


def credentials(monkeypatch, tmp_path, token='a'*40):
    path = tmp_path / 'access.json'
    path.write_text(json.dumps({'principals': [{'id': 'stable-user', 'role': 'collector',
        'token_sha256': [hashlib.sha256(token.encode()).hexdigest()]}]}))
    monkeypatch.setenv('LIPNET_ACCESS_FILE', str(path))
    return path


def test_rotation_preserves_owner(monkeypatch, tmp_path):
    credentials(monkeypatch, tmp_path)
    before = access('Bearer '+'a'*40)
    credentials(monkeypatch, tmp_path, 'b'*40)
    assert access('Bearer '+'b'*40) == before == {'actor': 'stable-user', 'role': 'collector'}
    with pytest.raises(HTTPException) as error:
        access('Bearer '+'a'*40)
    assert error.value.status_code == 401


def test_invalid_credentials_fail_closed(monkeypatch, tmp_path):
    path = credentials(monkeypatch, tmp_path)
    path.write_text('{broken')
    with pytest.raises(HTTPException) as error:
        access('Bearer '+'a'*40)
    assert error.value.status_code == 503


def test_sensitive_routes_reject_before_json(monkeypatch, tmp_path):
    from app.main import app
    credentials(monkeypatch, tmp_path)
    with TestClient(app) as client:
        for route in ['/api/word/train-personal', '/api/adaptive/captures', '/api/studio/bundles']:
            assert client.post(route, content=b'{bad').status_code == 401
        result = client.post('/api/word/train-personal', content=b'{bad',
                             headers={'Authorization': 'Bearer '+'a'*40})
        assert result.status_code == 403  # Legacy collection has no per-clip owner semantics.


def test_expiry_blocks_access_before_cleanup_and_deletes(store, source):
    save(store, source)
    with store.db() as db:
        db.execute("UPDATE retention SET expires='2000-01-01T00:00:00+00:00'")
    with pytest.raises(LookupError, match='expired'):
        store.blob('capture-01', 'media', 'owner')
    assert lifecycle.report(store)['expired_pending'] == 1
    assert lifecycle.maintenance(store)['expired'] == 1
    assert lifecycle.maintenance(store)['expired'] == 0
    row = store.inspect('capture-01')
    assert row['status'] == 'WITHDRAWN' and row['prediction'] is None
    assert 'similarity_signature' not in row['metadata']


def test_renewal_requires_owner_and_consent(store, source):
    save(store, source)
    with pytest.raises(ValueError): lifecycle.renew(store, 'capture-01', 'owner', False)
    with pytest.raises(LookupError): lifecycle.renew(store, 'capture-01', 'other', True)
    assert lifecycle.renew(store, 'capture-01', 'owner', True)['expires_at'] > datetime.now(timezone.utc).isoformat()


def test_withdrawal_invalidates_lineage_and_deletes_only_private_derivatives(store, source):
    save(store, source); predict(store)
    candidate = store.directory / 'derivatives/candidate/head.pt'
    candidate.parent.mkdir(parents=True); candidate.write_bytes(b'private fixture')
    reference = store.root / 'models/reference.pt'
    reference.parent.mkdir(); reference.write_bytes(b'preserved fixture')
    model = {'id': 'candidate', 'task': 'isolated_word', 'owner': 'owner',
             'checkpoint': str(candidate.relative_to(store.root)), 'training_dataset_version': 'version'}
    with store.db() as db:
        db.execute('INSERT INTO versions VALUES(?,?,?,?)', ('version', 'isolated_word', 'unused',
            json.dumps({'record_ids': ['capture-01']})))
        db.execute('INSERT INTO models VALUES(?,?,?)', ('candidate', 'isolated_word', json.dumps(model)))
        db.execute('INSERT INTO champions VALUES(?,?,?)', ('owner:isolated_word', 'candidate', 'candidate'))
    store.withdraw('capture-01', 'owner')
    with store.db(write=False) as db:
        assert db.execute('SELECT model,previous FROM champions').fetchone()[:] == (BASE_WORD, None)
        assert json.loads(db.execute('SELECT payload FROM models').fetchone()[0])['invalidated']
    lifecycle.cleanup(store)
    assert not candidate.exists() and reference.read_bytes() == b'preserved fixture'


def test_cleanup_refuses_reference_target(store):
    reference = store.root / 'models/reference.pt'; reference.parent.mkdir(); reference.write_bytes(b'protected')
    with store.db() as db:
        db.execute('INSERT INTO models VALUES(?,?,?)', ('bad', 'isolated_word', json.dumps({'checkpoint': 'models/reference.pt'})))
        db.execute("INSERT INTO cleanup VALUES('bad','withdrawal','now','PENDING',NULL)")
    lifecycle.cleanup(store)
    with store.db(write=False) as db:
        assert db.execute('SELECT status FROM cleanup').fetchone()[0] == 'RETRY'
    assert reference.read_bytes() == b'protected'


def test_restart_recovers_jobs_without_claiming_success(store):
    with store.db() as db:
        db.execute("INSERT INTO jobs VALUES('job','version','RUNNING','now',NULL,NULL)")
        db.execute("INSERT INTO attempts VALUES('model','binding','RUNNING','now')")
    lifecycle.recover(FeedbackStore(store.root))
    with store.db(write=False) as db:
        assert db.execute('SELECT status FROM jobs').fetchone()[0] == 'INTERRUPTED'
        assert db.execute('SELECT state FROM attempts').fetchone()[0] == 'INTERRUPTED'


def test_readers_do_not_reserve_writer_lock(store):
    with store.db(write=False) as db:
        # Deferred BEGIN permits an independent writer before the first read.
        with ThreadPoolExecutor(max_workers=1) as executor:
            def write():
                with store.db() as writer:
                    store.event(writer, 'record', 'operator', 'concurrent_write', {})
            executor.submit(write).result(timeout=3)
        assert db.execute('SELECT COUNT(*) FROM events').fetchone()[0] == 1


def test_managed_collection_expiry(store):
    path = store.root / 'data/personal_recordings/bundle.json'
    path.parent.mkdir(parents=True); path.write_bytes(b'private source fixture')
    lifecycle.register_file(store, 'bundle', path, 'owner')
    with store.db() as db:
        db.execute("UPDATE private_files SET expires='2000-01-01'")
    lifecycle.maintenance(store)
    assert not path.exists()
    assert lifecycle.report(store)['managed_collections'][0]['state'] == 'DELETED'


def test_research_expanded_route_disabled_by_default(monkeypatch):
    from app.main import app
    monkeypatch.delenv('LIPNET_ENABLE_RESEARCH_INFERENCE', raising=False)
    with TestClient(app) as client:
        assert client.post('/api/predict/word/expanded', json={'frames': ['fixture']*8}).status_code == 503
        health = client.get('/api/word/expanded/health').json()
        assert health['inference_ready'] is False and health['research_only'] is True


def test_research_job_recovery_and_capacity(tmp_path):
    import threading
    from app.services.job_manager import JobManager, JobState
    directory = tmp_path / 'jobs'
    directory.mkdir()
    (directory / 'previous.json').write_text(json.dumps({
        'job_id': 'previous', 'name': 'research', 'state': 'RUNNING'}))
    manager = JobManager(directory)
    assert manager.get('previous').state == JobState.INTERRUPTED
    entered, release = threading.Event(), threading.Event()
    def work():
        entered.set()
        assert release.wait(5)
        return {'software_fixture': True}
    job = manager.submit('fixture', work)
    try:
        assert entered.wait(3)
        assert manager.cancel(job)
        with pytest.raises(RuntimeError, match='capacity'):
            manager.submit('must-not-start', lambda: None)
    finally:
        release.set()
        manager._workers[job].join(3)
    assert manager.get(job).state == JobState.CANCELLED
    assert JobManager(directory).get(job).state == JobState.CANCELLED


def test_openapi_auth_matches_middleware():
    from app.main import app
    schema = app.openapi()
    paths = schema['paths']
    assert paths['/api/word/train-personal']['post']['security'] == [{'PrivateBearer': []}]
    assert paths['/api/adaptive/models/{identifier}/deploy']['post']['x-minimum-role'] == 'reviewer'
    assert paths['/api/adaptive/predict']['post']['x-minimum-role'] == 'collector'
    assert paths['/api/adaptive/captures']['post']['x-upload-limit-bytes'] == 90*1024**2
    assert 'security' not in paths['/api/predict/word']['post']


def test_global_personal_adapter_cannot_serve_or_auto_fit(monkeypatch, tmp_path):
    from app.main import app
    path = credentials(monkeypatch, tmp_path)
    value = json.loads(path.read_text()); value['principals'][0]['role'] = 'reviewer'
    path.write_text(json.dumps(value))
    headers = {'Authorization': 'Bearer '+'a'*40}
    with TestClient(app) as client:
        health = client.get('/api/word/personal/health', headers=headers).json()
        assert health['inference_ready'] is False and health['research_only'] is True
        assert client.post('/api/predict/word/personal', headers=headers,
                           json={'frames': ['fixture']*8}).status_code == 503
        tracks = client.get('/api/models').json()['tracks']
        assert tracks['personal_word_recognition']['ready'] is False


def test_final_test_resume_is_bound_to_same_candidate(store, monkeypatch):
    version = {'id': 'version', 'owner': 'owner', 'records': [
        {'id': 'test-record', 'speaker': 'participant', 'session': 'held-out', 'split': 'test'}]}
    monkeypatch.setattr(store, 'version', lambda _: version)
    with store.db() as db:
        for model in ['first', 'second']:
            db.execute('INSERT INTO attempts VALUES(?,?,?,?)', (model, 'binding', 'PREPARED', 'now'))
    assert store.claim_test('version', 'reviewer', resume_model='first')['claimed'] == 1
    assert store.claim_test('version', 'reviewer', resume_model='first')['claimed'] == 1
    with pytest.raises(FileExistsError):
        store.claim_test('version', 'reviewer', resume_model='second')


def test_jpeg_bomb_rejected_before_native_allocation(monkeypatch):
    import base64
    from app.services.browser_inference import BrowserInference
    import cv2
    # SOF dimensions 65535 by 65535; no pixel payload is needed to reject it.
    raw = b'\xff\xd8\xff\xc0\x00\x0b\x08\xff\xff\xff\xff\x01\x01\x11\x00'
    def forbidden(*args, **kwargs):
        raise AssertionError('Native decoder must not see an oversized header')
    monkeypatch.setattr(cv2, 'imdecode', forbidden)
    with pytest.raises(ValueError, match='pre-decode'):
        BrowserInference.decode_jpegs([base64.b64encode(raw).decode()]*8)


def test_similarity_threshold_is_quarantine_heuristic():
    from pipeline.adaptive.quality_policy import near_duplicate
    assert near_duplicate([100]*512, [102]*512)
    assert not near_duplicate([100]*512, [103]*512)
    assert not near_duplicate([], [])


def test_real_codec_correspondence_positive_and_mismatch(tmp_path):
    import cv2
    import numpy as np
    from pipeline.adaptive.quality_policy import correspondence
    path = tmp_path / 'software-fixture.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (128, 128))
    assert writer.isOpened(), 'MP4 codec is required to verify source correspondence'
    images = [np.full((128,128,3), 30+i*20, np.uint8) for i in range(8)]
    try:
        for frame in images: writer.write(frame)
    finally:
        writer.release()
    raw = path.read_bytes()
    report = correspondence(raw, '.mp4', images, [100*i for i in range(8)])
    assert report['usable_sample_ratio'] == 1
    with pytest.raises(ValueError, match='does not correspond'):
        correspondence(raw, '.mp4', [255-frame for frame in images], [100*i for i in range(8)])


def test_rejected_media_deleted_immediately(store, source):
    save(store, source)
    row = store.review('capture-01', 'reviewer', 'REJECTED', 'Unusable recording, re-record with visible lips')
    assert row['status'] == 'REJECTED' and row['media_available'] is False
    with pytest.raises(LookupError): store.blob('capture-01', 'rois', 'owner')
    assert 'quality' not in store.inspect('capture-01')['metadata']


def test_portable_manifest_read_preserves_original_bytes(tmp_path):
    from pipeline.paths import relocate_row
    path = tmp_path / 'data/source/clip.mp4'
    path.parent.mkdir(parents=True); path.write_bytes(b'genuine-byte-placeholder-software-only')
    original = {'video': 'D:/old-checkout/data/source/clip.mp4', 'id': 'unchanged'}
    result = relocate_row(original, tmp_path)
    assert result['video'] == str(path.resolve()) and result['id'] == 'unchanged'
    assert original['video'].startswith('D:/old-checkout')
    assert relocate_row({'video':'D:/old/data/missing.mp4'}, tmp_path)['video'].endswith('data/missing.mp4')


def test_eligible_backup_expires_with_source_and_blocks_traversal(store, source):
    save(store, source)
    backup = store.directory / 'backups/eligible.zip'
    backup.parent.mkdir(parents=True); backup.write_bytes(b'private fixture archive')
    with pytest.raises(ValueError):
        lifecycle.register_backup(store, 'unsafe', 'owner', 'capture-01', '../../outside.zip')
    with pytest.raises(LookupError):
        lifecycle.register_backup(store, 'other', 'wrong-owner', 'capture-01', 'eligible.zip')
    receipt = lifecycle.register_backup(store, 'backup', 'owner', 'capture-01', 'eligible.zip')
    assert receipt['source_record'] == 'capture-01'
    store.withdraw('capture-01', 'owner')
    lifecycle.cleanup(store)
    assert not backup.exists()


def test_trainer_shutdown_and_reopen(store):
    from pipeline.adaptive.training import BatchTrainer
    trainer = BatchTrainer(store)
    trainer.close()
    assert trainer.executor._shutdown
    trainer.restart_executor()
    assert not trainer.executor._shutdown
    trainer.close()


def test_model_selection_defaults_reference_and_rejects_other_owner(store, monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from pipeline.adaptive import models
    with store.db() as db:
        for identifier, payload in [
            (BASE_WORD, {'id': BASE_WORD, 'checkpoint_sha256': 'fixture-version'}),
            ('bob-model', {'id':'bob-model', 'owner':'bob', 'checkpoint':'private-fixture'})]:
            db.execute('INSERT INTO models VALUES(?,?,?)', (identifier, 'isolated_word', json.dumps(payload)))
        db.execute('INSERT INTO champions VALUES(?,?,NULL)', ('alice:isolated_word', 'bob-model'))
    monkeypatch.setattr(models, 'baseline_adapter', lambda _: SimpleNamespace(
        predict=lambda _: {'model_id': BASE_WORD, 'model_version':'fixture-version'}))
    registry = models.ModelRegistry(store)
    rois = np.zeros((8,96,96), np.uint8)
    assert registry.predict('isolated_word', rois, owner='alice')['model_id'] == BASE_WORD
    with pytest.raises(ValueError, match='owner'):
        registry.predict('isolated_word', rois, owner='alice', selection='adaptive')
    assert [m['id'] for m in registry.list('alice')['models']] == [BASE_WORD]
