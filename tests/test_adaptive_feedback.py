"""Synthetic software checks only. These fixtures are never biometric dataset records."""
from datetime import datetime, timezone, timedelta
import base64
import io
import json
from types import SimpleNamespace
import zipfile

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pipeline.adaptive.quality import validate_capture, canonical, digest
from pipeline.adaptive.store import FeedbackStore
from pipeline.adaptive.models import ModelRegistry, TASK_LABELS
from pipeline.adaptive.training import BatchTrainer, metrics, compare


def tracked(images,root):
    p=[SimpleNamespace(x=.5,y=.5) for _ in range(478)]
    for i,x,y in [(61,.4,.5),(291,.6,.5),(13,.5,.47),(14,.5,.53),(33,.3,.3),(263,.7,.3)]:
        p[i]=SimpleNamespace(x=x,y=y)
    return [[p] for _ in images]


@pytest.fixture
def source(tmp_path):
    asset=tmp_path/'frontend/public/models/face_landmarker.task';asset.parent.mkdir(parents=True);asset.write_bytes(b'synthetic-test-identity')
    rng=np.random.default_rng(42)
    frames=['data:image/png;base64,'+base64.b64encode(cv2.imencode('.png',rng.integers(30,220,(128,128,3),dtype=np.uint8))[1]).decode() for _ in range(8)]
    return {'id':'capture-01','task':'isolated_word','speaker_id':'new-speaker','session_id':'session-1',
            'recorded_at':datetime.now(timezone.utc).isoformat(),'capture_kind':'webcam_replay',
            'consent':True,'frames':frames,'timestamps_ms':[i*100 for i in range(8)]}


@pytest.fixture
def store(tmp_path):return FeedbackStore(tmp_path)


def save(store,source,identifier='capture-01',owner='owner'):
    row={**source,'id':identifier}
    meta,media,rois=validate_capture(row,store.root,tracked)
    return store.save_capture(identifier,owner,digest(canonical(row)),meta,media,rois)


def predict(store,identifier='capture-01',accepted='hello',owner='owner'):
    return store.prediction(identifier,{'task':'isolated_word','prediction':accepted,'candidate':'hello','confidence':.97,
            'model_id':'baseline','model_version':'abc','top_k':[{'label':'hello','score':.97},{'label':'stop','score':.02}]},owner)


def approved(store,source,identifier,label,speaker):
    row={**source,'speaker_id':speaker}
    # Distinct original evidence for each synthetic fixture; no source duplication hidden.
    frame=cv2.imdecode(np.frombuffer(base64.b64decode(row['frames'][0].split(',')[1]),np.uint8),cv2.IMREAD_COLOR)
    frame[0,0]=np.frombuffer(bytes.fromhex(digest(identifier.encode()))[:3],np.uint8)
    row['frames']=[*row['frames']];row['frames'][0]='data:image/png;base64,'+base64.b64encode(cv2.imencode('.png',frame)[1]).decode()
    save(store,row,identifier);predict(store,identifier,accepted=label)
    store.feedback(identifier,'owner','correct',label,[label])
    return store.review(identifier,'reviewer','APPROVED','Reviewed fixture for software test')


def test_source_roundtrip_and_measured_fps(store,source):
    value=save(store,source)
    assert value['status']=='PENDING_REVIEW' and value['captured_fps']==10
    assert value['label'] is None and value['verdict'] is None
    meta=store.inspect(value['id'])['metadata']
    assert meta['quality'][0]['occlusion']=='not_measured' and meta['device_attestation'] is False
    with zipfile.ZipFile(io.BytesIO(store.blob(value['id'],'media'))) as archive:
        assert archive.read('0000.png')==base64.b64decode(source['frames'][0].split(',')[1])
    rois=np.load(io.BytesIO(store.blob(value['id'],'rois')),allow_pickle=False)
    assert rois.shape==(8,96,96)


@pytest.mark.parametrize('change',[
    {'consent':False},{'timestamps_ms':[0]*8},{'timestamps_ms':[0,100,200,300,400,500,600,float('nan')]},
    {'timestamps_ms':[i*400 for i in range(8)]},{'frames':['data:image/png;base64,broken']*8},
    {'frames':['data:image/jpeg;base64,broken']*8},{'recorded_at':'2026-01-01T00:00:00'},
    {'recorded_at':'2000-01-01T00:00:00+00:00'},
])
def test_bad_source_rejected(store,source,change):
    with pytest.raises(ValueError):validate_capture({**source,**change},store.root,tracked)


def test_missing_face_rejected(store,source):
    with pytest.raises(ValueError,match='Exactly one'):
        validate_capture(source,store.root,lambda images,root:[[] for _ in images])


def test_multiple_faces_rejected(store,source):
    with pytest.raises(ValueError,match='Exactly one'):
        validate_capture(source,store.root,lambda images,root:[tracked(images,root)[0]*2 for _ in images])


def test_wrong_geometry_rejected(store,source):
    frame=cv2.imencode('.png',np.zeros((32,32,3),np.uint8))[1]
    source['frames']=['data:image/png;base64,'+base64.b64encode(frame).decode()]*8
    with pytest.raises(ValueError,match='resolution'):validate_capture(source,store.root,tracked)


@pytest.mark.parametrize('video',['data:video/webm;base64,broken','data:video/mp4;base64,AAAA','data:text/html;base64,AAAA'])
def test_corrupt_original_video_rejected(store,source,video):
    with pytest.raises(ValueError):validate_capture({**source,'original_video':video},store.root,tracked)


def test_confirmation_requires_real_prediction(store,source):
    save(store,source)
    with pytest.raises(ValueError,match='actual inference'):store.feedback('capture-01','owner','correct','hello',['hello'])
    assert store.inspect('capture-01')['label'] is None


def test_correct_confirmation_idempotent(store,source):
    save(store,source);predict(store)
    a=store.feedback('capture-01','owner','correct','hello',['hello'])
    b=store.feedback('capture-01','owner','correct','hello',['hello'])
    assert a==b and a['status']=='PENDING_REVIEW'
    assert len([e for e in store.inspect('capture-01')['audit'] if e['action']=='feedback'])==1


def test_correction_keeps_original_prediction(store,source):
    save(store,source);predict(store)
    value=store.feedback('capture-01','owner','incorrect','stop',['hello','stop'])
    assert value['label']=='stop' and value['prediction']['prediction']=='hello'
    with pytest.raises(ValueError,match='already exists'):store.feedback('capture-01','owner','incorrect','next',['next'])


@pytest.mark.parametrize('label',['','A','invented'])
def test_missing_unknown_wrong_task_correction(store,source,label):
    save(store,source);predict(store)
    with pytest.raises(ValueError):store.feedback('capture-01','owner','incorrect',label,['hello','stop'])


def test_unknown_is_quarantined_not_a_class(store,source):
    save(store,source);predict(store)
    row=store.feedback('capture-01','owner','incorrect','UNKNOWN',['hello'])
    assert row['status']=='QUARANTINED'
    assert store.review(row['id'],'reviewer','APPROVED','Reviewed genuine negative for evaluation')['status']=='APPROVED'
    with pytest.raises(ValueError,match='never classifier training'):
        store.freeze('isolated_word',['hello'],{'new-speaker':'train','b':'valid','c':'test'},'reviewer')


def test_uncertain_cannot_be_confirmed(store,source):
    save(store,source);predict(store,accepted=None)
    with pytest.raises(ValueError,match='uncertain'):store.feedback('capture-01','owner','correct','hello',['hello'])
    assert store.feedback('capture-01','owner','incorrect','hello',['hello'])['label']=='hello'


def test_duplicates_and_capture_idempotency(store,source):
    a=save(store,source);assert save(store,source)==a
    b=save(store,source,'capture-02');assert b['status']=='DUPLICATE'
    with pytest.raises(ValueError):store.review('capture-02','reviewer','APPROVED','Not allowed duplicate')
    with pytest.raises(ValueError):save(store,source,owner='different-owner')


def test_decoded_duplicates_with_different_png_encoding(store,source):
    save(store,source)
    frames=[]
    for raw in source['frames']:
        image=cv2.imdecode(np.frombuffer(base64.b64decode(raw.split(',')[1]),np.uint8),1)
        frames.append('data:image/png;base64,'+base64.b64encode(cv2.imencode('.png',image,[cv2.IMWRITE_PNG_COMPRESSION,9])[1]).decode())
    result=save(store,{**source,'frames':frames},'capture-02')
    assert result['status']=='DUPLICATE'


def test_withdraw_deletes_media_and_preserves_audit(store,source):
    save(store,source);predict(store);store.feedback('capture-01','owner','correct','hello',['hello'])
    assert store.withdraw('capture-01','owner')['media_deleted']
    assert store.inspect('capture-01')['prediction']['candidate']=='hello'
    assert 'quality' not in store.inspect('capture-01')['metadata']
    with pytest.raises(LookupError):store.blob('capture-01','media')
    with pytest.raises(ValueError):store.feedback('capture-01','owner','correct','hello',['hello'])


def test_owner_isolation(store,source):
    save(store,source)
    assert not store.list('other')
    for action,args in [(store.inspect,('capture-01','other')),(store.blob,('capture-01','media','other')),
                        (store.withdraw,('capture-01','other'))]:
        with pytest.raises(LookupError):action(*args)


def test_expire_uncommitted(store,source):
    save(store,source)
    with store.db() as db:
        db.execute('UPDATE captures SET created=?',((datetime.now(timezone.utc)-timedelta(days=31)).isoformat(),))
    assert store.expire_uncommitted()['expired']==1
    assert store.stats()['statuses']=={'WITHDRAWN':1}


def test_audit_chain(store,source):
    save(store,source);predict(store);store.feedback('capture-01','owner','correct','hello',['hello'])
    with store.db() as db:events=[dict(r) for r in db.execute('SELECT * FROM events ORDER BY seq')]
    previous='0'*64
    for event in events:
        assert event['previous']==previous
        payload={k:event[k] for k in ('record','actor','action','at','previous')};payload['payload']=json.loads(event['payload'])
        assert event['hash']==digest(canonical(payload));previous=event['hash']


def test_empty_missing_classes_and_speaker_assignment(store,source):
    assignment={'a':'train','b':'valid','c':'test'}
    with pytest.raises(ValueError,match='No approved'):store.freeze('isolated_word',['hello'],assignment,'reviewer')
    approved(store,source,'a-hello','hello','a')
    with pytest.raises(ValueError,match='Every class'):store.freeze('isolated_word',['hello','stop'],assignment,'reviewer')
    with pytest.raises(ValueError,match='independent'):store.freeze('isolated_word',['hello'],{'a':'train'},'reviewer')


def frozen(store,source):
    assignments={'a':'train','b':'valid','c':'test'}
    for speaker in assignments:
        for label in ['hello','stop']:
            for n in range(2):approved(store,source,f'{speaker}-{label}-{n}',label,speaker)
    return store.freeze('isolated_word',['hello','stop'],assignments,'reviewer')


def test_immutable_dataset_and_withdrawn_lineage(store,source):
    version=frozen(store,source)
    assert len(version['records'])==12 and store.version(version['id'])==version
    row=version['record_ids'][0]
    assert store.inspect(row)['status']=='TRAINING_INCLUDED'
    with pytest.raises(ValueError):store.review(row,'reviewer','APPROVED','Cannot change committed history')
    assert store.withdraw(row,'owner')['requires_model_review']
    with pytest.raises(ValueError,match='withdrawn'):store.version(version['id'])


def test_historical_speaker_quarantine(store,source):
    manifest=store.root/'data/processed/multidataset/miracl_v1.jsonl';manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'speaker_id':'a','split':'test'})+'\n')
    approved(store,source,'a-hello','hello','a')
    with pytest.raises(ValueError,match='Historical evaluation speaker'):
        store.freeze('isolated_word',['hello'],{'a':'train','b':'valid','c':'test'},'reviewer')


def test_changed_manifest_hash_rejected(store,source):
    version=frozen(store,source)
    with store.db() as db:
        value={**version,'labels':['invented']};db.execute('UPDATE versions SET payload=?',(json.dumps(value),))
    with pytest.raises(ValueError,match='integrity'):store.version(version['id'])


def test_training_rejects_incompatible_mapping(store,source):
    version=frozen(store,source)
    with pytest.raises(ValueError,match='exact task-specific'):BatchTrainer(store).create(version['id'],'reviewer')


def test_metrics_risk_coverage_and_regression_gate():
    labels=np.arange(8);p=np.eye(8)*.93+.07/8
    report=metrics(p,labels)
    assert report['top1_accuracy']==1 and report['macro_f1']==1
    assert report['risk_coverage'][-1]['coverage']==1 and report['false_acceptance_rate'] is None
    decision=compare(report,report,champion_id='champ',latency_ms=100,memory_bytes=2000,model_size_bytes=500)
    assert not decision['eligible'] and len(decision['reasons'])==2


def test_hard_examples_are_review_only(store,source):
    save(store,source);predict(store);store.feedback('capture-01','owner','incorrect','stop',['stop'])
    row=store.hard_examples()[0]
    assert row['training_weight'] is None and 'large_margin_error' in row['review_reasons']


@pytest.fixture
def client(store,monkeypatch):
    import app.adaptive_routes as routes
    monkeypatch.setenv('LIPNET_FEEDBACK_TOKEN','c'*32);monkeypatch.setenv('LIPNET_REVIEW_TOKEN','r'*32)
    monkeypatch.setattr(routes,'store',store);monkeypatch.setattr(routes,'registry',ModelRegistry(store));monkeypatch.setattr(routes,'trainer',BatchTrainer(store))
    monkeypatch.setattr(routes,'validate_capture',lambda row,root:validate_capture(row,root,tracked))
    def inference(identifier,actor):
        return predict(store,identifier,owner=actor)
    monkeypatch.setattr(routes,'run_prediction',inference)
    app=FastAPI();app.include_router(routes.router);app.add_middleware(routes.PrivateBodyLimit)
    with TestClient(app) as client:yield client


COLLECTOR={'Authorization':'Bearer '+'c'*32}
REVIEWER={'Authorization':'Bearer '+'r'*32}


def test_api_auth_schema_capture_and_feedback(client,source):
    assert client.get('/api/adaptive/stats').status_code==401
    assert client.get('/api/adaptive/config',headers=COLLECTOR).json()['role']=='collector'
    assert client.post('/api/adaptive/captures',json={**source,'predicted_label':'hello'},headers=COLLECTOR).status_code==422
    saved=client.post('/api/adaptive/captures',json=source,headers=COLLECTOR)
    assert saved.status_code==201 and saved.json()['prediction']['prediction']=='hello'
    again=client.post('/api/adaptive/captures',json=source,headers=COLLECTOR)
    assert again.status_code==201
    result=client.post('/api/adaptive/captures/capture-01/feedback',json={'verdict':'incorrect','label':'stop'},headers=COLLECTOR)
    assert result.status_code==200 and result.json()['label']=='stop'
    assert client.get('/api/adaptive/stats',headers=COLLECTOR).json()['corrected']==1


def test_api_roles_private_media_and_withdrawal(client,source):
    client.post('/api/adaptive/captures',json=source,headers=COLLECTOR)
    assert client.get('/api/adaptive/records/capture-01/frames/0').status_code==401
    assert client.get('/api/adaptive/records/capture-01/frames/0',headers=COLLECTOR).headers['content-type']=='image/png'
    assert client.get('/api/adaptive/records/capture-01/frames/900',headers=COLLECTOR).status_code==404
    assert client.post('/api/adaptive/records/capture-01/review',json={'decision':'APPROVED','reason':'test review'},headers=COLLECTOR).status_code==403
    assert client.post('/api/adaptive/captures/capture-01/withdraw',json={},headers=COLLECTOR).status_code==200
    assert client.get('/api/adaptive/records/capture-01/source',headers=COLLECTOR).status_code==404


def test_api_fail_closed_without_keys(client,monkeypatch):
    monkeypatch.delenv('LIPNET_FEEDBACK_TOKEN');monkeypatch.delenv('LIPNET_REVIEW_TOKEN')
    assert client.get('/api/adaptive/config').status_code==503


def test_missing_speaker_and_wrong_task(client,source):
    for change in ({'speaker_id':''},{'task':'continuous_sentence'},{'session_id':'../escape'}):
        assert client.post('/api/adaptive/captures',json={**source,**change},headers=COLLECTOR).status_code==422


def test_training_failure_is_persistent_and_releases_worker(store,monkeypatch):
    trainer=BatchTrainer(store)
    monkeypatch.setattr(store,'version',lambda identifier:{'id':identifier,'task':'isolated_word','labels':TASK_LABELS['isolated_word']})
    finished=__import__('threading').Event()
    def fail(*args):
        try:raise RuntimeError('private secret path must never enter API')
        finally:finished.set()
    monkeypatch.setattr(trainer,'train',fail)
    job=trainer.create('version-fixture','reviewer')['job_id']
    assert finished.wait(5)
    trainer.executor.shutdown(wait=True)
    result=trainer.status(job)
    assert result['status']=='FAILED' and 'secret' not in result['error']
    assert trainer.lock.acquire(blocking=False);trainer.lock.release()


def test_training_capacity_is_bounded(store,monkeypatch):
    trainer=BatchTrainer(store)
    monkeypatch.setattr(store,'version',lambda identifier:{'id':identifier,'task':'isolated_word','labels':TASK_LABELS['isolated_word']})
    trainer.lock.acquire()
    try:
        with pytest.raises(BlockingIOError):trainer.create('fixture','reviewer')
    finally:trainer.lock.release()


def test_challenger_rejection_and_transactional_rollback(store):
    registry=ModelRegistry(store)
    for identifier in ('first','second'):
        path=store.root/(identifier+'.pt');path.write_bytes(identifier.encode())
        model={'id':identifier,'task':'isolated_word','checkpoint':path.name,'checkpoint_sha256':digest(path.read_bytes()),
               'deployment_status':'preserved_research_baseline','comparison':{'eligible':False}}
        with store.db() as db:db.execute('INSERT INTO models VALUES(?,?,?)',(identifier,'isolated_word',json.dumps(model)))
    with store.db() as db:db.execute('INSERT INTO champions VALUES(?,?,?)',('isolated_word','second','first'))
    with pytest.raises(ValueError,match='Explicit reviewer'):registry.deploy('second','reviewer',False)
    with pytest.raises(ValueError,match='lacks passing'):registry.deploy('second','reviewer',True)
    assert registry.rollback('isolated_word','reviewer')['champion']=='first'
    with store.db() as db:
        row=db.execute('SELECT * FROM champions').fetchone();assert row['model']=='first' and row['previous']=='second'
    (store.root/'second.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='integrity'):registry.rollback('isolated_word','reviewer')


def test_global_speaker_assignment_cannot_change_between_versions(store,source):
    with store.db() as db:db.execute('INSERT INTO assignments VALUES(?,?)',('new-speaker','test'))
    approved(store,source,'record-a','hello','new-speaker')
    with pytest.raises(ValueError,match='Speaker leakage'):
        store.freeze('isolated_word',['hello'],{'new-speaker':'train','b':'valid','c':'test'},'reviewer')


def test_private_database_cannot_use_legacy_video_download(tmp_path,monkeypatch):
    import app.main as main
    monkeypatch.setattr(main,'PROJECT_ROOT',tmp_path)
    path=tmp_path/'data/adaptive_private/feedback.sqlite3';path.parent.mkdir(parents=True);path.write_bytes(b'private')
    with TestClient(main.app) as client:
        assert client.get('/api/video/data/adaptive_private/feedback.sqlite3').status_code==404


def test_alphabet_checkpoint_contract_rejects_word_labels():
    from pipeline.adaptive.alphabet_training import load_alphabet
    with pytest.raises(ValueError,match='vocabulary'):load_alphabet({'vocabulary':['hello']})


def test_negative_reviewed_samples_are_separate_from_training(store,source):
    version=frozen(store,source)
    # Freeze a second independent version with new genuine-ID fixtures and an
    # explicitly reviewed negative. No UNKNOWN label becomes a classifier target.
    for speaker in ['d','e','f']:
        for label in ['hello','stop']:
            for n in range(2):approved(store,source,f'{speaker}-{label}-{n}',label,speaker)
    approved(store,source,'negative-1','hello','e')
    with store.db() as db:
        db.execute("UPDATE captures SET label='UNKNOWN',verdict='incorrect' WHERE id='negative-1'")
    second=store.freeze('isolated_word',['hello','stop'],{'d':'train','e':'valid','f':'test'},'reviewer')
    assert second['negative_records'][0]['label']=='UNKNOWN'
    assert all(r['label']!='UNKNOWN' for r in second['records'])


def test_batch_challenger_fits_only_train_valid_and_preserves_champion(store,monkeypatch):
    import torch
    import pipeline.adaptive.training as training
    words=TASK_LABELS['isolated_word']
    checkpoint=store.root/'runs/word_accuracy/avhubert_miracl8_v1/head.pt';checkpoint.parent.mkdir(parents=True)
    head={'format':'lipnet_avhubert_word_linear_v1','vocabulary':words,'feature_layer':9,'pooling':'mean_std',
          'contract':{'vocabulary':words},'mean':torch.zeros(1536),'scale':torch.ones(1536),
          'weight':torch.zeros(8,1536),'bias':torch.zeros(8)}
    torch.save(head,checkpoint);original=digest(checkpoint.read_bytes())
    encoder_path=store.root/'fixture-encoder.pt';encoder_path.write_bytes(b'synthetic encoder identity, never deployed')
    encoder=SimpleNamespace(ckpt_path=encoder_path)
    adapter=SimpleNamespace(service=SimpleNamespace(model=SimpleNamespace(encoder=encoder,head=head)))
    adapter.load=lambda:adapter
    monkeypatch.setattr(training,'baseline_adapter',lambda task:adapter)
    trainer=BatchTrainer(store);opened=[]
    def features(version,split,encoder,directory,negative=False):
        assert split in {'train','valid'} and not negative
        opened.append(split)
        labels=np.tile(np.arange(8),4 if split=='train' else 2)
        rng=np.random.default_rng(1 if split=='train' else 2)
        values=rng.normal(0,.01,(len(labels),29,768)).astype(np.float32)
        for i,label in enumerate(labels):values[i,:,label]+=1
        return values,labels,[1.]*len(labels)
    monkeypatch.setattr(trainer,'features',features)
    result=trainer.train('synthetic-challenger',{'id':'synthetic-version','task':'isolated_word','manifest_sha256':'a'*64})
    assert opened==['train','valid'] and result['model_activated'] is False
    assert digest(checkpoint.read_bytes())==original
    candidate=ModelRegistry(store).model('synthetic-challenger')
    assert candidate['test_report'] is None and candidate['training_configuration']['test_used_for_selection'] is False
    assert not candidate['comparison']['eligible']  # no independent negative evidence
    output=store.root/'runs/adaptive/synthetic-challenger'
    assert all((output/name).is_file() for name in ['head.pt','final.pt','config.json','history.json','comparison.json'])
    with store.db() as db:assert db.execute('SELECT model FROM champions').fetchone()[0]=='avhubert-miracl8-preserved'
    trainer.executor.shutdown(wait=True)


def test_final_test_access_is_one_time_and_exposed_speakers_quarantined(store,source):
    version=frozen(store,source)
    assert store.claim_test(version['id'],'reviewer')['claimed']==4
    with pytest.raises(FileExistsError):store.claim_test(version['id'],'reviewer')
    # Existing train/validation data can grow versions, but exposed test speakers
    # cannot masquerade as a fresh independent test population.
    with pytest.raises(ValueError,match='Historical evaluation speaker'):
        store.freeze('isolated_word',['hello','stop'],{'a':'train','b':'valid','c':'test'},'reviewer')


def test_temporal_agreement_must_match_full_window_candidate(monkeypatch):
    import pipeline.adaptive.training as training
    import pipeline.word_recognition.consensus as consensus
    p=np.zeros(8);p[2]=.99;p[0]=.01
    monkeypatch.setattr(training,'head_probabilities',lambda head,features:p)
    monkeypatch.setattr(consensus,'temporal_consensus',lambda *args:{'confirmed':True,'views':[{'candidate':'stop'}]})
    assert not training.word_acceptance({},np.zeros((29,768)))
    monkeypatch.setattr(consensus,'temporal_consensus',lambda *args:{'confirmed':True,'views':[{'candidate':'hello'}]})
    assert training.word_acceptance({},np.zeros((29,768)))
