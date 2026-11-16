"""Synthetic plumbing fixtures only; their classifier metrics are not speech results."""
import base64
import hashlib
import json
import numpy as np
import cv2
import pytest
import torch
from pipeline.word_recognition import personal_data as data
from pipeline.word_recognition import personalized as personal
from pipeline.word_recognition.common import sha256,write_rows,atomic_json


def clip(index=1):
    frames=[]
    for i in range(8):
        image=np.full((96,96),30+index,np.uint8);image[:,(index+i)%96]=190
        ok,jpeg=cv2.imencode('.jpg',image);assert ok
        frames.append('data:image/jpeg;base64,'+base64.b64encode(jpeg).decode())
    return {'id':f'clip-{index}','label':'hello','speaker_id':'participant01','session_id':'train-session','split':'train',
            'frames':frames,'timestamps_ms':[i*66 for i in range(8)],'duration_ms':462,'recorded_at':'2026-10-30T12:00:00+00:00',
            'preprocessing':'browser-eyeaxis96-jpeg-v1','landmarker_sha256':'a'*64,'source':'user_camera',
            'consent':True,'label_confirmed':True,'frames_sha256':hashlib.sha256(json.dumps(frames,separators=(',',':')).encode()).hexdigest()}


def test_browser_jpeg_import_keeps_exact_pixels_and_target_geometry():
    row=clip();images=data.validate_clip(row,'a'*64)
    assert images.shape==(8,96,96) and images.dtype==np.uint8
    assert np.array_equal(images[0],cv2.imdecode(np.frombuffer(base64.b64decode(row['frames'][0].split(',')[1]),np.uint8),0))


@pytest.mark.parametrize('change',[{'label':'bye'},{'consent':False},{'label_confirmed':False},{'source':'model_prediction'},
    {'landmarker_sha256':'b'*64},{'preprocessing':'upstream-22frames'},{'frames_sha256':'0'*64},
    {'timestamps_ms':[0]*8},{'duration_ms':5001},{'recorded_at':'not-a-date'},{'id':'../bad'}])
def test_invalid_or_unconfirmed_collection_fails_before_training(change):
    with pytest.raises((ValueError,TypeError)):data.validate_clip({**clip(),**change},'a'*64)


def rows():
    return [{'id':f'{s}-{w}','label':w,'speaker_id':'person','session_id':s+'-session','split':s,
             'roi_sha256':hashlib.sha256((s+w).encode()).hexdigest()} for s in ['train','valid','test'] for w in data.WORDS]


def test_session_disjoint_personal_protocol_never_claims_unseen_speakers():
    report=data.audit_partitions(rows(),{'train':1,'valid':1,'test':1})
    assert report['speaker_independent'] is False and report['sessions']==3
    with pytest.raises(ValueError,match='More confirmed'):data.audit_partitions(rows())
    r=rows();r[-1]['session_id']='train-session'
    with pytest.raises(ValueError,match='Session leakage'):data.audit_partitions(r,{'train':1,'valid':1,'test':1})


def test_duplicate_decoded_content_and_missing_people_or_classes_fail():
    r=rows();r[-1]['roi_sha256']=r[0]['roi_sha256']
    with pytest.raises(ValueError,match='Duplicate'):data.audit_partitions(r,{'train':1,'valid':1,'test':1})
    r=rows();r[-1]['speaker_id']='unseen-person'
    with pytest.raises(ValueError,match='same participant'):data.audit_partitions(r,{'train':1,'valid':1,'test':1})
    with pytest.raises(ValueError,match='More confirmed'):data.audit_partitions(rows()[:-1],{'train':1,'valid':1,'test':1})


def test_quarantined_external_geometry_and_fixture_exports_are_not_training(tmp_path,monkeypatch):
    asset=tmp_path/'frontend/public/models/face_landmarker.task';asset.parent.mkdir(parents=True);asset.write_bytes(b'fixture asset')
    monkeypatch.setattr(data,'ROOT',tmp_path)
    bundle=tmp_path/'bundle.json';bundle.write_text(json.dumps({'format':'lipnet-word-collection-v1','protocol':'personal_cross_session','vocabulary':data.WORDS,'qa_fixture':True,'clips':[]}))
    with pytest.raises(ValueError,match='actual eight-word'):data.ingest([bundle],tmp_path/'run')
    r=clip();ok,jpeg=cv2.imencode('.jpg',np.ones((80,112),np.uint8));assert ok
    r['frames']=['data:image/jpeg;base64,'+base64.b64encode(jpeg).decode()]*8
    r['frames_sha256']=hashlib.sha256(json.dumps(r['frames'],separators=(',',':')).encode()).hexdigest()
    with pytest.raises(ValueError,match='96'):data.validate_clip(r,'a'*64)


def test_actual_bundle_ingestion_coalesces_exports_and_preserves_integrity(tmp_path,monkeypatch):
    root=tmp_path/'workspace';asset=root/'frontend/public/models/face_landmarker.task';asset.parent.mkdir(parents=True);asset.write_bytes(b'fixture')
    encoder=root/'encoder.pt';encoder.write_bytes(b'fixture encoder')
    base=root/'baseline';base.mkdir();(base/'head.pt').write_bytes(b'fixture head')
    (base/'rejection_policy.json').write_text('{}');(base/'features.jsonl').write_text('')
    monkeypatch.setattr(data,'ROOT',root);monkeypatch.setattr(data,'BASE_RUN',base);monkeypatch.setattr(data,'ENCODER',encoder)
    monkeypatch.setattr(data,'QUOTAS',dict(train=1,valid=1,test=1))
    recordings=[]
    for split in ['train','valid','test']:
        for word in data.WORDS:
            r=clip(len(recordings)+1);r.update(label=word,split=split,session_id=split+'-session',landmarker_sha256=sha256(asset));recordings.append(r)
    bundle=root/'recordings.json';bundle.write_text(json.dumps({'format':'lipnet-word-collection-v1','protocol':'personal_cross_session','vocabulary':data.WORDS,'clips':recordings}))
    protocol=data.ingest([bundle,bundle],root/'run')
    assert protocol['audit']['sessions']==3
    loaded,rows=data.load_contract(root/'run');assert len(rows)==24 and loaded==protocol
    assert np.load(rows[0]['roi'],allow_pickle=False).shape==(8,96,96)
    with pytest.raises(FileExistsError):data.ingest([bundle],root/'run')
    (root/'run/recordings.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='changed'):data.load_contract(root/'run')


def fixture_run(tmp_path,monkeypatch):
    run=tmp_path/'personal';run.mkdir();rng=np.random.default_rng(42)
    head={'format':'lipnet_avhubert_word_linear_v1','feature_layer':9,'pooling':'mean_std','vocabulary':data.WORDS,
          'mean':torch.zeros(1536),'scale':torch.ones(1536),'weight':torch.zeros(8,1536),'bias':torch.zeros(8),
          'contract':{'vocabulary':data.WORDS,'preprocessing':data.PREPROCESS},'audio_inference':False}
    baseline=tmp_path/'baseline.pt';torch.save(head,baseline)
    atomic_json(tmp_path/'rejection_policy.json',{'head_sha256':sha256(baseline),'threshold':.6,'margin':.1})
    original=[];recordings=[]
    for index,word in enumerate(data.WORDS):
        for split in ['base','train','valid','test']:
            value=rng.normal(0,.01,(29,768)).astype(np.float32);value[:,index]+=5
            target=run/f'{word}-{split}.npy';np.save(target,np.stack([value]*3) if split=='base' else value)
            row={'id':f'{word}-{split}','feature':str(target),'feature_sha256':sha256(target),'word_label':word,
                 'label':word,'split':split,'speaker_id':'person','session_id':split+'-session','role':'known_word','usable':True}
            if split=='base':row['split']='train';original.append(row)
            else:recordings.append(row)
    write_rows(run/'features.jsonl',recordings);write_rows(run/'recordings.jsonl',recordings)
    write_rows(run/'base-features.jsonl',original);atomic_json(run/'personal_protocol.json',{'fixture':True})
    contract={'baseline_head':str(baseline),'baseline_head_sha256':sha256(baseline),'base_training_features':str(run/'base-features.jsonl'),
              'audit':{'participants':['person']}}
    monkeypatch.setattr(personal,'load_contract',lambda _: (contract,recordings))
    return run,recordings,baseline


def test_real_head_fit_selection_and_paired_metrics_plumbing_without_test_fitting(tmp_path,monkeypatch):
    run,recordings,baseline=fixture_run(tmp_path,monkeypatch)
    # Poisoned historical test paths must never be read by fitting.
    with (run/'base-features.jsonl').open('a') as stream:
        stream.write(json.dumps({'split':'test','role':'known_word','usable':True,'feature':'DO_NOT_READ','word_label':'hello'})+'\n')
    protected=sha256(baseline);opened=[];original=personal.feature
    def traced(row):opened.append(row['split']);return original(row)
    monkeypatch.setattr(personal,'feature',traced)
    selection=personal.train(run)
    assert selection['test_used'] is False and 'test' not in opened
    assert len(selection['candidates'])==10 and sha256(baseline)==protected
    with pytest.raises(FileExistsError):personal.train(run)
    report=personal.evaluate(run)
    assert report['test_samples']==8 and report['speaker_independent'] is False
    assert report['eligible_for_manual_research_activation'] is False  # Eight fixtures do not pass the real gate.
    assert (run/'paired_confusion.svg').is_file() and (run/'test_predictions.jsonl').is_file()
    assert all(row['total']==1 for row in report['per_word'].values())
    with pytest.raises(FileExistsError):personal.evaluate(run)
    assert sha256(baseline)==protected


def test_changed_feature_and_frozen_selection_are_rejected(tmp_path,monkeypatch):
    run,recordings,_=fixture_run(tmp_path,monkeypatch)
    personal.train(run)
    (run/'features.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='Frozen'):personal.evaluate(run)
    assert not (run/'test_access.json').exists()


def test_metric_counts_top5_and_matrix_axes_are_derived_from_predictions():
    p=np.eye(8)*.9+.1/8;y=np.arange(8)
    result=personal.metrics(p,y)
    assert result['correct']==8 and result['top5_accuracy']==1 and result['confusion_matrix']==np.eye(8,dtype=int).tolist()
    assert sum(r['correct'] for r in result['per_word'].values())==8


def test_feature_extraction_resumes_verified_cache_and_rejects_mutation(tmp_path,monkeypatch):
    from pipeline.avhubert import loader
    root=tmp_path/'workspace';asset=root/'frontend/public/models/face_landmarker.task';asset.parent.mkdir(parents=True);asset.write_bytes(b'fixture landmarks')
    encoder=root/'encoder.pt';encoder.write_bytes(b'fixture encoder');run=root/'run';run.mkdir()
    roi=run/'roi.npy';np.save(roi,np.ones((8,96,96),np.uint8))
    row={'id':'fixture','roi':str(roi),'roi_file_sha256':sha256(roi)}
    contract={'encoder_sha256':sha256(encoder),'landmarker_sha256':sha256(asset)}
    monkeypatch.setattr(data,'ROOT',root);monkeypatch.setattr(data,'ENCODER',encoder)
    monkeypatch.setattr(data,'load_contract',lambda _: (contract,[row]))
    class Fake:
        calls=0
        def __init__(self,**kwargs):assert kwargs=={'threads':1,'low_memory':True}
        def load(self):return self
        def encode_frames(self,images,**kwargs):
            assert kwargs=={'source_fps':None,'target_frames':29,'output_layer':9}
            Fake.calls+=1
            return np.ones((29,768),np.float32)
    monkeypatch.setattr(loader,'AVHubertLoader',Fake)
    data.extract(run);data.extract(run)
    assert Fake.calls==1
    manifest=[json.loads(s) for s in (run/'features.jsonl').read_text().splitlines()]
    assert np.load(manifest[0]['feature'],allow_pickle=False).shape==(29,768)
    np.save(manifest[0]['feature'],np.zeros((29,768),np.float32))
    with pytest.raises(ValueError,match='Cached'):data.extract(run)
