"""Integrity/abstention fixtures only; no fixture measures speech accuracy."""
from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.services.recording_intake import save_bundle
from app.services.temporal_buffer import PredictionStabilizer, TemporalBuffer, BufferedFrame
from pipeline.word_recognition.common import sha256
from tests.test_personal_words import clip
from scripts.audit_recognition_dataset import audit


def test_stabilizer_rejection_clears_candidate_and_never_counts_interrupted_evidence():
    stab=PredictionStabilizer(required_count=3)
    assert stab.update('A',.9,probability_margin=.3) is None
    assert stab.update('A',.9,probability_margin=.01) is None
    assert stab.current is None
    for _ in range(2): assert stab.update('A',.9,probability_margin=.3) is None
    assert stab.update('A',.9,probability_margin=.3)=='A'
    assert stab.update('A',.9,probability_margin=.3) is None  # no duplicate event
    for score in [float('nan'),float('inf'),-.1,.2,1.1]:
        stab.update('A',score); assert stab.current is None


def test_buffer_rejects_stale_frames_and_reports_oldest_age():
    buffer=TemporalBuffer(max_frames=3,min_frames=1)
    make=lambda n,t:BufferedFrame(n,t,np.zeros(192))
    assert buffer.append(make(1,10))
    assert buffer.append(make(2,12))
    assert not buffer.append(make(3,11))
    assert buffer.stats(15).age_s==5


@pytest.fixture
def collection_root(tmp_path):
    asset=tmp_path/'frontend/public/models/face_landmarker.task'
    asset.parent.mkdir(parents=True);asset.write_bytes(b'landmark fixture')
    row=clip();row['landmarker_sha256']=sha256(asset)
    return tmp_path,row


def test_personal_intake_saves_real_bytes_but_never_invents_or_activates_models(collection_root):
    root,row=collection_root
    result=save_bundle(root,[row],['hello','help'])
    assert result['model_activated'] is False and result['success'] is False
    assert result['recorded_words']==['hello'] and result['missing_recordings']==['help']
    bundle=root/'data/personal_recordings'/(result['bundle_id']+'.json')
    assert json.loads(bundle.read_text())['clips']==[row]
    assert sha256(bundle)==result['bundle_sha256']
    assert not (root/'runs').exists()
    second=save_bundle(root,[row],['hello','help'])
    assert second['bundle_id'] != result['bundle_id'] and bundle.is_file()


def test_intake_rejects_duplicate_pixels_wrong_labels_and_session_leakage(collection_root):
    root,row=collection_root
    copied={**row,'id':'copy'}
    with pytest.raises(ValueError,match='Duplicate'): save_bundle(root,[row,copied],['hello'])
    with pytest.raises(ValueError,match='matching'): save_bundle(root,[row],['help'])
    second=clip(2); second.update(landmarker_sha256=row['landmarker_sha256'],split='test')
    with pytest.raises(ValueError,match='Session leakage'): save_bundle(root,[row,second],['hello'])
    assert not (root/'data/personal_recordings').exists()


def test_personal_api_reports_pending_validation_and_preserves_active_service(collection_root,monkeypatch):
    root,row=collection_root;monkeypatch.setattr(main,'PROJECT_ROOT',root)
    service=main.word_inference
    with TestClient(main.app) as client:
        result=client.post('/api/word/train-personal',json={'clips':[row],'vocabulary':['hello']})
        assert result.status_code==202 and result.json()['model_activated'] is False
        assert client.post('/api/word/train-personal',json={'clips':[]}).status_code==422
    assert main.word_inference is service


def test_dataset_audit_reports_missing_source_duplicate_and_speaker_leakage(tmp_path):
    rows=[{'id':'same','video':'missing.mp4','label':'A','speaker_id':'s1','split':s,'video_sha256':'a'*64}
          for s in ['train','test']]
    manifest=tmp_path/'input.jsonl';manifest.write_text('\n'.join(map(json.dumps,rows)))
    result=audit(manifest,decode=True)
    assert result['corrupted_or_invalid']==2 and result['source_decoded_samples']==0
    assert result['speaker_leakage']=={'s1':['test','train']}
    assert result['duplicate_ids']=={'same':2} and len(result['cross_split_source_duplicates'])==1


def test_temporal_experiment_refuses_test_partition_before_opening_files():
    from pipeline.word_recognition.temporal_experiment import load_partition
    with pytest.raises(ValueError,match='test features'):
        load_partition([{'feature':'DO_NOT_OPEN'}],'test',['hello'],9)


def test_rejected_pretrained_response_has_no_accepted_word(monkeypatch):
    from types import SimpleNamespace
    from app.services.pretrained_word_inference import PretrainedWordInference
    service=PretrainedWordInference()
    fake={'word':None,'candidate':'hello','probabilities':[.55,.45],
          'vocabulary':['hello','stop'],'uncertain':True,'confidence':.55,'margin':.1}
    service.model=SimpleNamespace(predict_rois=lambda _:fake,policy={'threshold':.95})
    monkeypatch.setattr(service,'load',lambda:True)
    monkeypatch.setattr(main.browser_inference.__class__,'decode_jpegs',lambda _: [np.zeros((96,96),np.uint8)]*8)
    result=service.predict(['fixture']*8)
    assert result['word'] is None and result['predicted_word']=='hello'
    assert result['display_word']=='UNCERTAIN — PLEASE TRY AGAIN'


def test_common_api_fields_preserve_candidate_but_abstain():
    from app.services.prediction_contract import structured_prediction
    result=structured_prediction({'uncertain':True,'predicted_word':'hello','model_name':'test',
                                  'vocabulary':['hello','stop'],'top_predictions':[{'word':'hello','prob':.6}]},task='word_recognition')
    assert result['prediction'] is None and result['status']=='uncertain'
    assert result['confidence_calibrated'] is False and len(result['vocabulary_id'])==64


def test_temporal_consensus_requires_all_windows_to_agree():
    import torch
    from pipeline.word_recognition.consensus import temporal_consensus
    weight=torch.zeros((2,768));weight[0,0]=1;weight[1,0]=-1
    head={'pooling':'mean','vocabulary':['hello','stop'],'mean':torch.zeros(768),
          'scale':torch.ones(768),'weight':weight,'bias':torch.zeros(2)}
    features=np.ones((29,768),dtype=np.float32)*4
    result=temporal_consensus(head,features,{'threshold':.8,'margin':.2})
    assert result['confirmed'] is True and len(result['views'])==3
    features[:15,0]=6;features[15:,0]=-6
    assert temporal_consensus(head,features,{'threshold':.5,'margin':0})['confirmed'] is False
    assert result['independent_examples'] is False


def test_service_uses_frozen_evaluation_policy_without_overwriting_files(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from app.services.pretrained_word_inference import PretrainedWordInference
    path=tmp_path/'head.pt';path.write_bytes(b'fixture')
    policy={'threshold':.95,'margin':.2}
    (tmp_path/'evaluation.json').write_text(json.dumps({'head_sha256':sha256(path),'test_samples':192,
        'top1_accuracy':.95,'per_word':{'hello':{'recall':.9}},'rejection_policy':policy}))
    instance=PretrainedWordInference(path)
    model=SimpleNamespace(policy={'threshold':.6,'margin':.1},head={'vocabulary':['hello','stop']})
    monkeypatch.setattr(instance,'_make_classifier',lambda:SimpleNamespace(load=lambda:model))
    assert instance.load()
    assert model.policy==policy and instance.health()['policy_file_differs_from_evaluation'] is True
    assert not (tmp_path/'rejection_policy.json').exists()


def test_http_timeout_discards_late_response_and_keeps_cpu_backpressure(monkeypatch):
    import threading
    release=threading.Event()
    class Slow:
        def predict(self, frames):
            release.wait(2)
            return {'task':'isolated_word','predicted_word':'hello','uncertain':False,'top_predictions':[]}
    monkeypatch.setattr(main,'word_inference',Slow())
    monkeypatch.setattr(main,'INFERENCE_TIMEOUT_SECONDS',.02)
    with TestClient(main.app) as client:
        try:
            result=client.post('/api/predict/word',json={'frames':['fixture']*8})
            assert result.status_code==504
            assert main.word_cpu_lock.locked()
            assert client.post('/api/predict/word',json={'frames':['fixture']*8}).status_code==429
        finally:release.set()


def test_alphabet_training_rejects_wrong_sequence_geometry(tmp_path):
    from scripts.train_alphabet import AlphabetDataset
    path=tmp_path/'bad.npy';np.save(path,np.zeros((8,96,96),dtype=np.uint8))
    dataset=AlphabetDataset(tmp_path,[(path,0)])
    with pytest.raises(ValueError,match='geometry'):dataset[0]
