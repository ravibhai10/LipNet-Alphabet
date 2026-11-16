"""API schema checks; test metadata is not a recognition evaluation."""
import json
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.services import recognition_capabilities as capabilities


def test_capability_schema_fails_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(capabilities,'ROOT',tmp_path)
    app=FastAPI();app.include_router(capabilities.router)
    value=TestClient(app).get('/api/recognition/capabilities').json()
    assert value['schema_version']==1 and value['audio_inference'] is False
    assert value['video_to_text']['live_ready'] is False
    assert value['isolated_word']['live_ready'] is False
    assert value['isolated_word']['evaluation'] is None
    assert value['isolated_word']['vocabulary']==[]


def test_offline_evidence_is_not_live_readiness(tmp_path,monkeypatch):
    monkeypatch.setattr(capabilities,'ROOT',tmp_path)
    path=tmp_path/'runs/multidataset/miracl8_cpu_v2/evaluation.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'vocabulary':['fixture'],'test_samples':1,'top1_accuracy':1.,'top5_accuracy':1.,'checkpoint_sha256':'a'*64}))
    value=capabilities.recognition_capabilities()
    assert value['isolated_word']['status']=='offline_research_evaluated'
    assert value['isolated_word']['live_ready'] is False
    assert value['isolated_word']['production_ready'] is False


def test_registered_sequence_track_is_unavailable():
    from app.main import list_available_models
    assert list_available_models()['tracks']['video_to_text']['ready'] is False


@pytest.mark.parametrize('record_vocabulary',[True,False])
def test_existing_word_service_requires_recorded_mapping(tmp_path,monkeypatch,record_vocabulary):
    import torch
    from app.services import word_inference as service
    from pipeline.word_recognition.model import WordVideoNet
    path=tmp_path/'fixture.pt';path.write_bytes(b'local software fixture')
    artifact={'model_state_dict':WordVideoNet(2).state_dict(),'model_config':{'channels':[8,16,32],'hidden_size':48,'layers':2,'dropout':.15}}
    if record_vocabulary:artifact['vocabulary']=['red','green']
    monkeypatch.setattr(service,'CHECKPOINT_PATH',path)
    monkeypatch.setattr(service.torch,'load',lambda *args,**kwargs:artifact)
    instance=service.WordInferenceService()
    assert instance.load() is record_vocabulary
    if record_vocabulary:assert instance.vocabulary==['red','green']
    else:assert instance.vocabulary==[]
