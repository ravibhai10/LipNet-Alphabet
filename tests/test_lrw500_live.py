"""Capture, isolation and rejection contract tests; fixtures are not accuracy data."""
import base64
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from pipeline.lrw_tcn.preprocessing import align_sequence,resample_timed,prepare_webcam,MP68,WEB_CONVENTION,REFERENCE
from pipeline.lrw_tcn.diagnostics import Measurement
from app.services.lrw500_inference import LRW500Inference


def test_reference_mapping_and_exact_identity_geometry():
    assert len(MP68)==68 and len(set(MP68))==68
    reference=np.load(REFERENCE)
    # Genuine publisher geometry used as an identity transform fixture.
    yy,xx=np.mgrid[:256,:256];image=((xx+yy)%256).astype(np.uint8)
    crops,meta=align_sequence([image]*12,[reference/256]*12,convention='publisher-68-v1')
    x,y=np.round(reference[48:68].mean(0)).astype(int)
    np.testing.assert_array_equal(crops[0],image[y-48:y+48,x-48:x+48])
    assert meta['alignment_equivalence_verified']
    assert align_sequence([image]*12,[reference/256]*12)[1]['alignment_equivalence_verified'] is False


def test_timestamp_sampling_preserves_real_duration_and_rejects_gaps():
    clip=np.repeat(np.arange(15,dtype=np.uint8)[:,None,None],96,axis=1).repeat(96,axis=2)
    sampled,meta=resample_timed(clip,np.arange(15)*66.6667)
    assert len(sampled)==24 and meta['measured_capture_hz']==15
    assert meta['target_hz']==25 and meta['source_duration_ms']==pytest.approx(933.33,abs=.01)
    for stamps in ([0]*15,np.arange(15)*200,np.arange(15)*10):
        with pytest.raises(ValueError):resample_timed(clip,stamps)


@pytest.mark.parametrize('count',[0,2])
def test_face_count_gate(count):
    with pytest.raises(ValueError,match='Exactly one face'):
        prepare_webcam([np.ones((256,256),np.uint8)]*12,[],np.arange(12)*40,[count]*12)


def test_lighting_and_static_mouth_are_rejected_before_model():
    reference=np.load(REFERENCE)/256
    with pytest.raises(ValueError,match='Lighting'):
        prepare_webcam([np.zeros((256,256),np.uint8)]*12,[reference]*12,np.arange(12)*40,[1]*12)
    rng=np.random.default_rng(4)
    with pytest.raises(ValueError,match='mouth movement'):
        prepare_webcam([rng.integers(0,255,(256,256),dtype=np.uint8)]*12,[reference]*12,np.arange(12)*40,[1]*12)


def test_lrw_health_does_not_inherit_miracl_accuracy_or_load_by_default():
    instance=LRW500Inference();report=instance.health()
    assert len(report['vocabulary'])==500 and report['held_out_accuracy'] is None
    assert report['state']=='not_loaded' and not report['inference_ready']
    assert report['audio_inference'] is report['production_ready'] is False


def test_unknown_threshold_configuration_and_cpu_backpressure(monkeypatch):
    monkeypatch.setenv('LIPNET_LRW_CONFIDENCE_THRESHOLD','1.1')
    with pytest.raises(ValueError,match='thresholds'):LRW500Inference()
    monkeypatch.delenv('LIPNET_LRW_CONFIDENCE_THRESHOLD')
    instance=LRW500Inference();instance._predict_lock.acquire()
    try:
        with pytest.raises(BlockingIOError):instance.predict([])
    finally:instance._predict_lock.release()


def test_service_uses_actual_model_vector_and_separates_uncertainty():
    instance=LRW500Inference()
    class Fake:
        vocabulary=instance.words
        def predict_rois(self,rois,**settings):
            assert rois.shape==(12,96,96) and settings['threshold']==.7
            return {'candidate':self.vocabulary[0],'word':None,'confidence':.01,'uncertain':True,'margin':0,
                    'top5':[{'word':w,'probability':.002} for w in self.vocabulary[:5]],
                    'probabilities':[.002]*500,'model':'fixture only','checkpoint_sha256':'fixture'}
    instance.model=Fake();instance.state='ready'
    ok,jpeg=cv2.imencode('.jpg',np.full((96,96),128,np.uint8));assert ok
    encoded=base64.b64encode(jpeg).decode()
    result=instance.predict([encoded]*12,input_kind='aligned_mouth',convention='caller-aligned-mouth-v1')
    assert result['word'] is None and result['display_word']=='Uncertain'
    assert len(result['top_predictions'])==5 and len(result['all_probabilities'])==500
    assert result['local_accuracy'] is None and result['performance']['wall_ms']>0


def test_measurements_are_measured_not_constants():
    measure=Measurement();np.linalg.norm(np.ones((300,300)))
    data=measure.finish()
    assert data['wall_ms']>0 and data['process_cpu_ms']>=0
    assert data['requests_per_second_sequential']>0


def test_http_contract_rejects_wrong_convention_and_shared_cpu_contention(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app import lrw500_routes as routes
    app=FastAPI();app.include_router(routes.router)
    monkeypatch.setattr(routes,'lrw500_inference',LRW500Inference())
    _,jpeg=cv2.imencode('.jpg',np.full((96,96),128,np.uint8))
    payload={'frames':[base64.b64encode(jpeg).decode()]*12,'convention':'wrong'}
    with TestClient(app) as client:
        assert client.post('/api/predict/word/lrw500',json=payload).status_code==422
        routes.word_cpu_lock.acquire()
        try:assert client.post('/api/predict/word/lrw500',json=payload).status_code==429
        finally:routes.word_cpu_lock.release()


@pytest.mark.parametrize('agrees',[True,False])
def test_completed_utterance_uses_distinct_real_temporal_views(monkeypatch,agrees):
    from app.services import lrw500_inference as service
    instance=LRW500Inference();views=[]
    class Fake:
        vocabulary=instance.words
        def predict_rois(self,rois,**settings):
            views.append(len(rois));candidate=self.vocabulary[0 if agrees or len(views)==1 else 1]
            return {'candidate':candidate,'word':candidate,'confidence':.95,'uncertain':False,'margin':.9,
                    'top5':[{'word':w,'probability':.002} for w in self.vocabulary[:5]],'probabilities':[.002]*500,
                    'model':'fixture','checkpoint_sha256':'fixture'}
    instance.model=Fake();instance.state='ready'
    sequence=np.arange(29*96*96,dtype=np.uint8).reshape(29,96,96)
    monkeypatch.setattr(service,'prepare_webcam',lambda *args:(sequence,{}))
    _,jpeg=cv2.imencode('.jpg',sequence[0]);encoded=base64.b64encode(jpeg).decode()
    result=instance.predict([encoded]*29,convention=WEB_CONVENTION,final=True)
    assert views==[29,27] and result['diagnostics']['temporal_consensus']['confirmed'] is agrees
    assert result['uncertain'] is (not agrees)
    assert (result['word'] is not None) is agrees
