"""Word API lifecycle/exception fixtures. These are not recognition measurements."""
import importlib
import base64
import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.services.browser_inference import BrowserInference


@pytest.fixture(params=['word', 'lrw500'])
def route(request, monkeypatch):
    # A small router harness avoids startup/background jobs and preserves active services.
    if request.param == 'word':
        module = importlib.import_module('app.main')
        endpoint, handler, schema, service = '/api/predict/word', module.predict_word, module.PredictRequest, 'word_inference'
        payload = {'frames': ['fixture'] * 12}
    else:
        module = importlib.import_module('app.lrw500_routes')
        endpoint, handler, schema, service = '/api/predict/word/lrw500', module.predict_lrw500, module.LRW500Request, 'lrw500_inference'
        payload = {'frames': ['fixture'] * 12, 'convention': 'caller-aligned-mouth-v1', 'input_kind': 'aligned_mouth', 'final': True}
    app = FastAPI()
    app.post(endpoint)(handler)
    class Fake:
        result = {'task': 'isolated_word', 'predicted_word': 'hello', 'display_word': 'Uncertain',
                  'word': None, 'uncertain': True, 'confidence': None, 'top_predictions': [], 'all_probabilities': []}
        error = None
        def predict(self, frames, **kwargs):
            if self.error:
                raise self.error
            return dict(self.result)
    fake = Fake()
    monkeypatch.setattr(module, service, fake)
    with TestClient(app) as client:
        yield client, endpoint, payload, fake, module.word_cpu_lock


def test_valid_response_preserves_uncertainty_and_unavailable_confidence(route):
    client, endpoint, payload, _, lock = route
    response = client.post(endpoint, json=payload)
    assert response.status_code == 200
    assert response.json()['confidence'] is None
    assert response.json()['word'] is None and response.json()['uncertain'] is True
    assert not lock.locked()


@pytest.mark.parametrize('frames', [None, [], ['bad'] * 7, ['bad'] * 77])
def test_empty_missing_and_bad_length_clips_fail_schema(route, frames):
    client, endpoint, payload, _, lock = route
    if frames is None:
        payload.pop('frames')
    else:
        payload['frames'] = frames
    assert client.post(endpoint, json=payload).status_code == 422
    assert not lock.locked()


@pytest.mark.parametrize('error,code', [(ValueError('JPEG input invalid'), 422), (BlockingIOError('Busy'), 429),
                                      (RuntimeError('private checkpoint path'), 503), (KeyError('private filesystem path'), 500)])
def test_inference_errors_release_lock_and_internal_errors_are_sanitized(route, error, code):
    client, endpoint, payload, fake, lock = route
    fake.error = error
    response = client.post(endpoint, json=payload)
    assert response.status_code == code
    if code in (500, 503):
        assert 'private' not in response.text and 'Traceback' not in response.text
    assert not lock.locked()
    fake.error = None
    assert client.post(endpoint, json=payload).status_code == 200


def test_parallel_word_request_returns_busy_without_queueing(route):
    client, endpoint, payload, _, lock = route
    lock.acquire()
    try:
        assert client.post(endpoint, json=payload).status_code == 429
    finally:
        lock.release()


@pytest.mark.parametrize('frame', ['not-base64', base64.b64encode(b'not a JPEG').decode(), 'x' * 200001], ids=['base64', 'not-jpeg', 'oversized'])
def test_actual_decoder_rejects_invalid_format_and_oversize(frame):
    with pytest.raises(ValueError):
        BrowserInference.decode_jpegs([frame] * 8)


def test_actual_decoder_rejects_oversized_geometry():
    ok, jpeg = cv2.imencode('.jpg', np.ones((2050, 96), dtype=np.uint8))
    assert ok
    with pytest.raises(ValueError):
        BrowserInference.decode_jpegs([base64.b64encode(jpeg).decode()] * 8)


def test_health_does_not_publish_model_loader_paths(monkeypatch, tmp_path):
    from app.services.pretrained_word_inference import PretrainedWordInference
    from app.services.lrw500_inference import LRW500Inference
    from app.services.word_inference import WordInferenceService
    services = [PretrainedWordInference(tmp_path / 'private-head.pt'), LRW500Inference(), WordInferenceService()]
    for service in services:
        monkeypatch.setattr(service, 'load', lambda: False)
        service.error = str(tmp_path / 'private-checkpoint.pt')
        report = service.health()
        assert 'private-checkpoint' not in report['error']
        assert not report['inference_ready']
