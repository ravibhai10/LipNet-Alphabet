"""Research activation and request integrity fixtures; not recognition metrics."""
import json
import pytest
from app.services import pretrained_word_inference as service
from pipeline.word_recognition.common import sha256


@pytest.mark.parametrize('accuracy,recall,expected',[(.69,.9,False),(.95,.1,False),(.95,.9,True)])
def test_research_service_requires_held_out_evidence(tmp_path,monkeypatch,accuracy,recall,expected):
    head=tmp_path/'head.pt';head.write_bytes(b'fixture head')
    report={'head_sha256':sha256(head),'test_samples':192,'top1_accuracy':accuracy,'per_word':{'one':{'recall':recall}}}
    (tmp_path/'evaluation.json').write_text(json.dumps(report))
    class Fake:
        def __init__(self,*args):self.head={'vocabulary':['one','two']}
        def load(self):return self
    monkeypatch.setattr(service,'AVHubertWordClassifier',Fake)
    instance=service.PretrainedWordInference(head)
    assert instance.load() is expected
    assert instance.health()['research_activation_gate_passed'] is expected
    assert instance.health()['production_ready'] is False


def test_word_service_never_queues_cpu_requests(tmp_path):
    instance=service.PretrainedWordInference(tmp_path/'absent.pt')
    instance._predict_lock.acquire()
    try:
        with pytest.raises(BlockingIOError):instance.predict([])
    finally:instance._predict_lock.release()
    assert instance.health()['inference_ready'] is False
