"""Author-model compatibility tests using tiny local fixtures, not LRW metrics."""
import json
import numpy as np
import pytest
import torch
from pipeline.lrw_tcn import inference as module
from pipeline.word_recognition.common import sha256


def test_publisher_labels_match_official_existing_vocabulary():
    assert sha256(module.PACKAGE/'vocabulary.txt')==sha256(module.ROOT/'data/vocab.txt')
    words=(module.PACKAGE/'vocabulary.txt').read_text().splitlines()
    assert len(set(words))==len(words)==500


@pytest.mark.parametrize('confident',[False,True])
def test_cpu_500_word_contract_and_uncertainty(confident):
    class Fake(torch.nn.Module):
        def forward(self,x,lengths):
            assert tuple(x.shape)==(1,1,29,88,88) and lengths==[29]
            assert float(x.mean())==pytest.approx((128/255.-.421)/.165,abs=1e-5)
            logits=torch.zeros(1,500)
            if confident:logits[0,123]=10.
            return logits
    instance=module.LRWTemporalClassifier();instance.load=lambda:instance
    instance.model=Fake();instance.vocabulary=[f'WORD{i}' for i in range(500)];instance.checkpoint_sha256='a'*64
    result=instance.predict_rois(np.ones((29,96,96),dtype=np.uint8)*128)
    assert result['audio_inference'] is False and result['vocabulary_size']==500
    assert result['accuracy_evaluated_locally'] is False
    assert result['uncertain'] is (not confident)
    if confident:assert result['word']=='WORD123'
    else:assert result['word'] is None
    with pytest.raises(ValueError):instance.predict_rois(np.ones((29,64,64),dtype=np.uint8))


def test_real_strict_assign_load_rejects_label_tampering(tmp_path,monkeypatch):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.tcn=torch.nn.Module();self.tcn.tcn_output=torch.nn.Linear(3,500)
    config=tmp_path/'config.json';config.write_text('{}')
    labels=tmp_path/'vocabulary.txt';labels.write_text('\n'.join(f'WORD{i}' for i in range(500)))
    checkpoint=tmp_path/'weights.pth';torch.save({'model_state_dict':Tiny().state_dict()},checkpoint)
    (tmp_path/'acquisition.json').write_text(json.dumps({'sha256':sha256(checkpoint),'config_sha256':sha256(config),'labels_sha256':sha256(labels)}))
    monkeypatch.setattr(module,'PACKAGE',tmp_path);monkeypatch.setattr(module,'build',lambda cfg:Tiny())
    instance=module.LRWTemporalClassifier(checkpoint).load()
    assert instance.vocabulary[123]=='WORD123' and next(instance.model.parameters()).device.type=='cpu'
    labels.write_text('modified')
    with pytest.raises(ValueError,match='labels mismatch'):module.LRWTemporalClassifier(checkpoint).load()
