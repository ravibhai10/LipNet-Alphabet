"""Protocol/geometry fixtures only; these checks are not speech accuracy results."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from pipeline.word_recognition import pretrained as p
from pipeline.word_recognition.common import write_rows,sha256


def fixture_rows():
    result=[]
    for i in range(15):
        split='train' if i<9 else 'valid' if i<12 else 'test'
        for role,nclasses in [('isolated_word',8),('phrase',10)]:
            for word in range(nclasses):
                for repetition in range(1,11):
                    sample=f'{i}:{role}:{word}:{repetition}';label=f'word{word}' if role=='isolated_word' else f'phrase {word}'
                    result.append({'id':sample,'dataset':'miracl','speaker_id':str(i),'split':split,'task':role,'repetition':repetition,
                                   'transcript':label,'word_label':label if role=='isolated_word' else None,'alignment_verified':role=='isolated_word',
                                   'source_sha256':hashlib.sha256(sample.encode()).hexdigest(),'start_frame':0,'end_frame':3,'source_frames':3,'validation_status':'valid'})
    return result


def test_frozen_protocol_quarantines_historical_test(tmp_path,monkeypatch):
    source=tmp_path/'source.jsonl';write_rows(source,fixture_rows())
    encoder=tmp_path/'encoder.pt';encoder.write_bytes(b'fixture, not a real model')
    asset=tmp_path/'frontend/public/models/face_landmarker.task';asset.parent.mkdir(parents=True);asset.write_bytes(b'fixture landmarks')
    monkeypatch.setattr(p,'ROOT',tmp_path);monkeypatch.setattr(p,'ENCODER',encoder)
    protocol=p.initialize(source,tmp_path/'run')
    assert protocol['word_counts']==dict(train=720,valid=240,test=192)
    assert protocol['phrase_ood_counts']==dict(valid=60,test=60)
    assert len(protocol['quarantined_previous_test_ids'])==48
    _,rows=p.contract_for(tmp_path/'run')
    assert not set(protocol['quarantined_previous_test_ids'])&{r['id'] for r in rows}
    assert all(r['repetition']>=3 for r in rows if r['split']=='test' and r['role']=='known_word')
    with pytest.raises(FileExistsError):p.initialize(source,tmp_path/'run')
    source.write_text('{}\n')
    with pytest.raises(ValueError):p.contract_for(tmp_path/'run')


@pytest.mark.parametrize('kind,dimension',[('mean',768),('temporal3',2304),('mean_std',1536)])
def test_pool_preserves_order_and_output_dimension(kind,dimension):
    f=np.tile(np.arange(29,dtype=np.float32)[:,None],(1,768))
    pooled=p.pool(f,kind)
    assert pooled.shape==(dimension,) and np.isfinite(pooled).all()
    if kind=='temporal3':assert pooled[0]<pooled[768]<pooled[1536]


def test_head_probabilities_are_real_linear_outputs():
    import torch
    head={'vocabulary':['one','two'],'feature_layer':6,'pooling':'mean','mean':torch.zeros(768),'scale':torch.ones(768),
          'weight':torch.stack([torch.ones(768),-torch.ones(768)]),'bias':torch.zeros(2)}
    features=np.ones((3,29,768),dtype=np.float32)*.001
    values=p.predict_features(head,features)
    assert values.shape==(2,) and values[0]>.8 and values.sum()==pytest.approx(1)
    head['scale'][0]=0
    with pytest.raises(ValueError):p.predict_features(head,features)


def test_feature_cache_rejects_changed_content(tmp_path):
    feature=tmp_path/'feature.npy';np.save(feature,np.ones((3,29,768),dtype=np.float32))
    row={'usable':True,'feature':str(feature),'feature_sha256':sha256(feature),'role':'known_word','word_label':'hello'}
    x,y,rows=p.vectors([row],['hello','stop'],9,'temporal3')
    assert x.shape==(1,2304) and y.tolist()==[0]
    feature.write_bytes(b'changed')
    with pytest.raises(ValueError):p.vectors([row],['hello','stop'],9,'mean')


def test_unknown_fps_requires_explicit_fixed_length():
    from pipeline.avhubert.loader import AVHubertLoader
    from types import SimpleNamespace
    loader=AVHubertLoader();loader.available=True;loader._task=SimpleNamespace(cfg=SimpleNamespace(image_crop_size=88,image_mean=.421,image_std=.165))
    rois=np.ones((8,96,96),dtype=np.uint8)*128
    actual=loader.prepare_rois(rois,source_fps=None,target_frames=29)
    expected=loader.prepare_rois(rois,source_fps=25,target_frames=29)
    assert np.array_equal(actual.numpy(),expected.numpy()) and tuple(actual.shape)==(1,1,29,88,88)
    with pytest.raises(ValueError):loader.prepare_rois(rois,source_fps=None)
