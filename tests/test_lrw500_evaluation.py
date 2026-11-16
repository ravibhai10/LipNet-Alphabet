"""Evaluation integrity fixtures; no fixture metric is a recognition result."""
import numpy as np
import pytest
from pipeline.lrw_tcn.evaluate import metrics,validate_manifest,load_source


def test_failed_inputs_remain_in_recall_denominator_and_unknowns_are_separate():
    rows=[{'label':'START','unknown':False,'candidate':'START','word':'START','top5':['START']},
          {'label':'START','unknown':False,'error':'tracker failed'},
          {'label':'HELLO','unknown':True,'candidate':'START','word':'START','top5':['START']}]
    report,matrix=metrics(rows,['START','STOP'])
    assert report['known_samples']==2 and report['correct']==1 and report['failed_inputs']==1
    assert report['per_word']['START']['recall']==.5
    assert report['per_word']['STOP']['total']==0
    assert report['unknown_rejection']['false_accept_rate']==1
    assert not report['full_500_class_benchmark'] and report['covered_classes']==1


@pytest.mark.parametrize('change',[
    {'split':'train'}, {'unknown':True}, {'label':'NOT_IN_VOCAB'}, {'provenance':''}])
def test_manifest_rejects_training_rows_and_conflicting_unknown_labels(change):
    row={'id':'one','split':'test','source_sha256':'a'*64,'label':'START','unknown':False,'provenance':'real fixture bytes'}
    with pytest.raises(ValueError):validate_manifest([{**row,**change}],['START'])


def test_content_duplicates_cannot_be_counted_as_independent_utterances():
    row={'id':'one','split':'test','source_sha256':'a'*64,'label':'START','unknown':False,'provenance':'real fixture bytes'}
    with pytest.raises(ValueError,match='Repeated input bytes'):validate_manifest([row,{**row,'id':'two'}],['START'])


def test_source_hash_mismatch_is_not_silently_replaced(tmp_path):
    path=tmp_path/'roi.npy';np.save(path,np.zeros((12,96,96),np.uint8))
    with pytest.raises(ValueError,match='hash mismatch'):
        load_source({'path':'roi.npy','input_kind':'aligned_mouth_npy','source_sha256':'wrong','data_source':'user_recording'},tmp_path)
