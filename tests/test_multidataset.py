"""Software fixtures only; none of these metrics measure speech recognition."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile
import numpy as np
import pytest
pytest.importorskip('requests', reason='Use .venv-word-recognition for optional acquisition workflow tests.')
from pipeline.multidataset import acquisition as acq, adapters as a, sequence as seq


def zip_bytes(entries):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:
        for name,body in entries:z.writestr(name,body)
    return stream.getvalue()


class Response:
    def __init__(self,body,status=200,headers=None):
        self.body=body;self.status_code=status;self.headers={'Content-Length':str(len(body)),'Content-Type':'application/octet-stream',**(headers or {})}
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def iter_content(self,size):yield self.body


def test_transfer_and_resume(tmp_path):
    body=zip_bytes([('real.txt',b'data')]);target=tmp_path/'test.zip';part=target.with_name(target.name+'.part')
    part.write_bytes(body[:12]);source='https://publisher.test/archive'
    acq.atomic_json(part.with_name(part.name+'.json'),{'source_identity':hashlib.sha256(source.encode()).hexdigest(),'expected_bytes':len(body),'etag':'one'})
    def opener(headers):
        assert headers['Range']=='bytes=12-';assert headers['If-Range']=='one'
        return Response(body[12:],206,{'Content-Range':f'bytes 12-{len(body)-1}/{len(body)}','ETag':'one'}),source
    result=acq.transfer(opener,target,reserve_bytes=0,expected_sha256=hashlib.sha256(body).hexdigest())
    assert target.read_bytes()==body and result['publisher_checksum_compared']
    assert acq.transfer(None,target)['status']=='existing'


def test_completed_partial_recovery(tmp_path):
    body=zip_bytes([('x',b'x')]);target=tmp_path/'archive.zip';part=target.with_name(target.name+'.part');part.write_bytes(body)
    acq.atomic_json(part.with_name(part.name+'.json'),{'expected_bytes':len(body),'source_identity':'known'})
    assert acq.transfer(None,target,reserve_bytes=0)['status']=='recovered_complete_partial'


def test_transfer_html_rejected(tmp_path):
    with pytest.raises(PermissionError):acq.transfer(lambda _: (Response(b'<html>login',headers={'Content-Type':'text/html'}),'https://publisher.test'),tmp_path/'bad.zip',reserve_bytes=0)
    assert not (tmp_path/'bad.zip').exists()


def test_transfer_retries_temporary_publisher_error(tmp_path,monkeypatch):
    monkeypatch.setattr(acq.time,'sleep',lambda _:None)
    body=zip_bytes([('x',b'x')]);calls=[]
    def opener(headers):
        calls.append(headers)
        return Response(body,503 if len(calls)==1 else 200),'https://publisher.test/archive'
    result=acq.transfer(opener,tmp_path/'retry.zip',reserve_bytes=0)
    assert len(calls)==2 and result['http_status']==200


@pytest.mark.parametrize('changed',[False,True])
def test_acquisition_receipt_preserved_on_rerun(tmp_path,monkeypatch,changed):
    receipt={'status':'downloaded','source':'https://author.test/original','sha256':'a'*64,'bytes':100}
    acq.atomic_json(tmp_path/'acquisition.json',receipt)
    monkeypatch.setattr(acq,'transfer',lambda *args,**kwargs:{'status':'existing','sha256':('b' if changed else 'a')*64,'bytes':100})
    monkeypatch.setattr(acq,'extract_preserving',lambda *args,**kwargs:{'verified_existing':1})
    if changed:
        with pytest.raises(ValueError,match='preserved'):acq.acquire_miracl(tmp_path)
        assert json.loads((tmp_path/'acquisition.json').read_text())==receipt
    else:
        result=acq.acquire_miracl(tmp_path)
        assert result['source']==receipt['source'] and result['status']=='downloaded'
        assert result['last_verification']['status']=='existing'


def test_public_confirmation_must_identify_author_file():
    from types import SimpleNamespace
    class Session:
        def __init__(self):self.calls=0
        def get(self,*args,**kwargs):
            self.calls+=1
            if self.calls==1:return SimpleNamespace(text='<a href="https://drive.google.com/uc?id=original">all</a>',raise_for_status=lambda:None)
            html=b'<form action="https://drive.usercontent.google.com/download"><input name="confirm" value="t"><input name="id" value="different"></form>'
            return SimpleNamespace(headers={'Content-Type':'text/html'},raise_for_status=lambda:None,iter_content=lambda _:iter([html]),close=lambda:None)
    with pytest.raises(ValueError,match='different archive'):acq.miracl_public_response(Session(),{})


def test_transfer_size_and_checksum(tmp_path,monkeypatch):
    monkeypatch.setattr(acq.time,'sleep',lambda _:None)
    body=zip_bytes([('x',b'x')])
    with pytest.raises(RuntimeError):acq.transfer(lambda _: (Response(body),'https://publisher.test'),tmp_path/'size.zip',max_bytes=1,reserve_bytes=0)
    with pytest.raises(ValueError,match='checksum'):acq.transfer(lambda _: (Response(body),'https://publisher.test'),tmp_path/'sha.zip',expected_sha256='0'*64,reserve_bytes=0)


@pytest.mark.parametrize('name',['../escape','C:/escape','/absolute','a\\evil'])
def test_archive_path_rejected(tmp_path,name):
    # zipfile normalizes backslashes on Windows when *creating* test archives.
    with pytest.raises(ValueError):acq.safe_target(tmp_path/'out',name)
    if '\\' not in name:
        archive=tmp_path/'bad.zip';archive.write_bytes(zip_bytes([(name,b'x')]))
        with pytest.raises(ValueError):acq.extract_preserving(archive,tmp_path/'out',reserve_bytes=0)


def test_archive_preserves_and_verifies(tmp_path):
    archive=tmp_path/'good.zip';archive.write_bytes(zip_bytes([('a/b.txt',b'hello')]))
    assert acq.extract_preserving(archive,tmp_path/'out',reserve_bytes=0)['written']==1
    assert acq.extract_preserving(archive,tmp_path/'out',reserve_bytes=0)['verified_existing']==1
    (tmp_path/'out/a/b.txt').write_bytes(b'user edit')
    with pytest.raises(ValueError,match='preserved'):acq.extract_preserving(archive,tmp_path/'out',reserve_bytes=0)
    assert (tmp_path/'out/a/b.txt').read_bytes()==b'user edit'


def test_tar_and_case_collision(tmp_path):
    archive=tmp_path/'a.tar'
    with tarfile.open(archive,'w') as tar:
        item=tarfile.TarInfo('plain.txt');item.size=3;tar.addfile(item,io.BytesIO(b'abc'))
    assert acq.extract_preserving(archive,tmp_path/'tar',reserve_bytes=0)['members']==1
    archive=tmp_path/'a.zip';archive.write_bytes(zip_bytes([('A.txt',b'a'),('a.txt',b'b')]))
    with pytest.raises(ValueError,match='colliding'):acq.extract_preserving(archive,tmp_path/'zip',reserve_bytes=0)


def rows():
    return [{'id':str(i),'dataset':'fixture','speaker_id':f's{i}','split':split,'source_sha256':hashlib.sha256(str(i).encode()).hexdigest(),
             'task':'isolated_word','transcript':'hello','word_label':'hello','alignment_verified':True,'start_frame':0,'end_frame':3,'source_frames':3,'validation_status':'valid'}
            for i,split in enumerate(('train','valid','test'))]


def test_speaker_splits_real_nonconsecutive_ids():
    speakers=['F01','F02','F04','F05','F06','F07','F08','F09','F10','F11','M01','M02','M04','M07','M08']
    split=a.miracl_splits(set(speakers))
    assert [s for s in speakers if split[s]=='test']==['F10','F11','M08']
    assert len([s for s in split if split[s]=='train'])==9
    with pytest.raises(ValueError):a.miracl_splits(set(speakers[:-1]))


@pytest.mark.parametrize('mutation', ['speaker','content','bounds','label','audio','phrase'])
def test_manifest_validation_failures(mutation):
    r=rows();assert a.validate(r)['samples']==3
    if mutation=='speaker':r[1]['speaker_id']=r[0]['speaker_id']
    if mutation=='content':r[1]['source_sha256']=r[0]['source_sha256']
    if mutation=='bounds':r[1]['end_frame']=4
    if mutation=='label':r[1]['alignment_verified']=False
    if mutation=='audio':r[1]['audio_path']='not allowed'
    if mutation=='phrase':r[1]['task']='phrase'
    with pytest.raises(ValueError):a.validate(r)


def test_author_spelling_preserved():
    assert a.MIRACL_WORDS[0]=='Hegin' and a.MIRACL_WORDS[9]=='Well'
    assert a.MIRACL_UNRESOLVED=={1,10}
    assert a.normalize('  HeLLo   world ')== 'hello world'


def test_muavic_native_discards_audio(tmp_path,monkeypatch):
    video=tmp_path/'clip.mp4';video.write_bytes(b'fixture')
    tsv=tmp_path/'train.tsv';tsv.write_text(str(tmp_path)+'\nid\tclip.mp4\tsecret.wav\t3\t100\n')
    text=tmp_path/'train.en';text.write_text('Actual sentence.\n')
    identities=tmp_path/'speakers.jsonl';identities.write_text(json.dumps({'id':'id','speaker_id':'p1','split':'train','view':'frontal','provenance_reviewed':True}))
    monkeypatch.setattr(a,'video_info',lambda _:dict(frames=3,fps=25))
    r=a.muavic_native(tmp_path,tsv,text,identities)
    assert r[0]['task']=='sentence' and r[0]['word_label'] is None
    assert 'secret.wav' not in json.dumps(r)
    text.write_text('one\ntwo\n')
    with pytest.raises(ValueError):a.muavic_native(tmp_path,tsv,text,identities)


def test_ctc_scores_not_a_model():
    scores=np.eye(3)[[1,1,0,1,2,2]]
    assert seq.ctc_greedy(scores,['<blank>','a','b'])==['a','a','b']
    with pytest.raises(ValueError):seq.ctc_greedy(scores,['wrong'])
    assert seq.edit_distance('hello world'.split(),'hello'.split())==1


def test_sequence_score_requires_actual_ids_and_provenance(tmp_path):
    manifest=tmp_path/'rows.jsonl';prediction=tmp_path/'pred.jsonl'
    manifest.write_text(json.dumps({'id':'x','dataset':'fixture','task':'sentence','split':'test','speaker_id':'held','transcript':'hello world'}))
    prediction.write_text(json.dumps({'id':'x','hypothesis':'hello','modality':'video_only','checkpoint_sha256':'a'*64}))
    report=seq.score(manifest,prediction,tmp_path/'evaluation.json')
    assert report['metrics']['overall']['wer']==.5
    assert report['metrics']['overall']['cer']==6/11
    prediction.write_text(json.dumps({'id':'x','hypothesis':'hello'}))
    with pytest.raises(ValueError):seq.score(manifest,prediction,tmp_path/'reject.json')


def test_legacy_constant_sentence_evaluation_disabled(tmp_path):
    from pipeline.sequence_recognition.ctc_decoder import evaluate_continuous_sentences
    with pytest.raises(RuntimeError,match='unavailable'):evaluate_continuous_sentences(tmp_path/'absent.jsonl')


def test_uncertainty_model_shape_and_cache(tmp_path):
    import torch
    from pipeline.multidataset.experiment import Clips,uncertainty
    from pipeline.word_recognition.model import WordVideoNet
    torch.set_num_threads(1)
    assert uncertainty([.51,.49],['one','two'])['word'] is None
    assert uncertainty([.8,.2],['one','two'])['word']=='one'
    assert 'not validated' in uncertainty([.8,.2],['one','two'])['out_of_vocabulary_detection']
    with pytest.raises(ValueError):uncertainty([.8,float('nan')],['one','two'])
    model=WordVideoNet(8);assert model(torch.ones(2,1,24,64,64)).shape==(2,8)
    clip=tmp_path/'clip.npy';np.save(clip,np.ones((24,64,64),dtype=np.uint8))
    r={'clip':str(clip),'clip_sha256':acq.sha256(clip),'split':'test','word_label':'one'}
    assert Clips([r],['one','two'])[0][0].shape==(1,24,64,64)
    with pytest.raises(ValueError):Clips([r],['one','two'],training=True)
    clip.write_bytes(b'changed')
    with pytest.raises(ValueError):Clips([r],['one','two'])


