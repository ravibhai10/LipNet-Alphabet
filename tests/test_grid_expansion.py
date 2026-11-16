"""Offline transport/annotation tests. No generated files enter the real corpus."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from scripts.download_grid_official import discover_links, download, extract_preserving, verify_archive
from scripts.prepare_grid_dataset import assign_speakers, deduplicate, parse_alignment, validate_frame_count
from pipeline.avhubert.classification import audit_records


def test_links_are_discovered_not_fabricated():
    html='<a href="s2/video/s2.mpg_vcd.zip">normal</a><a href="s2/align/s2.tar">align</a><a href="s2/video/high.tar">high</a><a href="https://example.com/s3/video/s3.mpg_vcd.zip">foreign</a>'
    links=discover_links(html)
    assert set(links)=={(2,'video_normal'),(2,'alignment')}
    assert links[2,'alignment']=='https://spandh.dcs.shef.ac.uk/gridcorpus/s2/align/s2.tar'
    assert (21,'video_normal') not in links


@contextmanager
def server(payload, *, html=False):
    calls=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_GET(self):
            value=self.headers.get('Range'); calls.append(value)
            offset=int(value.split('=')[1].split('-')[0]) if value else 0
            self.send_response(206 if offset else 200)
            self.send_header('Content-Length',str(len(payload)-offset))
            self.send_header('Content-Type','text/html' if html else 'application/zip')
            if offset: self.send_header('Content-Range',f'bytes {offset}-{len(payload)-1}/{len(payload)}')
            self.end_headers(); self.wfile.write(payload[offset:])
    http=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
    try: yield f'http://127.0.0.1:{http.server_port}/asset',calls
    finally: http.shutdown();http.server_close();thread.join()


def test_resumable_download_checks_ranges_and_size(tmp_path):
    data=b'PK'+bytes(range(100))
    destination=tmp_path/'asset.zip'
    partial=tmp_path/'asset.zip.part';partial.write_bytes(data[:30])
    with server(data) as (url,calls):
        result,metadata=download(url,destination,attempts=1,deadline=time.monotonic()+10,reserve_bytes=0)
    assert calls==['bytes=30-'] and result.read_bytes()==data
    assert metadata['http_status']==206 and metadata['expected_bytes']==len(data)
    assert not destination.exists()  # Caller validates archive before final publication.


def test_html_is_rejected_before_it_becomes_an_archive(tmp_path):
    with server(b'<html>not a corpus</html>',html=True) as (url,_):
        with pytest.raises(ValueError,match='HTML'):
            download(url,tmp_path/'asset.zip',attempts=1,deadline=time.monotonic()+10,reserve_bytes=0)
    assert not (tmp_path/'asset.zip').exists()


def test_complete_partial_can_be_verified_without_another_request(tmp_path):
    (tmp_path/'asset.zip.part').write_bytes(b'complete')
    (tmp_path/'asset.zip.part.json').write_text(json.dumps({'expected_bytes':8}))
    result,_=download('http://127.0.0.1:1/not-requested',tmp_path/'asset.zip',attempts=1,deadline=time.monotonic()+1,reserve_bytes=0)
    assert result.read_bytes()==b'complete'


def test_crc_and_nonoverwriting_extraction(tmp_path):
    archive=tmp_path/'source.zip'
    with zipfile.ZipFile(archive,'w') as output: output.writestr('s2/test.mpg',b'unit-test transport bytes')
    assert verify_archive(archive,'video_normal')[0]==1
    corpus=tmp_path/'corpus'
    first=extract_preserving(archive,corpus,'s2','video_normal')
    second=extract_preserving(archive,corpus,'s2','video_normal')
    assert first['written']==1 and second['existing_verified']==1
    target=corpus/'T02/video/s2/test.mpg';target.write_bytes(b'user file')
    with pytest.raises(FileExistsError): extract_preserving(archive,corpus,'s2','video_normal')
    assert target.read_bytes()==b'user file'
    archive.write_text('<html>failed download</html>')
    with pytest.raises(zipfile.BadZipFile): verify_archive(archive,'video_normal')


def test_archive_traversal_is_rejected(tmp_path):
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as output: output.writestr('../escape.mpg',b'bad')
    with pytest.raises(ValueError,match='Unsafe'): extract_preserving(archive,tmp_path/'corpus','s2','video_normal')
    assert not (tmp_path/'escape.mpg').exists()


def test_grid_alignment_retains_units_and_filters_pauses(tmp_path):
    path=tmp_path/'example.align'
    path.write_text('0 1000 sil\n1000 2000 bin\n2000 3000 blue\n3000 4000 at\n4000 8000 a\n8000 9000 sp\n9000 11000 two\n11000 13000 now\n13000 14000 sil\n')
    tokens,end=parse_alignment(path)
    assert len(tokens)==6 and tokens[3]=={'start_units':4000,'end_units':8000,'word':'a'}
    assert end==14000  # 25 kHz units; not milliseconds.
    path.write_text('5 2 bin')
    with pytest.raises(ValueError):parse_alignment(path)


def test_actual_decoding_can_exceed_container_estimate_but_not_end_early():
    validate_frame_count(75,74)
    validate_frame_count(75,75)
    validate_frame_count(75,0)
    for actual,reported in [(72,74),(2,74),(1,0)]:
        with pytest.raises(ValueError,match='Decoding count mismatch'):
            validate_frame_count(actual,reported)


def test_speaker_partitions_and_duplicate_content():
    rows=[{'speaker':f's{n}','label':'A','status':'valid','video':f's{n}/clip','video_sha256':str(n),'decoded_thumbnail_sha256':str(n),'transcript':'bin blue at a two now'} for n in range(1,7)]
    assignments=assign_speakers(rows)
    assert assignments=={'s1':'train','s2':'train','s3':'train','s4':'train','s5':'valid','s6':'test'}
    assert len(set(assignments))==6
    rows[5]['video_sha256']=rows[0]['video_sha256']
    eligible,findings=deduplicate(rows)
    assert len(eligible)==5 and findings[0]['excluded']==['s6/clip']
    rows[5]['transcript']='bin blue at b two now'
    eligible,_=deduplicate(rows)
    assert all(row['video'] not in {'s1/clip','s6/clip'} for row in eligible)


@pytest.mark.parametrize('bounds',[{'start_frame':True,'end_frame':8},{'start_frame':0},{'start_frame':5,'end_frame':6},{'start_frame':0,'end_frame':501}])
def test_manifest_rejects_invalid_segments(bounds):
    row=dict(id='real-a',speaker_id='s1',label='A',split='train',data_source='real_camera',video_sha256='0'*64,**bounds)
    with pytest.raises(ValueError,match='segments'): audit_records([row],require_splits=False)


def test_segment_tracking_uses_source_indices_and_rejects_truncation(monkeypatch):
    import tracking.aligner as aligner
    import tracking.mouth_canonicalization as canonical
    calls=[]
    class Capture:
        def __init__(self,*args): self.position=0
        def get(self,*args): return 25.
        def read(self):
            if self.position==10:return False,None
            value=self.position;self.position+=1
            return True,np.full((4,4,3),value,np.uint8)
        def release(self):pass
    class Tracker:
        def __init__(self,*args,**kwargs):pass
        def process_frame(self,frame,index,timestamp):
            calls.append((index,timestamp))
            return SimpleNamespace(has_face=True,landmarks=np.zeros((300,3)),quality_flags=[])
        def close(self):pass
    monkeypatch.setattr(canonical.cv2,'VideoCapture',Capture)
    monkeypatch.setattr(aligner,'FaceTracker',Tracker)
    monkeypatch.setattr(canonical,'aligned_mouth_crop',lambda frame,landmarks:frame)
    frames,fps=canonical.read_aligned_mouth_video('mock-only',start_frame=3,end_frame=7)
    assert fps==25 and frames[:,0,0].tolist()==[3,4,5,6]
    assert calls==[(3,120.),(4,160.),(5,200.),(6,240.)]
    with pytest.raises(ValueError,match='beyond'):canonical.read_aligned_mouth_video('mock-only',start_frame=8,end_frame=12)
