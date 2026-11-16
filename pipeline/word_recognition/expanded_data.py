"""Pinned, lossless acquisition of the complete allenye66 mouth-sequence corpus.

Never executes upstream code or evaluates data.txt as Python. Git blob hashes
verify publisher bytes; local SHA-256 receipts make downloads resumable.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import gzip
import json
import os
from pathlib import Path
import shutil
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
from .common import ROOT, atomic_json, read_rows, sha256, timestamp, write_rows

COMMIT = 'e88f8a3e2c14cd25b44a0c180e1ebf3510790822'
REPOSITORY = 'https://github.com/allenye66/Computer-Vision-Lip-Reading-2.0'
UPSTREAM_WORDS = ('a','bye','can','cat','demo','dog','hello','here','is','lips','my','read','you')
BASE_WORDS = ('choose','connection','hello','navigation','next','previous','start','stop')
WORDS = tuple(sorted(set(UPSTREAM_WORDS) | set(BASE_WORDS)))
SOURCE = ROOT/'data/external/allenye66_complete_v1'
BASE_RUN = ROOT/'runs/word_accuracy/avhubert_miracl8_v1'
ADAPTER = {'version':'allenye66-bgr80x112-letterbox96-v1',
           'input_shape':[22,80,112,3], 'color':'original OpenCV BGR data.txt triplets',
           'spatial':'grayscale; symmetric reflected vertical padding 16+16 to 112 square; resize96',
           'alignment':'publisher pre-cropped mouth; eye-axis landmarks unavailable',
           'temporal':'linear interpolation 22 to 29 frames; native FPS unknown',
           'filters':'publisher filtering retained; no additional sharpening or denoising'}


def blob_sha(path):
    path=Path(path);h=hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def parse_sequence(path):
    # JSON grammar, a fixed geometry and integer range are all enforced before
    # uint8 conversion. ast.literal_eval/eval and pickle are not accepted.
    raw=json.loads(Path(path).read_text(encoding='utf-8'))
    value=np.asarray(raw)
    if value.shape!=(22,80,112,3) or not np.issubdtype(value.dtype,np.integer) or value.min()<0 or value.max()>255:
        raise ValueError('Expected genuine 22x80x112x3 integer BGR mouth sequence.')
    return value.astype(np.uint8)


def adapt_sequence(value):
    from src.video_training import resample_temporal
    value=np.asarray(value)
    if value.shape!=(22,80,112,3) or value.dtype!=np.uint8:raise ValueError('Unsupported source mouth geometry.')
    frames=[]
    for frame in value:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        square=cv2.copyMakeBorder(gray,16,16,0,0,cv2.BORDER_REFLECT_101)
        frames.append(cv2.resize(square,(96,96),interpolation=cv2.INTER_AREA))
    return np.rint(resample_temporal(frames,29)).clip(0,255).astype(np.uint8)


def download_one(item,directory):
    name=item['path'].split('/')[1]
    if not name.replace('_','').isalnum():raise ValueError('Unsafe source folder name.')
    target=directory/'original'/name/'data.txt';target.parent.mkdir(parents=True,exist_ok=True)
    url=f'https://raw.githubusercontent.com/allenye66/Computer-Vision-Lip-Reading-2.0/{COMMIT}/{item["path"]}'
    if target.exists():
        if target.stat().st_size!=item['size'] or blob_sha(target)!=item['sha']:raise ValueError(f'Protected source changed: {name}')
    else:
        partial=target.with_suffix('.txt.part')
        for attempt in range(5):
            offset=partial.stat().st_size if partial.exists() else 0
            if offset>=item['size']:
                if offset==item['size'] and blob_sha(partial)==item['sha']:
                    partial.rename(target);break
                partial.unlink();offset=0
            # gzip reduces redundant JSON in transit; partial byte ranges use
            # identity encoding so offsets refer to original publisher bytes.
            headers={'User-Agent':'LipNet-research-dataset-ingestion/1.0','Accept-Encoding':'identity' if offset else 'gzip'}
            if offset:headers['Range']=f'bytes={offset}-'
            try:
                with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=90) as response:
                    resume=response.status==206 and offset>0
                    if resume and response.headers.get('Content-Encoding'):
                        # Byte ranges of a compressed representation cannot be
                        # appended to an uncompressed partial file.
                        partial.unlink(missing_ok=True)
                        raise ValueError('Compressed ranged response is not safely resumable.')
                    if resume and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                        raise ValueError('Mismatched partial download range.')
                    if response.status not in (200,206) or 'text/html' in response.headers.get('Content-Type',''):
                        raise ValueError('Not publisher data bytes.')
                    stream=gzip.GzipFile(fileobj=response) if response.headers.get('Content-Encoding')=='gzip' else response
                    with partial.open('ab' if resume else 'wb') as out:
                        for block in iter(lambda:stream.read(256*1024),b''):
                            out.write(block)
                            if out.tell()>item['size']:raise ValueError('Download exceeds publisher size.')
                if partial.stat().st_size!=item['size'] or blob_sha(partial)!=item['sha']:
                    # A complete corrupt response cannot be repaired by appending.
                    partial.unlink(missing_ok=True)
                    raise ValueError('Publisher Git blob/length integrity check failed.')
                partial.rename(target);break
            except (OSError,ValueError,EOFError,urllib.error.URLError) as error:
                if isinstance(error,urllib.error.HTTPError) and error.code==416:
                    partial.unlink(missing_ok=True)
                if attempt==4:raise
                time.sleep(min(2**attempt,8))
    receipt=target.parent/'receipt.json'
    label=name.rsplit('_',1)[0]
    expected_label=label if label in UPSTREAM_WORDS else None
    if receipt.exists():
        row=json.loads(receipt.read_text())
        if row.get('id') != 'allenye66:'+name or row.get('word_label') != expected_label or row.get('commit') != COMMIT:
            raise ValueError('Cached publisher identity or label changed.')
        if row['source_sha256']!=sha256(target) or row['array_sha256']!=sha256(row['array']):raise ValueError('Cached source/decoded content changed.')
        return row
    value=parse_sequence(target);array=target.parent/'frames.npy'
    if array.exists():
        if not np.array_equal(np.load(array,allow_pickle=False),value):
            raise ValueError('Protected decoded publisher array changed.')
    else:
        with array.open('xb') as out:np.save(out,value,allow_pickle=False)
    label=name.rsplit('_',1)[0]
    row={'id':'allenye66:'+name,'dataset':'allenye66','source_url':url,'commit':COMMIT,
         'source_path':str(target.resolve()),'source_sha256':sha256(target),'git_blob_sha1':item['sha'],
         'bytes':item['size'],'array':str(array.resolve()),'array_sha256':sha256(array),
         'pixels_sha256':hashlib.sha256(value.tobytes()).hexdigest(),'shape':list(value.shape),
         'word_label':label if label in UPSTREAM_WORDS else None,
         'label_status':'publisher_folder_label' if label in UPSTREAM_WORDS else 'unresolved_test_folder',
         'speaker_id':None,'session_id':None,'fps':None,'acquired_at':timestamp()}
    atomic_json(receipt,row);return row


def acquire(directory=SOURCE,workers=6):
    cv2.setNumThreads(1);directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    tree_path=ROOT/'.qa/allenye66-audit/tree.json'
    tree=json.loads(tree_path.read_text())
    if tree.get('commit')!=COMMIT or tree.get('truncated'):raise ValueError('Complete pinned publisher index required.')
    items=[r for r in tree['tree'] if r['path'].startswith('collected_data/') and r['path'].endswith('/data.txt')]
    if len(items)!=685:raise ValueError('Publisher inventory unexpectedly changed.')
    expected=sum(r['size'] for r in items)
    if shutil.disk_usage(directory).free<expected+2*1024**3:raise OSError('Insufficient disk reserve for lossless acquisition.')
    started=time.perf_counter();rows=[];errors=[]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(download_one,item,directory):item for item in items}
        for future in as_completed(futures):
            try:rows.append(future.result())
            except Exception as error:
                errors.append({'path':futures[future]['path'],'error':str(error)})
                print(f'Acquisition retry required: {errors[-1]}',flush=True)
            if (len(rows)+len(errors))%20==0:
                print(f'Lossless sources {len(rows)}/{len(items)}, errors {len(errors)}, {time.perf_counter()-started:.1f}s',flush=True)
    rows.sort(key=lambda r:r['id']);write_rows(directory/'inventory.jsonl',rows)
    report={'repository':REPOSITORY,'commit':COMMIT,'publisher_index_sha256':sha256(tree_path),
            'expected_samples':len(items),'acquired':len(rows),'labeled':sum(r['word_label'] is not None for r in rows),
            'per_word':dict(Counter(r['word_label'] for r in rows if r['word_label'])),
            'unresolved_labels':[r['id'] for r in rows if r['word_label'] is None],
            'bytes':sum(r['bytes'] for r in rows),'seconds':time.perf_counter()-started,'errors':errors,
            'integrity':'Publisher Git blob SHA-1 and byte length compared; local SHA-256 and decoded pixel SHA-256 recorded.',
            'license':'No repository LICENSE found in pinned tree. Public access does not establish a redistribution/commercial grant.',
            'speaker_session_metadata':'not supplied; no speaker-independent upstream accuracy may be claimed',
            'audio_used':False,'adapter':ADAPTER}
    atomic_json(directory/'acquisition.json',report);print(json.dumps(report,indent=2),flush=True)
    if errors:raise RuntimeError('Incomplete acquisition; rerun to resume verified sources.')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=SOURCE);parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if not 1<=args.workers<=12:parser.error('Choose 1..12 bounded download workers.')
    acquire(args.output,args.workers)
