"""Genuine source through the real feedback API in a disposable private database.

Confirmation is automated QA, NOT human annotation. No recording is retained or
added to any real training dataset. No model response is mocked.
"""
from __future__ import annotations
import argparse
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import cv2

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


class ASGIClient:
    """Exercise the standard ASGI interface without adding encoder dependencies."""
    def __init__(self,app):self.app=app
    async def __aenter__(self):return self
    async def __aexit__(self,*args):return False
    async def request(self,method,path,body=None,headers=None):
        encoded=json.dumps(body).encode() if body is not None else b''
        messages=[];consumed=False
        async def receive():
            nonlocal consumed
            if consumed:return {'type':'http.disconnect'}
            consumed=True;return {'type':'http.request','body':encoded,'more_body':False}
        async def send(message):messages.append(message)
        scope={'type':'http','method':method,'path':path,'raw_path':path.encode(),'query_string':b'',
               'headers':[(k.lower().encode(),v.encode()) for k,v in (headers or {}).items()]+[(b'content-type',b'application/json')],
               'scheme':'http','http_version':'1.1','root_path':'','server':('127.0.0.1',8018),'client':('127.0.0.1',12345)}
        await self.app(scope,receive,send)
        status=next(m['status'] for m in messages if m['type']=='http.response.start')
        content=b''.join(m.get('body',b'') for m in messages if m['type']=='http.response.body')
        return SimpleNamespace(status_code=status,content=content,json=lambda:json.loads(content))
    async def get(self,path,headers=None):return await self.request('GET',path,headers=headers)
    async def post(self,path,json=None,headers=None):return await self.request('POST',path,json,headers)


async def main():
    from fastapi import FastAPI
    import app.adaptive_routes as routes
    from pipeline.adaptive.store import FeedbackStore
    from pipeline.adaptive.models import ModelRegistry
    from pipeline.adaptive.training import BatchTrainer
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Preserve the earlier run.')
    rows=[json.loads(line) for line in (ROOT/'data/processed/multidataset/miracl_v1.jsonl').read_text().splitlines()]
    row=next(r for r in rows if r['task']=='isolated_word' and r['split']=='valid' and r['word_label']=='hello' and len(r['frames'])>=8)
    frames=['data:image/png;base64,'+base64.b64encode(cv2.imencode('.png',cv2.imread(str(name)))[1]).decode() for name in row['frames']]
    capture={'id':'qa-real-source','task':'isolated_word','speaker_id':'qa-only-not-a-new-speaker','session_id':'qa-only',
             'recorded_at':datetime.now(timezone.utc).isoformat(),'capture_kind':'webcam_replay','consent':True,
             'frames':frames,'timestamps_ms':[i*100 for i in range(len(frames))]}
    with tempfile.TemporaryDirectory(prefix='lipnet-qa-video-') as directory:
        video=Path(directory)/'qa.webm';first=cv2.imread(str(row['frames'][0]))
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'VP80'),10.,(first.shape[1],first.shape[0]))
        if not writer.isOpened():raise RuntimeError('Local QA WebM encoder is unavailable.')
        for name in row['frames']:writer.write(cv2.imread(str(name)))
        writer.release();original_video_bytes=video.read_bytes()
    capture['original_video']='data:video/webm;base64,'+base64.b64encode(original_video_bytes).decode()
    os.environ['LIPNET_FEEDBACK_TOKEN']='qa-collector-only-credential-00000000000000'
    os.environ['LIPNET_REVIEW_TOKEN']='qa-reviewer-only-credential-000000000000000'
    header={'Authorization':'Bearer '+os.environ['LIPNET_FEEDBACK_TOKEN']}
    checks=[]
    def check(value,name):
        if not value:raise AssertionError(name)
        checks.append(name)
    with tempfile.TemporaryDirectory(prefix='lipnet-adaptive-qa-') as directory:
        store=FeedbackStore(ROOT);store.directory=Path(directory)
        routes.store=store;routes.registry=ModelRegistry(store);routes.trainer=BatchTrainer(store)
        app=FastAPI();app.include_router(routes.router);app.add_middleware(routes.PrivateBodyLimit)
        async with ASGIClient(app) as client:
            response=await client.post('/api/adaptive/captures',json=capture,headers=header)
            check(response.status_code==201,'real source retained and independently validated through ASGI')
            saved=response.json();prediction=saved['prediction']
            check(prediction is not None and prediction['model_id']=='avhubert-miracl8-preserved','actual registered model response linked')
            check(prediction['candidate']=='hello','publisher HELLO replay candidate is HELLO')
            check(prediction['model_version']==routes.registry.model(prediction['model_id'])['checkpoint_sha256'],'prediction hash matches registry')
            duplicate=await client.post('/api/adaptive/captures',json=capture,headers=header)
            check(duplicate.status_code==201 and duplicate.json()['prediction']==prediction,'capture retry is idempotent and reuses actual prediction')
            raw=await client.get('/api/adaptive/records/qa-real-source/source',headers=header)
            check(raw.status_code==200 and raw.content.startswith(b'PK'),'original source archive accessible only with ownership')
            original=await client.get('/api/adaptive/records/qa-real-source/original-video',headers=header)
            check(original.status_code==200 and original.content==original_video_bytes,'bounded original video retained byte-for-byte and privately replayable')
            check((await client.get('/api/adaptive/records/qa-real-source/source')).status_code==401,'unauthorized source access denied')
            label=prediction['prediction']
            if label:
                feedback=await client.post('/api/adaptive/captures/qa-real-source/feedback',json={'verdict':'correct','label':label},headers=header)
            else:
                feedback=await client.post('/api/adaptive/captures/qa-real-source/feedback',json={'verdict':'incorrect','label':'hello'},headers=header)
            check(feedback.status_code==200 and feedback.json()['status']=='PENDING_REVIEW','automated QA decision remains pending, never automatically approved')
            check((await client.get('/api/adaptive/stats',headers=header)).json()['total']==1,'statistics update from actual private record')
            withdrawn=await client.post('/api/adaptive/captures/qa-real-source/withdraw',json={},headers=header)
            check(withdrawn.status_code==200 and withdrawn.json()['media_deleted'],'withdrawal deletes retained source')
            check((await client.get('/api/adaptive/records/qa-real-source/source',headers=header)).status_code==404,'withdrawn archive no longer served')
            check((await client.get('/api/adaptive/records/qa-real-source/original-video',headers=header)).status_code==404,'withdrawal removes original video as well as frames')
            check(not list((ROOT/'runs/adaptive').glob('qa-*')),'no QA model was fitted')
        routes.trainer.executor.shutdown(wait=True)
    result={'scope':'Real historical HELLO sequence, independent server tracking and real model through ASGI. Automated QA feedback only; temporary database removed; not human-verified training or physical webcam accuracy.',
            'source_id':row['id'],'source_native_fps':None,'replay_schedule_ms':100,'checks':checks,'checks_passed':len(checks),
            'prediction':prediction,'recordings_retained':0,'models_trained':0}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'checks_passed':len(checks),'candidate':prediction['candidate'],'recordings_retained':0}))


if __name__=='__main__':
    import asyncio
    asyncio.run(main())
