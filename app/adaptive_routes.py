"""Private feedback and research model management API; disabled without access keys."""
from __future__ import annotations
import asyncio
import base64
import hmac
import io
import json
import logging
import os
import threading
import time
import zipfile
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Header, Response
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from pipeline.adaptive.quality import canonical, digest, validate_capture
from pipeline.adaptive.store import FeedbackStore, ROOT
from pipeline.adaptive.models import ModelRegistry, TASK_LABELS
from pipeline.adaptive.training import BatchTrainer
from pipeline.adaptive import lifecycle

from app.adaptive_schemas import AdaptivePrediction

router=APIRouter(prefix='/api/adaptive',tags=['private adaptive feedback'])
store=FeedbackStore(); registry=ModelRegistry(store); trainer=BatchTrainer(store)
_capture_lock=threading.Lock()
Task=Literal['alphabet_recognition','isolated_word']
Identifier=Annotated[str,StringConstraints(pattern=r'^[A-Za-z0-9_-]{1,80}$')]


class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid')


class Capture(Strict):
    id: Identifier
    task: Task
    speaker_id: Identifier
    session_id: Identifier
    recorded_at: str=Field(max_length=50)
    capture_kind: Literal['physical_webcam','webcam_replay']
    consent: bool
    frames: list[Annotated[str,Field(max_length=4_000_000)]]=Field(min_length=8,max_length=76)
    timestamps_ms: list[float]=Field(min_length=8,max_length=76)
    original_video: str|None=Field(default=None,max_length=24*1024*1024)


class Feedback(Strict):
    verdict: Literal['correct','incorrect']
    label: str=Field(min_length=1,max_length=32)


class Review(Strict):
    decision: Literal['APPROVED','REJECTED','QUARANTINED']
    reason: str=Field(min_length=3,max_length=1000)


class Version(Strict):
    task: Task
    owner: Identifier
    assignments: dict[Identifier,Literal['train','valid','test']]=Field(min_length=3,max_length=1000)


class Training(Strict):
    dataset_version: Identifier


class Deployment(Strict):
    approved: Literal[True]


from app.security import access


def reviewer(user=Depends(access)):
    if user['role']!='reviewer':raise HTTPException(403,'Reviewer access is required.')
    return user


def owner(user):return None if user['role']=='reviewer' else user['actor']


def call(action,*args,**kwargs):
    try:return action(*args,**kwargs)
    except LookupError as exc:raise HTTPException(404,str(exc)) from exc
    except FileExistsError as exc:raise HTTPException(409,'This final evaluation or immutable artifact already exists.') from exc
    except BlockingIOError as exc:raise HTTPException(429,'The CPU worker is busy. Retry after the current operation finishes.') from exc
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    except HTTPException:raise
    except Exception as exc:
        logging.getLogger(__name__).exception('Private feedback operation failed')
        raise HTTPException(503,'This operation is unavailable. Check the protected local log.') from exc


@router.get('/config')
def config(user=Depends(access)):
    collection={**TASK_LABELS,'isolated_word':[*TASK_LABELS['isolated_word'],'help','i','am']}
    return {'tasks':TASK_LABELS,'correction_labels':collection,'unknown_label':'UNKNOWN',
            'retention_days':__import__('app.settings',fromlist=['settings']).settings().retention_days,
            'adaptation_scope':'per_user_cross_session', 'reference_default':True,
            'minimum_frames':8,'maximum_frames':76,'maximum_duration_ms':5000,'role':user['role'],
            'recording_consent':'I consent to storing this bounded face/mouth recording for review and research training. I can withdraw it. Recordings expire after 90 days by default; renewal is explicit. Withdrawal or expiry invalidates affected private models.',
            'accuracy_notice':'Scores are uncalibrated. Alphabet and physical webcam accuracy remain unproven.',
            'training_support':{'isolated_word':'eight-word frozen-encoder batch adapter','alphabet_recognition':'preserved CNN-BiLSTM; complete reviewed A–Z partitions required'}}


def run_prediction(identifier, actor):
    from app.services.word_compute import word_cpu_lock
    row=store.inspect(identifier,actor)
    if row['prediction']:return row
    rois=np.load(io.BytesIO(store.blob(identifier,'rois',actor)),allow_pickle=False)
    if not word_cpu_lock.acquire(blocking=False):raise BlockingIOError()
    try:
        started=time.perf_counter();prediction=registry.predict(row['task'],rois,owner=actor)
        prediction['latency_ms']=round(1000*(time.perf_counter()-started),2)
        return store.prediction(identifier,prediction,actor)
    finally:word_cpu_lock.release()


async def bounded(action,*args):
    def work():
        if not _capture_lock.acquire(blocking=False):raise BlockingIOError()
        try:return call(action,*args)
        finally:_capture_lock.release()
    try:return await asyncio.wait_for(asyncio.to_thread(work),120)
    except asyncio.TimeoutError as exc:
        raise HTTPException(504,'Recording processing timed out. The bounded worker will release when it finishes; retry using the same recording ID.') from exc


@router.post('/captures',status_code=201)
async def capture(request:Capture,user=Depends(access)):
    def save():
        row=request.model_dump();request_hash=digest(canonical(row))
        try:
            prior=store.inspect(row['id'],user['actor'])
        except LookupError:prior=None
        if prior:
            with store.db() as db:
                existing=store.row(db,row['id'],user['actor'])
                if existing['request_hash']!=request_hash:raise ValueError('Capture ID belongs to different source content.')
        else:
            try:
                meta,media,rois=validate_capture(row,store.root)
            except ValueError:
                with store.db() as db:
                    store.event(db,row['id'],user['actor'],'source_rejected',{'reason':'source_validation_failed','media_retained':False})
                raise
            store.save_capture(row['id'],user['actor'],request_hash,meta,media,rois)
        # A failed model must not destroy a successfully retained consented source.
        try:return run_prediction(row['id'],user['actor'])
        except (RuntimeError,ImportError,BlockingIOError):
            return {**store.inspect(row['id'],user['actor']),'prediction_error':'Source saved; selected model is unavailable or busy. Retry prediction.'}
    return await bounded(save)


@router.post('/captures/{identifier}/predict')
async def predict(identifier:Identifier,user=Depends(access)):
    # Owner inference only, even when reviewer access is available elsewhere.
    return await bounded(run_prediction,identifier,user['actor'])


@router.post('/captures/{identifier}/feedback')
def feedback(identifier:Identifier,request:Feedback,user=Depends(access)):
    row=call(store.inspect,identifier,user['actor'])
    labels=list(TASK_LABELS[row['task']])
    if row['task']=='isolated_word':labels+=['help','i','am']
    return call(store.feedback,identifier,user['actor'],request.verdict,request.label,labels)


@router.post('/captures/{identifier}/withdraw')
def withdraw(identifier:Identifier,user=Depends(access)):
    result=call(store.withdraw,identifier,user['actor'],user['role']=='reviewer')
    lifecycle.cleanup(store)
    return result


@router.get('/records')
def records(task:Task|None=None,status:str|None=None,label:str|None=None,speaker:str|None=None,
            session:str|None=None,verdict:str|None=None,user=Depends(access)):
    return {'records':call(store.list,owner(user),task=task,status=status,label=label,speaker=speaker,session=session,verdict=verdict)}


@router.get('/records/{identifier}')
def inspect(identifier:Identifier,user=Depends(access)):return call(store.inspect,identifier,owner(user))


@router.get('/records/{identifier}/frames/{index}')
def frame(identifier:Identifier,index:int,user=Depends(access)):
    media=call(store.blob,identifier,'media',owner(user))
    with zipfile.ZipFile(io.BytesIO(media)) as archive:
        if f'{index:04d}.png' not in archive.namelist() or index<0:raise HTTPException(404,'Frame not found.')
        raw=archive.read(f'{index:04d}.png')
    return Response(raw,media_type='image/png',headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'})


@router.get('/records/{identifier}/original-video')
def original_video(identifier:Identifier,user=Depends(access)):
    media=call(store.blob,identifier,'media',owner(user))
    with zipfile.ZipFile(io.BytesIO(media)) as archive:
        name=next((n for n in ('original.webm','original.mp4') if n in archive.namelist()),None)
        if not name:raise HTTPException(404,'This recording retains original PNG frames only.')
        raw=archive.read(name)
    return Response(raw,media_type='video/webm' if name.endswith('.webm') else 'video/mp4',
                    headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'})


@router.get('/records/{identifier}/source')
def source(identifier:Identifier,user=Depends(access)):
    return Response(call(store.blob,identifier,'media',owner(user)),media_type='application/zip',
                    headers={'Cache-Control':'no-store','Content-Disposition':'attachment; filename="original-frames.zip"'})


@router.get('/stats')
def stats(user=Depends(access)):return call(store.stats,owner(user))


@router.get('/hard-examples')
def hard_examples(user=Depends(access)):return {'records':call(store.hard_examples,owner(user))}


@router.post('/records/{identifier}/review')
def review(identifier:Identifier,request:Review,user=Depends(reviewer)):
    return call(store.review,identifier,user['actor'],request.decision,request.reason)


@router.post('/datasets',status_code=201)
def dataset(request:Version,user=Depends(reviewer)):
    return call(lifecycle.freeze_personal,store,request.task,TASK_LABELS[request.task],request.owner,request.assignments,user['actor'])


@router.get('/datasets/{identifier}')
def dataset_manifest(identifier:Identifier,user=Depends(reviewer)):return call(store.version,identifier)


@router.post('/training/jobs',status_code=202)
def train(request:Training,user=Depends(reviewer)):return call(trainer.create,request.dataset_version,user['actor'])


@router.get('/training/jobs/{identifier}')
def job(identifier:Identifier,user=Depends(reviewer)):return call(trainer.status,identifier)


@router.get('/models')
def models(user=Depends(access)):return call(registry.list,owner(user))


@router.get('/models/{identifier}/comparison')
def comparison(identifier:Identifier,user=Depends(reviewer)):
    value=call(registry.model,identifier)
    return {'model_id':identifier,'comparison':value.get('comparison'),'test_report':value.get('test_report'),
            'regression_passed':value.get('regression_passed'), 'dataset_version':value['training_dataset_version']}


@router.post('/models/{identifier}/final-evaluation',status_code=202)
def final_evaluation(identifier:Identifier,user=Depends(reviewer)):return call(trainer.finalize,identifier,user['actor'])


@router.post('/models/{identifier}/deploy')
def deploy(identifier:Identifier,request:Deployment,user=Depends(reviewer)):
    return call(registry.deploy,identifier,user['actor'],request.approved)


@router.post('/models/{task}/rollback')
def rollback(task:Task,owner:Identifier,user=Depends(reviewer)):return call(registry.rollback,task,user['actor'],owner)


@router.post('/retention/expire')
def expire(user=Depends(reviewer)):return call(lifecycle.maintenance,store)


class Participant(Strict):
    owner: Identifier
    participant_id: Identifier
    permission_record: str=Field(min_length=3,max_length=1000)


@router.post('/participants/verify')
def verify_participant(request:Participant,user=Depends(reviewer)):
    return call(lifecycle.verify_participant,store,request.owner,request.participant_id,user['actor'],request.permission_record)


class Renewal(Strict):
    consent: Literal[True]


@router.post('/captures/{identifier}/renew')
def renew(identifier:Identifier,request:Renewal,user=Depends(access)):
    return call(lifecycle.renew,store,identifier,user['actor'],request.consent)


@router.get('/retention/report')
def retention_report(user=Depends(reviewer)):
    return call(lifecycle.report,store)


class ManagedBackup(Strict):
    id: Identifier
    owner: Identifier
    source_record: Identifier
    filename: str=Field(min_length=1,max_length=255)


@router.post('/retention/backups',status_code=201)
def register_backup(request:ManagedBackup,user=Depends(reviewer)):
    return call(lifecycle.register_backup,store,request.id,request.owner,request.source_record,request.filename)


class Selection(Strict):
    task: Task
    capture_id: Identifier
    selection: Literal['reference','adaptive']='reference'


@router.post('/predict',response_model=AdaptivePrediction)
async def selected_prediction(request:Selection,user=Depends(access)):
    def work():
        from app.services.word_compute import word_cpu_lock
        row=store.inspect(request.capture_id,user['actor'])
        if row['task']!=request.task:raise ValueError('Capture task mismatch.')
        rois=np.load(io.BytesIO(store.blob(request.capture_id,'rois',user['actor'])),allow_pickle=False)
        if not word_cpu_lock.acquire(blocking=False):raise BlockingIOError()
        try:return registry.predict(request.task,rois,owner=user['actor'],selection=request.selection)
        finally:word_cpu_lock.release()
    return await bounded(work)


@router.post('/models/{identifier}/probe')
async def candidate_probe(identifier:Identifier,user=Depends(reviewer)):
    from pipeline.adaptive.integration import check_candidate
    def work():
        from app.services.word_compute import word_cpu_lock
        model=registry.model(identifier)
        if model.get('invalidated'):raise ValueError('Invalidated candidate.')
        version=store.version(model['training_dataset_version'])
        if not word_cpu_lock.acquire(blocking=False):raise BlockingIOError()
        try:return check_candidate(store,model,version)
        finally:word_cpu_lock.release()
    return await bounded(work)


class PrivateBodyLimit:
    """Enforce a streaming request limit before FastAPI buffers or parses source JSON."""
    def __init__(self,app):self.app=app
    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or not scope.get('path','').startswith('/api/adaptive'):
            return await self.app(scope,receive,send)
        if scope.get('method')=='OPTIONS':
            return await self.app(scope,receive,send)
        headers=dict(scope.get('headers',[]))
        try:access(headers.get(b'authorization',b'').decode('latin1') or None)
        except HTTPException as error:
            await send({'type':'http.response.start','status':error.status_code,'headers':[(b'content-type',b'application/json'),(b'cache-control',b'no-store')]})
            await send({'type':'http.response.body','body':json.dumps({'detail':error.detail}).encode()})
            return
        messages=[];size=0
        while True:
            message=await receive()
            if message['type']=='http.disconnect':return
            size+=len(message.get('body',b''))
            if size>90*1024*1024:
                await send({'type':'http.response.start','status':413,'headers':[(b'content-type',b'application/json')]})
                await send({'type':'http.response.body','body':b'{"detail":"Recording request exceeds 90 MiB."}'})
                return
            messages.append(message)
            if not message.get('more_body'):break
        async def replay():return messages.pop(0) if messages else await receive()
        await self.app(scope,replay,send)
