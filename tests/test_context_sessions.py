import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from pipeline.adaptive.context_sessions import MIGRATION_VERSION, migrate, restore_database, rollback
from app.services.context_sessions import (ContextSessionError, ContextSessionStore,
                                           NOTICE_VERSION, PURPOSE)


def legacy_db(path: Path):
    db=sqlite3.connect(path)
    db.executescript('''
      CREATE TABLE participants(owner TEXT NOT NULL,id TEXT NOT NULL,verified_by TEXT NOT NULL,
        permission TEXT NOT NULL,PRIMARY KEY(owner,id));
      CREATE TABLE captures(id TEXT PRIMARY KEY,owner TEXT NOT NULL,session TEXT NOT NULL);
      INSERT INTO participants VALUES('owner-a','p-1','reviewer','permission-1');
      INSERT INTO participants VALUES('owner-b','p-1','reviewer','permission-2');
      INSERT INTO participants VALUES('owner-b','p-2','reviewer','permission-3');
      INSERT INTO captures VALUES('record-1','owner-a','recording-session');
      PRAGMA user_version=0;
    ''')
    db.commit();db.close()


def test_migration_is_additive_backup_and_restore_round_trip(tmp_path):
    path=tmp_path/'feedback.sqlite3';legacy_db(path)
    backup=migrate(path)
    assert backup.is_file() and backup.stat().st_size>0
    db=sqlite3.connect(path)
    assert db.execute('PRAGMA user_version').fetchone()[0]==MIGRATION_VERSION
    assert db.execute("SELECT id,owner,session FROM captures").fetchall()==[('record-1','owner-a','recording-session')]
    assert db.execute("SELECT owner,id,verified_by,permission FROM participants ORDER BY owner").fetchall()==[
        ('owner-a','p-1','reviewer','permission-1'),('owner-b','p-1','reviewer','permission-2'),
        ('owner-b','p-2','reviewer','permission-3')]
    assert {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")} >= {
        'captures','participants','context_sessions','context_session_consents'}
    db.close()
    # Explicit restore keeps a safety copy of the migrated state and restores v0.
    pre_restore=restore_database(path,backup)
    assert pre_restore.is_file()
    db=sqlite3.connect(path)
    assert db.execute('PRAGMA user_version').fetchone()[0]==0
    assert db.execute('SELECT COUNT(*) FROM captures').fetchone()[0]==1
    assert db.execute('SELECT COUNT(*) FROM participants').fetchone()[0]==3
    db.close()


def test_rollback_refuses_to_drop_live_session_rows(tmp_path):
    path=tmp_path/'feedback.sqlite3';legacy_db(path);migrate(path)
    store=ContextSessionStore(path)
    created=store.create('owner-a')
    with pytest.raises(RuntimeError,match='sessions exist'):
        rollback(path)
    assert created['session_token']


def test_explicit_rollback_succeeds_only_for_empty_session_tables(tmp_path):
    path=tmp_path/'feedback.sqlite3';legacy_db(path);migrate(path)
    backup=rollback(path)
    assert backup.is_file()
    db=sqlite3.connect(path)
    assert db.execute('PRAGMA user_version').fetchone()[0]==0
    assert db.execute('SELECT COUNT(*) FROM captures').fetchone()[0]==1
    assert not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='context_sessions'").fetchone()
    db.close()


def test_tokens_are_hashed_and_session_is_owner_bound(tmp_path):
    path=tmp_path/'feedback.sqlite3';legacy_db(path);migrate(path)
    store=ContextSessionStore(path);created=store.create('owner-a','p-1')
    assert created['participant_bound'] is True
    db=sqlite3.connect(path)
    saved=db.execute('SELECT token_sha256,owner,participant_id FROM context_sessions').fetchone();db.close()
    assert saved==(hashlib.sha256(created['session_token'].encode()).hexdigest(),'owner-a','p-1')
    assert created['session_token'] not in saved[0]
    with pytest.raises(ContextSessionError):store.authorize('owner-b',created['session_token'],False)


def test_participant_must_be_verified_and_owned(tmp_path):
    path=tmp_path/'feedback.sqlite3';legacy_db(path);migrate(path)
    store=ContextSessionStore(path)
    with pytest.raises(ContextSessionError,match='participant_not_verified'):
        store.create('owner-a','unknown')
    with pytest.raises(ContextSessionError,match='participant_not_verified'):
        store.create('owner-a','p-2')


def test_consent_is_purpose_version_and_session_scoped(tmp_path):
    path=tmp_path/'feedback.sqlite3';legacy_db(path);migrate(path)
    store=ContextSessionStore(path);a=store.create('owner-a');b=store.create('owner-a')
    with pytest.raises(ContextSessionError):store.authorize('owner-a',a['session_token'])
    with pytest.raises(ContextSessionError):store.grant_consent('owner-a',a['session_token'],'recording','old',True)
    scope=store.grant_consent('owner-a',a['session_token'],PURPOSE,NOTICE_VERSION,True)
    authorized=store.authorize('owner-a',a['session_token'])
    assert authorized['consent_id'] is not None
    with pytest.raises(ContextSessionError):store.authorize('owner-a',b['session_token'])
    store.revoke_consent('owner-a',a['session_token'])
    assert not store.is_current('owner-a',scope,authorized['consent_id'])
    with pytest.raises(ContextSessionError,match='consent_required'):store.authorize('owner-a',a['session_token'])


def test_expiry_and_termination_are_fail_closed(tmp_path):
    path=tmp_path/'feedback.sqlite3';legacy_db(path);migrate(path)
    now=[datetime(2026,11,18,tzinfo=timezone.utc)]
    store=ContextSessionStore(path,clock=lambda:now[0]);created=store.create('owner-a')
    store.grant_consent('owner-a',created['session_token'],PURPOSE,NOTICE_VERSION,True)
    now[0]+=timedelta(hours=9)
    with pytest.raises(ContextSessionError,match='session_expired'):store.authorize('owner-a',created['session_token'],False)
    now[0]-=timedelta(hours=9)
    another=store.create('owner-a');store.terminate('owner-a',another['session_token'])
    with pytest.raises(ContextSessionError,match='session_terminated'):store.authorize('owner-a',another['session_token'],False)


def test_interpretation_cache_isolated_and_revocation_discards_late_result(tmp_path,monkeypatch):
    import app.services.context_interpretation as ci
    monkeypatch.setenv('LIPNET_LLM_ENABLED','1');monkeypatch.setenv('OPENAI_API_KEY','test-only-no-provider-call')
    path=tmp_path/'feedback.sqlite3';legacy_db(path);migrate(path)
    store=ContextSessionStore(path)
    a=store.create('owner-a');b=store.create('owner-a')
    sa=store.grant_consent('owner-a',a['session_token'],PURPOSE,NOTICE_VERSION,True)
    sb=store.grant_consent('owner-a',b['session_token'],PURPOSE,NOTICE_VERSION,True)
    req=ci.InterpretationRequest(mode='word',event_id='e1',visual_candidate='hello',original_word=None,
        visual_uncertain=True,candidates=[{'word':'hello','prob':.55},{'word':'next','prob':.35}],
        context='menu',history=[],consent_text=True)
    async def run():
        calls=[];now=[10.]
        async def provider(payload):calls.append(1);return {'word':'next','reason':'context_fit'}
        service=ci.InterpretationService(provider,clock=lambda:now[0])
        await service.interpret(req,'owner-a',sa,sa['consent_id'],store)
        assert (await service.interpret(req,'owner-a',sa,sa['consent_id'],store))['cached']
        now[0]+=2.1
        await service.interpret(req,'owner-a',sb,sb['consent_id'],store)
        assert len(calls)==2
        entered=asyncio.Event();release=asyncio.Event()
        async def delayed(payload):entered.set();await release.wait();return {'word':'next','reason':'context_fit'}
        delayed_service=ci.InterpretationService(delayed)
        task=asyncio.create_task(delayed_service.interpret(req,'owner-a',sa,sa['consent_id'],store))
        await entered.wait();store.revoke_consent('owner-a',a['session_token']);release.set()
        result=await task
        assert result['status']=='session_inactive' and result['llm_suggestion'] is None
        assert not delayed_service.cache
    asyncio.run(run())


def test_revocation_fence_blocks_submission_and_waits_for_submission_boundary(tmp_path,monkeypatch):
    import app.services.context_interpretation as ci
    monkeypatch.setenv('LIPNET_LLM_ENABLED','1');monkeypatch.setenv('OPENAI_API_KEY','test-only-no-provider-call')
    path=tmp_path/'feedback.sqlite3';legacy_db(path);migrate(path)
    store=ContextSessionStore(path);created=store.create('owner-a')
    scope=store.grant_consent('owner-a',created['session_token'],PURPOSE,NOTICE_VERSION,True)
    req=ci.InterpretationRequest(mode='word',event_id='fence',visual_candidate='hello',original_word=None,
        visual_uncertain=True,candidates=[{'word':'hello','prob':.55},{'word':'next','prob':.35}],
        context='menu',history=[],consent_text=True)
    calls=[]
    def provider(_payload):
        calls.append('submitted')
        async def response(): return {'word':'next','reason':'context_fit'}
        return response()
    service=ci.InterpretationService(provider)

    # Revoke after the preliminary read check but before the fenced authorization.
    # This deterministically exercises the former check-to-submit race window.
    class RevokeBetweenChecks:
        triggered=False
        def is_current(self,*args):
            current=store.is_current(*args)
            if current and not self.triggered:
                self.triggered=True
                thread=threading.Thread(target=lambda:store.revoke_consent('owner-a',created['session_token']))
                thread.start();thread.join(3)
                assert not thread.is_alive()
            return current
        def submit_if_current(self,*args): return store.submit_if_current(*args)
        def commit_if_current(self,*args): return store.commit_if_current(*args)
    denied=asyncio.run(service.interpret(req,'owner-a',scope,scope['consent_id'],RevokeBetweenChecks()))
    assert denied['status']=='session_inactive' and calls==[]

    # A fresh consent permits a boundary callback. Revocation starts while that
    # callback holds the fence and cannot return until the submission handoff ends.
    scope=store.grant_consent('owner-a',created['session_token'],PURPOSE,NOTICE_VERSION,True)
    callback_entered=threading.Event();release_boundary=threading.Event();revoke_done=threading.Event()
    class BoundaryProvider:
        def __call__(self,_payload):
            calls.append('boundary-entered');callback_entered.set()
            assert release_boundary.wait(3)
            async def response():
                await asyncio.sleep(.05)
                return {'word':'next','reason':'context_fit'}
            return response()
    service=ci.InterpretationService(BoundaryProvider())
    result_box=[]
    def run_service():
        result_box.append(asyncio.run(service.interpret(req,'owner-a',scope,scope['consent_id'],store)))
    worker=threading.Thread(target=run_service);worker.start()
    assert callback_entered.wait(3)
    def revoke():
        store.revoke_consent('owner-a',created['session_token']);revoke_done.set()
    revoker=threading.Thread(target=revoke);revoker.start()
    time.sleep(.05)
    assert not revoke_done.is_set(), 'revocation returned before the fenced submission callback ended'
    release_boundary.set();worker.join(3);revoker.join(3)
    assert not worker.is_alive() and not revoker.is_alive()
    assert revoke_done.is_set() and calls==['boundary-entered']
    assert result_box[0]['status']=='session_inactive'
    assert result_box[0]['llm_suggestion'] is None and not service.cache

    # Once revoke has returned, a later request cannot cross the provider boundary.
    after=asyncio.run(service.interpret(req,'owner-a',scope,scope['consent_id'],store))
    assert after['status']=='session_inactive' and calls==['boundary-entered']


def test_cache_eviction_is_thread_safe_with_concurrent_access():
    import app.services.context_interpretation as ci
    service=ci.InterpretationService()
    errors=[]
    def writer():
        try:
            for i in range(5000):
                key=('actor',f'session-{i%5}',None,'model',str(i))
                service._cache_put(key,float(i),object())
                service._cache_get(key)
                service.evict_session('actor',f'session-{(i+1)%5}')
        except Exception as error: errors.append(error)
    threads=[threading.Thread(target=writer) for _ in range(4)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert not errors
    assert len(service.cache)<=128
    assert all(len(key)>1 for key in service.cache)


def test_cache_capacity_expiry_and_cancellation_preserve_visual(tmp_path,monkeypatch):
    import app.services.context_interpretation as ci
    monkeypatch.setenv('LIPNET_LLM_ENABLED','1');monkeypatch.setenv('OPENAI_API_KEY','test-only-no-provider-call')
    path=tmp_path/'feedback.sqlite3';legacy_db(path);migrate(path)
    store=ContextSessionStore(path);a=store.create('owner-a')
    scope=store.grant_consent('owner-a',a['session_token'],PURPOSE,NOTICE_VERSION,True)
    now=[0.]
    req=ci.InterpretationRequest(mode='word',event_id='e',visual_candidate='hello',original_word=None,
        visual_uncertain=True,candidates=[{'word':'hello','prob':.55},{'word':'next','prob':.35}],context='menu',history=[],consent_text=True)
    async def run():
        async def provider(_):return {'word':'next','reason':'context_fit'}
        svc=ci.InterpretationService(provider,clock=lambda:now[0])
        first=await svc.interpret(req,'owner-a',scope,scope['consent_id'],store)
        assert first['safe_prediction'] is None and len(svc.cache)==1
        now[0]=61
        await svc.interpret(req,'owner-a',scope,scope['consent_id'],store)
        assert len(svc.cache)==1
        svc.evict_session('owner-a',scope['session_id']);assert not svc.cache
    asyncio.run(run())


def test_cache_capacity_remains_128(tmp_path,monkeypatch):
    import app.services.context_interpretation as ci
    monkeypatch.setenv('LIPNET_LLM_ENABLED','1');monkeypatch.setenv('OPENAI_API_KEY','test-only-no-provider-call')
    path=tmp_path/'feedback.sqlite3';legacy_db(path);migrate(path)
    store=ContextSessionStore(path);created=store.create('owner-a')
    scope=store.grant_consent('owner-a',created['session_token'],PURPOSE,NOTICE_VERSION,True)
    now=[0.]
    req=ci.InterpretationRequest(mode='word',event_id='e',visual_candidate='hello',original_word=None,
        visual_uncertain=True,candidates=[{'word':'hello','prob':.55},{'word':'next','prob':.35}],context='menu',history=[],consent_text=True)
    async def run():
        async def provider(_):return {'word':'next','reason':'context_fit'}
        service=ci.InterpretationService(provider,clock=lambda:now[0])
        for index in range(129):
            now[0]+=2.1
            data=req.model_dump();data['context']=f'menu-{index}'
            await service.interpret(ci.InterpretationRequest.model_validate(data),'owner-a',scope,scope['consent_id'],store)
        assert len(service.cache)==128
    asyncio.run(run())
