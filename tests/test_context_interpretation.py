"""Text-only policy/transport tests; no models, secrets or remote calls."""
import asyncio
import sqlite3
import json
import pytest
import httpx
from fastapi import FastAPI
from pydantic import ValidationError
from app.services import context_interpretation as m
from app.context_routes import router
from app.security import RequestBoundary, required_role


def request(**changes):
    return m.InterpretationRequest.model_validate(dict(mode='word', event_id='word_1',
        visual_candidate='hello', original_word=None, visual_uncertain=True,
        candidates=[{'word':'hello','prob':.55},{'word':'next','prob':.35}],
        context='navigating a menu', history=['start'], consent_text=True, **changes))


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv('LIPNET_LLM_ENABLED','1')
    monkeypatch.setenv('OPENAI_API_KEY','test-only-not-a-real-key')


async def good(_payload):
    return {'word':'next','reason':'context_fit'}


def interpret(req, provider=good):
    class Session:
        def is_current(self,*_): return True
        def commit_if_current(self,*args): args[-1](); return True
        def submit_if_current(self,*args): return True,args[-1]()
    return asyncio.run(m.InterpretationService(provider).interpret(req,'test',
        {'session_id':'unit-session','participant_id':None},1,Session()))


@pytest.mark.parametrize('field,value',[
    ('candidates',[{'word':'hello','prob':float('nan')}]),
    ('candidates',[{'word':'hello','prob':.2},{'word':'next','prob':.7}]),
    ('candidates',[{'word':'hello','prob':.8},{'word':'next','prob':.7}]),
    ('candidates',[{'word':'hello','prob':.5},{'word':'hello','prob':.2}]),
    ('original_word','next'),('original_word','hello'),('history',['two words']),
    ('context','x'*241),('history',['start']*7),('frames',['private media']),
])
def test_strict_request(field,value):
    data=request().model_dump(); data[field]=value
    with pytest.raises(ValidationError): m.InterpretationRequest.model_validate(data)


@pytest.mark.parametrize('change,status',[
    ({'context':''},'no_genuine_context'),
    ({'history':['invented']},'invalid_vocabulary'),
    ({'mode':'alphabet'},'invalid_vocabulary'),
    ({'candidates':[{'word':'hello','prob':.2},{'word':'next','prob':.1}]},'insufficient_visual_evidence'),
])
def test_bypass(change,status):
    data=request().model_dump(); data.update(change)
    async def never(_): raise AssertionError('Provider must not run')
    result=interpret(m.InterpretationRequest.model_validate(data),never)
    assert result['status']==status
    assert result['safe_prediction'] is None and not result['applied']


def test_high_confidence_and_missing_configuration(monkeypatch):
    data=request().model_dump(); data.update(original_word='hello',visual_uncertain=False,
        candidates=[{'word':'hello','prob':.96},{'word':'next','prob':.03}])
    assert interpret(m.InterpretationRequest.model_validate(data))['status']=='high_confidence_preserved'
    monkeypatch.delenv('OPENAI_API_KEY')
    assert interpret(request())['status']=='unavailable'
    monkeypatch.setenv('OPENAI_API_KEY','fixture'); monkeypatch.setenv('LIPNET_LLM_ENABLED','0')
    assert interpret(request())['status']=='unavailable'


@pytest.mark.parametrize('word,reason,status',[
    ('next','context_fit','suggestion_only'),('invented','context_fit','rejected_outside_top_k'),
    (None,'insufficient_context','llm_abstained'),('hello','keep_original','llm_abstained'),
])
def test_suggestions_never_replace_visual_prediction(word,reason,status):
    async def provider(_): return {'word':word,'reason':reason}
    result=interpret(request(),provider)
    assert result['status']==status
    assert result['original_word'] is None and result['safe_prediction'] is None and result['applied'] is False
    if status=='suggestion_only':
        assert result['suggested_visual_score']==.35 and result['requires_confirmation']


def test_low_visual_score_and_malformed_provider():
    data=request().model_dump(); data['candidates'][1]['prob']=.09
    assert interpret(m.InterpretationRequest.model_validate(data))['status']=='rejected_low_visual_score'
    async def malformed(_): return {'word':'next','reason':'context_fit','confidence':.99}
    assert interpret(request(),malformed)['status']=='provider_unavailable'
    async def failure(_): raise RuntimeError('private provider error')
    assert interpret(request(),failure)['status']=='provider_unavailable'


def test_cache_actor_isolation_expiry_and_cooldown():
    async def run():
        now=[10.]; calls=[]
        async def provider(payload): calls.append(payload); return await good(payload)
        service=m.InterpretationService(provider,clock=lambda:now[0])
        class Session:
            def is_current(self,*_): return True
            def commit_if_current(self,*args): args[-1](); return True
            def submit_if_current(self,*args): return True,args[-1]()
        scope={'session_id':'s-a','participant_id':None}
        a=await service.interpret(request(),'a',scope,1,Session()); b=await service.interpret(request(),'a',scope,1,Session())
        assert not a['cached'] and b['cached'] and len(calls)==1
        await service.interpret(request(),'b',{'session_id':'s-b','participant_id':None},1,Session()); assert len(calls)==2
        data=request().model_dump(); data['context']='different context'
        assert (await service.interpret(m.InterpretationRequest.model_validate(data),'a',scope,1,Session()))['status']=='rate_limited'
        now[0]+=61
        assert not (await service.interpret(request(),'a',scope,1,Session()))['cached']
        assert len(calls)==3
    asyncio.run(run())


def test_concurrency_and_cancellation_release_slots():
    async def run():
        entered=asyncio.Event()
        async def blocked(_): entered.set(); await asyncio.Event().wait()
        service=m.InterpretationService(blocked)
        class Session:
            def is_current(self,*_): return True
            def commit_if_current(self,*args): args[-1](); return True
            def submit_if_current(self,*args): return True,args[-1]()
        task=asyncio.create_task(service.interpret(request(),'a',{'session_id':'s-a'},1,Session())); await entered.wait()
        assert (await service.interpret(request(),'a',{'session_id':'s-a'},1,Session()))['status']=='busy'
        task2=asyncio.create_task(service.interpret(request(),'b',{'session_id':'s-b'},1,Session())); await asyncio.sleep(0)
        assert (await service.interpret(request(),'c',{'session_id':'s-c'},1,Session()))['status']=='busy'
        task.cancel(); task2.cancel()
        await asyncio.gather(task,task2,return_exceptions=True)
        assert not service.busy and service.active==0
    asyncio.run(run())


def test_provider_timeout_is_bounded():
    async def blocked(_): await asyncio.Event().wait()
    result=interpret(request(),blocked)
    assert result['status']=='provider_unavailable' and 1400<=result['latency_ms']<2200


@pytest.mark.parametrize('status,content,raises',[
    ('completed',[{'type':'output_text','text':'{"word":"next","reason":"context_fit"}'}],False),
    ('incomplete',[],True),('completed',[{'type':'refusal','refusal':'No'}],True),
    ('completed',[{'type':'output_text','text':'not json'}],True),
])
def test_responses_transport_schema_and_privacy(monkeypatch,status,content,raises):
    def handle(body,key):
        assert key=='test-only-not-a-real-key'
        assert body['model']==m.MODEL and body['store'] is False
        assert body['reasoning']=={'effort':'none'}
        schema=body['text']['format']['schema']
        assert schema['properties']['word']['enum']==['hello','next',None]
        assert set(json.loads(body['input']))=={'mode','candidates','context','history'}
        return {'status':status,'output':[{'type':'message','content':content}]}
    monkeypatch.setattr(m,'_post_json',handle)
    payload={k:v for k,v in request().model_dump().items() if k in {'mode','candidates','context','history'}}
    if raises:
        with pytest.raises(ValueError): asyncio.run(m.openai_suggestion(payload))
    else: assert asyncio.run(m.openai_suggestion(payload)).word=='next'


def test_private_route_fail_closed_and_body_limit(monkeypatch,tmp_path):
    from pipeline.adaptive.context_sessions import migrate
    from app.services.context_sessions import ContextSessionStore, PURPOSE, NOTICE_VERSION
    db=tmp_path/'feedback.sqlite3';con=__import__('sqlite3').connect(db)
    con.executescript("CREATE TABLE participants(owner TEXT NOT NULL,id TEXT NOT NULL,verified_by TEXT,permission TEXT,PRIMARY KEY(owner,id)); CREATE TABLE captures(id TEXT PRIMARY KEY); PRAGMA user_version=0;");con.close()
    migrate(db)
    from app import context_routes
    monkeypatch.setattr(context_routes,'context_session_store',ContextSessionStore(db))
    monkeypatch.delenv('LIPNET_ACCESS_FILE',raising=False)
    monkeypatch.delenv('LIPNET_REVIEW_TOKEN',raising=False)
    monkeypatch.delenv('LIPNET_FEEDBACK_TOKEN',raising=False)
    app=FastAPI(); app.include_router(router); app.add_middleware(RequestBoundary)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            assert (await client.get('/api/context/health')).status_code==200
            assert (await client.post('/api/context/interpret',content='bad json')).status_code==503
            monkeypatch.setenv('LIPNET_FEEDBACK_TOKEN','fixture-only-'+'x'*32)
            assert (await client.post('/api/context/interpret',json=request().model_dump())).status_code==401
            headers={'Authorization':'Bearer '+'fixture-only-'+'x'*32}
            assert (await client.post('/api/context/interpret',json=request().model_dump(),headers=headers)).status_code==401
            created=await client.post('/api/context/sessions',json={},headers=headers)
            session_token=created.json()['session_token']; scoped={**headers,'X-Interpretation-Session':session_token}
            assert (await client.post('/api/context/interpret',json=request().model_dump(),headers=scoped)).status_code==403
            grant=await client.post('/api/context/sessions/consent',json={'purpose':PURPOSE,'notice_version':NOTICE_VERSION,'granted':True},headers=scoped)
            assert grant.status_code==200
            data=request().model_dump(); data['context']=''
            reply=await client.post('/api/context/interpret',json=data,headers=scoped)
            assert reply.status_code==200 and reply.json()['status']=='no_genuine_context'
            assert (await client.post('/api/context/interpret',content='x'*8193,headers=headers)).status_code==413
            feedback=await client.post('/api/context/feedback',json={'mode':'word','event_id':'word_1','word':'next','accepted':True},headers=headers)
            assert feedback.json()=={'recorded':True,'visual_prediction_changed':False,'used_for_training':False}
            assert (await client.post('/api/context/feedback',json={'mode':'word','event_id':'word_1','word':'invented','accepted':True},headers=headers)).status_code==422
    asyncio.run(run())


def test_stdlib_transport_limit_tls_endpoint_and_no_redirect(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self,limit):
            assert limit==65537
            return b'{"status":"completed"}'
    class Opener:
        def open(self,req,timeout):
            assert req.full_url=='https://api.openai.com/v1/responses'
            assert req.get_header('Authorization')=='Bearer fixture'
            assert json.loads(req.data)=={'test':'text only'}
            assert timeout==1.2
            return Response()
    def factory(*handlers):
        assert handlers[0].proxies=={} and isinstance(handlers[1],m.NoRedirect)
        return Opener()
    monkeypatch.setattr(m.urllib.request,'build_opener',factory)
    assert m._post_json({'test':'text only'},'fixture')=={'status':'completed'}
    monkeypatch.setattr(Response,'read',lambda self,limit:b'x'*65537)
    with pytest.raises(ValueError): m._post_json({'test':'text only'},'fixture')
    assert required_role('/api/context/interpret','POST')=='collector'
    assert required_role('/api/predict/word','POST') is None


@pytest.mark.parametrize('mode,words',[('alphabet',['A','B']),('lrw500',['AFTERNOON','START'])])
def test_uncalibrated_tasks_do_not_call_provider(mode,words):
    data=request().model_dump(); data.update(mode=mode,visual_candidate=words[0],history=[],
        candidates=[{'word':words[0],'prob':.55},{'word':words[1],'prob':.35}])
    assert interpret(m.InterpretationRequest.model_validate(data))['status']=='uncalibrated_task'


def test_complete_endpoint_with_mock_provider(monkeypatch,tmp_path):
    from app import context_routes
    from app.security import access
    from pipeline.adaptive.context_sessions import migrate
    from app.services.context_sessions import ContextSessionStore, PURPOSE, NOTICE_VERSION
    db=tmp_path/'feedback.sqlite3';con=sqlite3.connect(db)
    con.executescript("CREATE TABLE participants(owner TEXT NOT NULL,id TEXT NOT NULL,verified_by TEXT,permission TEXT,PRIMARY KEY(owner,id)); CREATE TABLE captures(id TEXT PRIMARY KEY); PRAGMA user_version=0;");con.close()
    migrate(db);monkeypatch.setattr(context_routes,'context_session_store',ContextSessionStore(db))
    service=m.InterpretationService(good)
    monkeypatch.setattr(context_routes,'interpretation_service',service)
    app=FastAPI(); app.include_router(router)
    app.dependency_overrides[access]=lambda:{'actor':'fixture','role':'collector'}
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            headers={'X-Interpretation-Session':'fixture-token'}
            created=context_routes.context_session_store.create('fixture')
            headers['X-Interpretation-Session']=created['session_token']
            context_routes.context_session_store.grant_consent('fixture',created['session_token'],PURPOSE,NOTICE_VERSION,True)
            response=await client.post('/api/context/interpret',json=request().model_dump(),headers=headers)
            assert response.status_code==200
            result=response.json()
            assert result['suggested_word']=='next' and result['safe_prediction'] is None
            assert result['visual_confidence']==.55 and result['applied'] is False
            assert result['llm_suggestion']=='next'
    asyncio.run(run())
