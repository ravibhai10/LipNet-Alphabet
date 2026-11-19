import test from 'node:test'
import assert from 'node:assert/strict'
import { ContextClient, validateContextReply, recordInterpretation } from '../src/live-demo/contextInterpretation.ts'
const event={mode:'word',event_id:'word_1',visual_candidate:'hello',original_word:null,visual_uncertain:true,
  candidates:[{word:'hello',prob:.55},{word:'next',prob:.35}],context:'menu navigation',history:['start'],consent_text:true}
const reply={event_id:'word_1',visual_candidate:'hello',original_word:null,safe_prediction:null,visual_confidence:.55,
  suggested_word:'next',suggested_visual_score:.35,applied:false,requires_confirmation:true,status:'suggestion_only',latency_ms:10,cached:false}

test('strict response preserves visual output, scores and top-K',()=>{
  assert.equal(validateContextReply(reply,event),reply)
  for(const change of [{event_id:'old'},{safe_prediction:'next'},{applied:true},{suggested_word:'invented'},
    {visual_confidence:.9},{suggested_visual_score:.9},{latency_ms:NaN},{requires_confirmation:false},{cached:undefined}])
    assert.throws(()=>validateContextReply({...reply,...change},event))
})
test('only text metadata and local access token are posted',async t=>{
  let step=0
  t.mock.method(globalThis,'fetch',async(url,init)=>{
    assert.equal(init.headers.Authorization,'Bearer private-local-token')
    if(step++===0){assert.equal(url,'/api/context/sessions');return {ok:true,json:async()=>({session_token:'s'.repeat(43)})}}
    if(step===2){assert.equal(url,'/api/context/sessions/consent');assert.equal(init.headers['X-Interpretation-Session'],'s'.repeat(43));return {ok:true}}
    assert.equal(url,'/api/context/interpret'); assert.equal(init.headers['X-Interpretation-Session'],'s'.repeat(43))
    assert.deepEqual(JSON.parse(init.body),event);return {ok:true,json:async()=>reply}
  })
  const client=new ContextClient();await client.createSession('private-local-token');await client.grantConsent('private-local-token')
  assert.equal((await client.interpret(event,'private-local-token')).safe_prediction,null)
})
test('revocation and termination are session scoped and token stays in headers',async t=>{
  const calls=[]
  t.mock.method(globalThis,'fetch',async(url,init)=>{calls.push([url,init]);return {ok:true,json:async()=>({session_token:'z'.repeat(43)})}})
  const client=new ContextClient();await client.createSession('collector-key');await client.grantConsent('collector-key')
  await client.revokeConsent('collector-key');await client.terminate('collector-key')
  assert.deepEqual(calls.map(x=>x[0]),['/api/context/sessions','/api/context/sessions/consent',
    '/api/context/sessions/consent/revoke','/api/context/sessions/terminate'])
  assert.equal(calls[2][1].headers['X-Interpretation-Session'],'z'.repeat(43))
  assert.equal(calls[3][1].headers['X-Interpretation-Session'],'z'.repeat(43))
  assert.equal(JSON.stringify(calls.map(x=>x[1].body)).includes('z'.repeat(43)),false)
})
test('failed termination retains token for retry; success clears it',async t=>{
  const seen=[]
  const client=new ContextClient();client.sessionToken='retry-session-token'
  t.mock.method(globalThis,'fetch',async(_url,init)=>{
    seen.push(init.headers['X-Interpretation-Session'])
    return seen.length===1 ? {ok:false,status:503} : {ok:true,status:200}
  })
  await assert.rejects(client.terminate('collector'),/retry is available/)
  assert.equal(client.sessionToken,'retry-session-token')
  await client.terminate('collector')
  assert.deepEqual(seen,['retry-session-token','retry-session-token'])
  assert.equal(client.sessionToken,null)
})
test('overlapping and cancelled responses are discarded',async t=>{
  const pending=[]
  t.mock.method(globalThis,'fetch',()=>new Promise(resolve=>pending.push(resolve)))
  const client=new ContextClient();client.sessionToken='fixture-session-token'
  const first=client.interpret(event,'local'),second=client.interpret(event,'local')
  pending[1]({ok:true,json:async()=>reply}); assert.deepEqual(await second,reply)
  pending[0]({ok:true,json:async()=>reply}); assert.equal(await first,null)
  const third=client.interpret(event,'local'); client.cancel()
  pending[2]({ok:true,json:async()=>reply}); assert.equal(await third,null)
})
test('provider and malformed response failures retain original result',async t=>{
  const client=new ContextClient();client.sessionToken='fixture-session-token'
  t.mock.method(globalThis,'fetch',async()=>({ok:false,status:503}))
  await assert.rejects(client.interpret(event,'local'),/retained/)
})
test('non-cooperative transport has a bounded deadline',async t=>{
  const client=new ContextClient();client.sessionToken='fixture-session-token'
  t.mock.method(globalThis,'fetch',()=>new Promise(()=>{}))
  const start=performance.now()
  await assert.rejects(client.interpret(event,'local'),/timed out/)
  assert.ok(performance.now()-start<3000)
})
test('human feedback is a separate audit-only endpoint without context or media',async t=>{
  t.mock.method(globalThis,'fetch',async(url,init)=>{
    assert.equal(url,'/api/context/feedback')
    assert.deepEqual(JSON.parse(init.body),{mode:'word',event_id:'word_1',word:'next',accepted:true})
    return {ok:true}
  })
  await recordInterpretation('word',reply,true,'local')
})
