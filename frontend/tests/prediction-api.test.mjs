import test from 'node:test'
import assert from 'node:assert/strict'
import { predictionJson, parseWordPrediction, requestWordPrediction, probeWordHealth } from '../src/lib/api.ts'
const valid = { task:'isolated_word', predicted_word:'hello', uncertain:false, confidence:.8, top_predictions:[{word:'hello',prob:.8}] }

test('response boundary preserves uncertainty, accepts absent confidence, and rejects malformed model output', () => {
  assert.equal(parseWordPrediction({...valid, uncertain:true}).uncertain, true)
  assert.equal(parseWordPrediction({...valid, confidence:undefined}).confidence, null)
  for(const extra of [{task:'sentence'},{predicted_word:'hello there'},{confidence:NaN},{confidence:1.1},{uncertain:undefined},
    {top_predictions:[{word:'hello',prob:-1}]},{latency_ms:-2},{diagnostics:{temporal_consensus:{confirmed:'yes'}}},{performance:{memory_after:null}}]) {
    assert.throws(()=>parseWordPrediction({...valid,...extra}), /invalid/)
  }
})

test('one POST uses the existing endpoint and exact metadata', async t => {
  const calls=[]
  t.mock.method(globalThis,'fetch',async(...args)=>{calls.push(args);return {ok:true,json:async()=>valid}})
  const capture={input_kind:'webcam',convention:'lrw-meanface-mediapipe68-v1',final:true,timestamps_ms:[1,2]}
  await requestWordPrediction(['real encoded input'],undefined,'lrw500',capture)
  assert.equal(calls.length,1);assert.equal(calls[0][0],'/api/predict/word/lrw500')
  assert.deepEqual(JSON.parse(calls[0][1].body),{frames:['real encoded input'],...capture})
})

test('each word mode uses its own prediction and health routes without replacing the original eight-word mode', async t => {
  const calls = [], vocabulary = ['a','bye','can','cat','choose','connection','demo','dog','hello','here','is','lips','my','navigation','next','previous','read','start','stop','you']
  const health = { model_loaded:true, inference_ready:true, model_kind:'Test fixture only', vocabulary,
    research_activation_gate_passed:true, evaluation_correct:17, evaluation_samples:20 }
  t.mock.method(globalThis,'fetch',async(url, init)=>{ calls.push([url, init]); return {ok:true,json:async()=>init?.method === 'POST' ? valid : health} })
  for (const [mode, prediction, readiness] of [
    ['word','/api/predict/word','/api/word/health'],
    ['expanded','/api/predict/word/expanded','/api/word/expanded/health'],
    ['lrw500','/api/predict/word/lrw500','/api/word/lrw500/health'],
  ]) {
    await requestWordPrediction(['actual encoded request payload'], undefined, mode)
    const result = await probeWordHealth(undefined, mode)
    assert.equal(calls.at(-2)[0], prediction)
    assert.equal(calls.at(-1)[0], readiness)
    assert.deepEqual(result.vocabulary, vocabulary)
    assert.equal(result.evaluation_correct, 17)
    assert.equal(result.evaluation_samples, 20)
  }
  const requests = calls.length
  await assert.rejects(requestWordPrediction([], undefined, 'alphabet'), /word recognition mode/)
  assert.equal(calls.length, requests, 'Alphabet mode must not silently dispatch to the eight-word endpoint')
})

test('expanded health unavailability does not reuse eight-word readiness or metrics', async t => {
  t.mock.method(globalThis,'fetch',async()=>({ok:false,status:503}))
  const result = await probeWordHealth(undefined, 'expanded')
  assert.equal(result.inference_ready, false)
  assert.equal(result.model_kind, 'Combined 20-word model')
  assert.deepEqual(result.vocabulary, [])
  assert.equal(result.evaluation_correct, undefined)
  assert.equal(result.evaluation_samples, undefined)
})

test('timeout covers a stalled JSON body after headers', async t => {
  t.mock.method(globalThis,'fetch',async()=>({ok:true,json:()=>new Promise(()=>{})}))
  await assert.rejects(predictionJson('/api/predict/word',{},undefined,15),/timed out/)
})

test('timeout also covers an HTTP error body and a noncooperative fetch', async t => {
  t.mock.method(globalThis,'fetch',async()=>({ok:false,status:500,json:()=>new Promise(()=>{})}))
  await assert.rejects(predictionJson('/word',{},undefined,15),/timed out/)
  globalThis.fetch=()=>new Promise(()=>{})
  await assert.rejects(predictionJson('/word',{},undefined,15),/timed out/)
})

test('abort before/during fetch cancels and malformed JSON or HTTP errors stay explicit', async t => {
  let requests=0
  t.mock.method(globalThis,'fetch',async()=>{requests++;return {ok:true,json:()=>new Promise(()=>{})}})
  const a=new AbortController();a.abort()
  await assert.rejects(predictionJson('/word',{},a.signal),{name:'AbortError'});assert.equal(requests,0)
  const b=new AbortController(), pending=predictionJson('/word',{},b.signal);b.abort()
  await assert.rejects(pending,{name:'AbortError'})
  globalThis.fetch=async()=>({ok:true,json:async()=>{throw new SyntaxError('broken')}})
  await assert.rejects(predictionJson('/word',{}),/invalid JSON/)
  globalThis.fetch=async()=>({ok:false,status:429,json:async()=>({detail:'Word model is busy'})})
  await assert.rejects(predictionJson('/word',{}),e=>e.status===429 && /busy/.test(e.message))
})
