import test from 'node:test'
import assert from 'node:assert/strict'
import {compactLandmarks,captureGapExceeded,utteranceStartIndex,MP68,UtterancePublication} from '../src/live-demo/wordCapture.ts'

test('capture transmits actual anatomical landmarks and rejects off-frame points',()=>{
  const measured=Array.from({length:478},(_,i)=>({x:i/500,y:.4}))
  const compact=compactLandmarks(measured)
  assert.equal(compact.length,68)
  assert.deepEqual(compact[48],[measured[61].x,.4])
  assert.equal(new Set(MP68).size,68)
  measured[61].x=1.1
  assert.equal(compactLandmarks(measured),null)
})
test('one utterance produces at most one history entry even if word changes',()=>{
  const state=new UtterancePublication()
  assert.equal(state.claim(3),true)
  assert.equal(state.claim(3),false)
  assert.equal(state.claim(4),true)
  state.reset()
  assert.equal(state.claim(3),true)
})
test('timing discontinuities start a fresh capture rather than bridge missing speech',()=>{
  assert.equal(captureGapExceeded(undefined,500),false)
  assert.equal(captureGapExceeded(100,260),false)
  assert.equal(captureGapExceeded(100,261),true)
  assert.equal(captureGapExceeded(100,99),true)
})
test('word preroll follows milliseconds across different capture rates',()=>{
  assert.equal(utteranceStartIndex([0,40,80,120,160,200,240,280,320],320),2)
  assert.equal(utteranceStartIndex([0,80,160,240,320],320),1)
})
