import test from 'node:test'
import assert from 'node:assert/strict'
// The helper imports the typed API boundary through Vite in production. Read just
// the pure decision/window functions for native Node tests, matching existing tests.
import { readFile } from 'node:fs/promises'
import { stripTypeScriptTypes } from 'node:module'
const source=await readFile(new URL('../src/live-demo/adaptiveFeedback.ts',import.meta.url),'utf8')
const pure=source.slice(source.indexOf('export function feedbackDecision'))
const {feedbackDecision,validSourceWindow}=await import(`data:text/javascript;base64,${Buffer.from(stripTypeScriptTypes(pure)).toString('base64')}`)
const prediction={prediction:'hello',candidate:'hello'}
test('correct confirmation requires the accepted label',()=>{
  assert.deepEqual(feedbackDecision(prediction,'correct','hello',['hello']),{verdict:'correct',label:'hello'})
  assert.throws(()=>feedbackDecision(prediction,'correct','stop',['stop']))
})
test('incorrect feedback needs an actual correction',()=>{
  assert.throws(()=>feedbackDecision(prediction,'incorrect','',['hello']))
  assert.throws(()=>feedbackDecision(prediction,'incorrect','hello',['hello']))
  assert.deepEqual(feedbackDecision(prediction,'incorrect','stop',['hello','stop']),{verdict:'incorrect',label:'stop'})
})
test('unknown stays explicit; task labels cannot cross',()=>{
  assert.equal(feedbackDecision(prediction,'incorrect','UNKNOWN',['hello']).label,'UNKNOWN')
  assert.throws(()=>feedbackDecision(prediction,'incorrect','A',['hello']))
})
test('uncertain result cannot be confirmed',()=>{
  assert.throws(()=>feedbackDecision({...prediction,prediction:null},'correct','hello',['hello']))
  assert.equal(feedbackDecision({...prediction,prediction:null},'incorrect','hello',['hello']).label,'hello')
})
test('complete chronological source window accepts genuine sample timestamps',()=>{
  assert.doesNotThrow(()=>validSourceWindow(Array.from({length:12},(_,i)=>i*100),2000))
})
test('invalid, incomplete, stalled or oversized source windows fail',()=>{
  for(const times of [[],[0,1],[0,1,2,3,4,5,6,NaN],[0,0,1,2,3,4,5,6],Array.from({length:8},(_,i)=>i*500)])assert.throws(()=>validSourceWindow(times,1))
  assert.throws(()=>validSourceWindow(Array.from({length:8},(_,i)=>i*100),100*1024*1024))
})
