// Collection software fixtures, not real labelled speech.
import test from 'node:test'
import assert from 'node:assert/strict'
import { collectionCounts, validateCollection, validateRecording, EIGHT_WORDS, digest } from '../src/live-demo/wordCollection.ts'
const row=(extra={})=>({id:'clip-1',label:'hello',speaker_id:'person',session_id:'train-session',split:'train',frames:Array(8).fill('data:image/jpeg;base64,fixture'),timestamps_ms:Array.from({length:8},(_,i)=>i*66),duration_ms:462,recorded_at:'2026-10-30T12:00:00Z',preprocessing:'browser-eyeaxis96-jpeg-v1',landmarker_sha256:'a'.repeat(64),frames_sha256:'b'.repeat(64),source:'user_camera',label_confirmed:true,consent:true,...extra})
test('recordings require explicit labels, consent, bounded frames, timestamps and matching convention',()=>{
  assert.doesNotThrow(()=>validateRecording(row()))
  for(const change of [{consent:false},{label_confirmed:false},{label:'bye'},{id:'../unsafe'},{frames:[]},{timestamps_ms:Array(8).fill(0)},{duration_ms:5001},{landmarker_sha256:''}])assert.throws(()=>validateRecording(row(change)))
})
test('a recording session cannot cross train/validation/test and duplicate content is rejected',()=>{
  assert.throws(()=>validateCollection([row(),row({id:'clip-2',frames_sha256:'c'.repeat(64),split:'test'})]),/session cannot cross/)
  assert.throws(()=>validateCollection([row(),row({id:'clip-2'})]),/duplicate/)
  assert.doesNotThrow(()=>validateCollection([row(),row({id:'clip-2',session_id:'test-session',split:'test',frames_sha256:'c'.repeat(64)})]))
})
test('progress counts only actual saved labels in the selected partition',()=>{
  const counts=collectionCounts([row(),row({split:'valid'}),row({label:'choose'})],'train')
  assert.equal(counts.hello,1);assert.equal(counts.choose,1);assert.equal(counts.next,0);assert.equal(Object.keys(counts).length,8)
  assert.deepEqual(EIGHT_WORDS,['choose','connection','hello','navigation','next','previous','start','stop'])
})
test('content SHA-256 is computed from bytes',async()=>{
  assert.equal(await digest(new TextEncoder().encode('abc')),'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
})
