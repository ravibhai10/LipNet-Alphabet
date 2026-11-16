// Software contract fixtures only; these do not measure recognition accuracy.
import test from 'node:test'
import assert from 'node:assert/strict'
import { WordAssistant, assistantReducer, initialAssistant } from '../src/live-demo/wordAssistant.ts'

const response = (extra = {}) => ({ task: 'isolated_word', predicted_word: 'hello', display_word: 'hello',
  confidence: .87, uncertain: false, top_predictions: [{ word: 'hello', prob: .87 }], all_probabilities: [], latency_ms: 120, ...extra })
const settled = () => new Promise(resolve => setImmediate(resolve))
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b }); return { promise, resolve, reject } }
function harness(overrides = {}) {
  let now = 0
  const calls = []
  const controller = new WordAssistant({ now: () => now, wallTime: () => 100000,
    request: async (...args) => { calls.push(args); return response() }, ...overrides })
  const ready = () => { controller.startCamera(); controller.cameraReady() }
  const clip = (prefix = 'clip', mode = 'word', count = mode === 'lrw500' ? 12 : 8) => {
    const start = now
    for (let i = 0; i < count; i++) {
      now = start + i * 50
      controller.sample({ image: `${prefix}-${i}`, time: now, landmarks: mode === 'lrw500' ? Array.from({ length: 68 }, () => [.5, .5]) : undefined })
    }
  }
  return { controller, calls, ready, clip, advance: ms => { now += ms } }
}

test('central reducer rejects invalid transitions and visits the full explicit cycle', async () => {
  const h = harness(), states = [], c = h.controller
  assert.equal(assistantReducer(initialAssistant(), { type: 'PROCESS', now: 0 }).phase, 'IDLE')
  c.subscribe(() => states.push(c.getSnapshot().phase))
  assert.equal(c.finish(), false); assert.equal(c.begin('word'), false)
  h.ready(); assert.equal(c.startCamera(), false)
  c.begin('word'); h.clip(); c.finish(); await settled()
  assert.equal(c.getSnapshot().phase, 'WAITING_NEXT')
  for (const state of ['CAMERA_STARTING','READY','CAPTURING','PROCESSING','RESULT','WAITING_NEXT']) assert.ok(states.includes(state), state)
  assert.equal(c.getSnapshot().result.word, 'hello')
  assert.equal(h.calls.length, 1)
  c.stop(); assert.ok(states.includes('STOPPING')); assert.equal(c.getSnapshot().phase, 'IDLE')
})

test('next and retry use independent buffers, unique IDs, one request and retained history', async () => {
  const h = harness(), c = h.controller
  h.ready(); c.begin('word'); assert.equal(c.begin('word'), false); h.clip('first')
  assert.equal(c.finish(), true); assert.equal(c.finish(), false); assert.equal(c.next('word'), false)
  await settled(); const first = c.getSnapshot().result.id
  h.advance(1000); assert.equal(h.calls.length, 1)
  assert.equal(c.next('word'), true); assert.equal(c.next('word'), false)
  assert.equal(c.getSnapshot().result, null); assert.equal(c.getSnapshot().frames, 0)
  h.clip('second'); c.finish(); await settled()
  assert.ok(h.calls[0][0].every(f => f.startsWith('first-')))
  assert.ok(h.calls[1][0].every(f => f.startsWith('second-')))
  assert.equal(c.getSnapshot().history.length, 2)
  h.advance(500); c.retry('word'); h.clip('retry'); c.finish(); await settled()
  const s = c.getSnapshot()
  assert.equal(s.totals.retries, 1); assert.equal(s.totals.attempts, 3)
  assert.equal(s.totals.predictions, 3); assert.equal(s.history.length, 3)
  assert.notEqual(s.history[0].id, first); assert.equal(s.history[0].retryOf, s.history[1].id)
  assert.equal(s.totals.inferenceTotalMs / s.totals.timedPredictions, 120)
})

test('stop, disconnect cancellation and disposal discard late responses', async () => {
  for (const action of ['stop', 'cancel', 'dispose']) {
    const pending = deferred(), h = harness({ request: () => pending.promise }), c = h.controller
    h.ready(); c.begin('word'); h.clip(); c.finish(); await settled()
    if (action === 'cancel') c.cancel('Camera disconnected')
    else c[action]()
    pending.resolve(response()); await settled()
    assert.equal(c.getSnapshot().history.length, 0)
    assert.notEqual(c.getSnapshot().phase, 'PROCESSING')
  }
})

test('stop retains result/history/feedback and reset explicitly clears the session', async () => {
  const h = harness(), c = h.controller
  h.ready(); c.begin('word'); h.clip(); c.finish(); await settled()
  c.send({ type: 'FEEDBACK', id: 1, feedback: 'incorrect' }); c.stop()
  assert.equal(c.getSnapshot().result.word, 'hello'); assert.equal(c.getSnapshot().history[0].feedback, 'incorrect')
  assert.equal(c.getSnapshot().totals.predictions, 1)
  h.ready(); assert.equal(c.getSnapshot().result.word, 'hello')
  c.send({ type: 'CLEAR_HISTORY' }); assert.equal(c.getSnapshot().history.length, 0)
  assert.equal(c.getSnapshot().totals.predictions, 1)
  c.reset(); assert.deepEqual(c.getSnapshot(), initialAssistant())
})

test('capture validation rejects no/too few frames, gaps, duplicates and oversized clips without a POST', () => {
  for (const invalid of ['empty','few','gap','repeat','too-long','too-many']) {
    const h = harness(), c = h.controller; h.ready(); c.begin(invalid === 'gap' ? 'lrw500' : 'word')
    if (invalid === 'few') h.clip('few', 'word', 3)
    if (invalid === 'gap' || invalid === 'repeat') {
      const landmarks = invalid === 'gap' ? Array.from({length:68},()=>[.5,.5]) : undefined
      c.sample({ image: 'one', time: 0, landmarks }); c.sample({ image: 'two', time: invalid === 'gap' ? 200 : 0, landmarks })
    }
    if (invalid === 'too-long') { h.advance(5001); c.tick() }
    if (invalid === 'too-many') h.clip('excess', 'word', 77)
    c.finish(); assert.equal(c.getSnapshot().phase, 'ERROR', invalid); assert.equal(h.calls.length, 0)
    c.recover(); assert.equal(c.getSnapshot().phase, 'READY')
  }
})

test('deadline and null/error transports leave processing safely and count failed requests', async () => {
  for (const request of [() => new Promise(() => {}), async () => null, () => { throw new Error('Service unavailable') }]) {
    const h = harness({ request, timeoutMs: 15 }), c = h.controller
    h.ready(); c.begin('word'); h.clip(); c.finish()
    await new Promise(r => setTimeout(r, 35))
    assert.equal(c.getSnapshot().phase, 'ERROR'); assert.equal(c.getSnapshot().totals.failedRequests, 1)
    assert.equal(c.getSnapshot().history.length, 0)
    c.stop()
  }
})

test('uncertainty and optional score filter only reject; missing measurements stay unavailable', async () => {
  for (const [extra, threshold, status] of [
    [{ uncertain: true }, .1, 'uncertain'], [{ confidence: .7 }, .8, 'uncertain'],
    [{ confidence: null, latency_ms: undefined }, null, 'available'],
  ]) {
    const h = harness({ request: async () => response(extra) }), c = h.controller
    h.ready(); c.send({ type: 'THRESHOLD', value: threshold }); c.begin('word'); h.clip(); c.finish(); await settled()
    assert.equal(c.getSnapshot().result.status, status)
    assert.equal(c.getSnapshot().result.confidence, 'confidence' in extra ? extra.confidence : .87)
    if (extra.confidence === null) { assert.equal(c.getSnapshot().result.inferenceMs, null); assert.equal(c.getSnapshot().totals.timedPredictions, 0) }
  }
})

test('LRW keeps real full-face metadata and final consensus, never reuses the eight-word input contract', async () => {
  for (const confirmed of [true,false]) {
    const calls = [], h = harness({ request: async (...args) => { calls.push(args); return response({ diagnostics: { temporal_consensus: { confirmed } } }) } }), c = h.controller
    h.ready(); c.begin('lrw500'); h.clip('face', 'lrw500'); c.finish(); await settled()
    const meta = calls[0][2]
    assert.equal(meta.final, true); assert.equal(meta.input_kind, 'webcam')
    assert.equal(meta.landmarks.length, 12); assert.equal(meta.timestamps_ms[11] - meta.timestamps_ms[0], 550)
    assert.ok(meta.face_counts.every(n => n === 1))
    assert.equal(c.getSnapshot().result.status, confirmed ? 'available' : 'uncertain')
  }
})

test('LRW rejects incomplete landmarks and unrealistic duration before requesting', () => {
  const h = harness(), c = h.controller; h.ready(); c.begin('lrw500')
  c.sample({ image: 'face', time: 0, landmarks: [] }); assert.equal(c.getSnapshot().phase, 'ERROR')
  c.retry('lrw500')
  for(let i=0;i<12;i++) c.sample({ image: 'face', time: i * 10, landmarks: Array.from({ length:68 },()=>[.5,.5]) })
  assert.equal(c.finish(), false); assert.equal(h.calls.length, 0)
})

test('expanded mode captures a fresh mouth clip and keeps its mode in results alongside eight-word history', async () => {
  const h = harness(), c = h.controller
  h.ready(); c.begin('word'); h.clip('original-eight'); c.finish(); await settled()
  assert.equal(c.next('expanded'), true)
  h.clip('new-twenty', 'expanded', 8); c.finish(); await settled()
  assert.equal(h.calls.length, 2)
  assert.ok(h.calls[1][0].every(frame => frame.startsWith('new-twenty-')))
  assert.equal(h.calls[1][2], undefined, 'Expanded mouth crops do not use LRW full-face metadata')
  assert.equal(c.getSnapshot().result.mode, 'expanded')
  assert.equal(c.getSnapshot().result.status, 'available')
  assert.deepEqual(c.getSnapshot().history.map(row => row.mode), ['expanded', 'word'])
})

test('stop before asynchronous dispatch sends no request', async () => {
  const h = harness(), c = h.controller; h.ready(); c.begin('word'); h.clip(); c.finish(); c.stop()
  await settled(); assert.equal(h.calls.length, 0)
})
