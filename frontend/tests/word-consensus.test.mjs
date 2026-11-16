import test from 'node:test'
import assert from 'node:assert/strict'
import { WordConsensus } from '../src/live-demo/wordConsensus.ts'

test('words require two accepted windows from the same utterance', () => {
  const state = new WordConsensus()
  assert.equal(state.update('hello', false, 1), false)
  assert.equal(state.update('hello', false, 1), true)
})
test('rejected predictions never confirm or accumulate', () => {
  const state = new WordConsensus()
  assert.equal(state.update('hello', true, 1), false)
  assert.equal(state.update('hello', true, 1), false)
  assert.equal(state.update('hello', false, 1), false)
})
test('a changed word starts a new consensus', () => {
  const state = new WordConsensus()
  state.update('hello', false, 1)
  assert.equal(state.update('stop', false, 1), false)
  assert.equal(state.update('stop', false, 1), true)
})
test('utterance changes and face-loss reset discard previous confirmation', () => {
  const state = new WordConsensus()
  state.update('hello', false, 1)
  state.update('hello', false, 1)
  assert.equal(state.update('hello', false, 2), false)
  state.reset()
  assert.equal(state.update('hello', false, 2), false)
})
