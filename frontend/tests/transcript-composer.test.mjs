import test from 'node:test'
import assert from 'node:assert/strict'

test('Transcript Composer: Confidence thresholds categorisation', () => {
  const getBadgeClass = (conf) => {
    if (conf >= 0.85) return 'chip-emerald'
    if (conf >= 0.60) return 'chip-amber'
    return 'chip-rose'
  }

  assert.equal(getBadgeClass(0.948), 'chip-emerald')
  assert.equal(getBadgeClass(0.85), 'chip-emerald')
  assert.equal(getBadgeClass(0.72), 'chip-amber')
  assert.equal(getBadgeClass(0.60), 'chip-amber')
  assert.equal(getBadgeClass(0.59), 'chip-rose')
  assert.equal(getBadgeClass(0.12), 'chip-rose')
})

test('Transcript Composer: Sentence accumulation and alternatives replacement', () => {
  let tape = [
    { id: 'w1', word: 'HELLO', confidence: 0.948, alternatives: [{ label: 'HELLO', score: 0.948 }] },
    { id: 'w2', word: 'WORLD', confidence: 0.912, alternatives: [{ label: 'WORLD', score: 0.912 }, { label: 'WORD', score: 0.05 }] },
  ]

  const getFullSentence = (words) => words.map(w => w.word).join(' ')
  assert.equal(getFullSentence(tape), 'HELLO WORLD')

  // Ingest new word
  tape.push({ id: 'w3', word: 'TODAY', confidence: 0.88, alternatives: [] })
  assert.equal(getFullSentence(tape), 'HELLO WORLD TODAY')

  // One-click correction: replace WORLD with WORD
  tape = tape.map(w => w.id === 'w2' ? { ...w, word: 'WORD', confidence: 1.0 } : w)
  assert.equal(getFullSentence(tape), 'HELLO WORD TODAY')
})
