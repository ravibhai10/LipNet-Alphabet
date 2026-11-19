import test from 'node:test'
import assert from 'node:assert/strict'
import { LRW_VOCABULARY } from '../src/data/lrwVocabulary.ts'

test('LRW-500 Explorer: Vocabulary integrity and cluster distributions', () => {
  assert.ok(LRW_VOCABULARY.length >= 30, 'Corpus sample contains representative classes')

  // Every word has standard 29 frames @ 25 FPS
  for (const item of LRW_VOCABULARY) {
    assert.equal(item.frames, 29, `${item.word} must have exactly 29 frames`)
    assert.ok(item.trainClips > 500, `${item.word} has sufficient training clips`)
    assert.equal(item.valClips, 50, `${item.word} validation partition must be 50`)
    assert.equal(item.testClips, 50, `${item.word} test partition must be 50`)
    assert.ok(item.phonemes.startsWith('/'), `${item.word} must have IPA phoneme transcript`)
  }

  // Check cluster representation
  const clusters = new Set(LRW_VOCABULARY.map(w => w.cluster))
  assert.ok(clusters.has('Bilabial'), 'Must contain Bilabials (P/B/M)')
  assert.ok(clusters.has('Labiodental'), 'Must contain Labiodentals (F/V)')
  assert.ok(clusters.has('Alveolar'), 'Must contain Alveolars (T/D/N/L)')
  assert.ok(clusters.has('Velar'), 'Must contain Velars (K/G)')
  assert.ok(clusters.has('Vocalic'), 'Must contain Vocalic clusters')
})
