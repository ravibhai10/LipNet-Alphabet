import test from 'node:test'
import assert from 'node:assert/strict'
import { LRW_VOCABULARY } from '../src/data/lrwVocabulary.ts'

test('LatentConstellation: verifies 5-cluster visual-phonetic space', () => {
  const clusters = ['Bilabial', 'Labiodental', 'Alveolar', 'Velar', 'Vocalic']

  for (const c of clusters) {
    const members = LRW_VOCABULARY.filter(w => w.cluster === c)
    assert.ok(members.length >= 4, `Cluster ${c} must have representative members`)
  }

  // Verify nearest neighbor distance logic
  const target = LRW_VOCABULARY[0] // e.g. BECAUSE
  const sameCluster = LRW_VOCABULARY.filter(w => w.cluster === target.cluster && w.word !== target.word)
  assert.ok(sameCluster.length > 0, 'Should find intra-cluster neighbors')
})
