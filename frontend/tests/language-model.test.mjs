import test from 'node:test'
import assert from 'node:assert/strict'
import { predictNextWords, rerankCandidates } from '../src/live-demo/languageModel.ts'

test('LanguageModel: predicts next words given previous context', () => {
  // Empty context returns high-frequency starters
  const defaults = predictNextWords([], 4)
  assert.equal(defaults.length, 4, 'Should return 4 suggestions')
  assert.ok(defaults.some(s => s.word === 'THE' || s.word === 'PEOPLE'))

  // Contextual lookup after 'PEOPLE'
  const afterPeople = predictNextWords(['PEOPLE'], 4)
  assert.equal(afterPeople.length, 4)
  assert.ok(afterPeople.some(s => ['WHO', 'HAVE', 'ARE', 'IN'].includes(s.word)), 'Should predict grammatically sound continuations')

  // Contextual lookup after 'SECURITY'
  const afterSecurity = predictNextWords(['NATIONAL', 'SECURITY'], 4)
  assert.equal(afterSecurity.length, 4)
  assert.ok(afterSecurity.some(s => ['COUNCIL', 'FORCES', 'SERVICES', 'SYSTEM'].includes(s.word)))
})

test('LanguageModel: rerankCandidates boosts contextual homophenes', () => {
  const candidates = [
    { word: 'SYSTEM', prob: 0.50 },
    { word: 'AFFAIRS', prob: 0.45 },
  ]

  // When preceded by 'FINANCIAL', 'SYSTEM' should receive boost
  const rerankedFinancial = rerankCandidates(candidates, 'FINANCIAL')
  assert.ok(rerankedFinancial[0].word === 'SYSTEM')
  assert.ok(rerankedFinancial[0].contextBoost > 0)

  // When preceded by 'FOREIGN', 'AFFAIRS' should receive boost
  const rerankedForeign = rerankCandidates(candidates, 'FOREIGN')
  assert.ok(rerankedForeign[0].word === 'AFFAIRS')
  assert.ok(rerankedForeign[0].contextBoost > 0)
})
