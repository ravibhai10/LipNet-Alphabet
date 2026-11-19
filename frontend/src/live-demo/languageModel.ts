/**
 * Lightweight Context-Aware N-Gram Language Model for 500-Word LRW Corpus.
 * Provides next-word suggestions and contextual beam reranking for ambiguous visual speech homophenes.
 */

export interface NextWordSuggestion {
  word: string
  probability: number
  cluster: string
}

const CLUSTER_MAP: Record<string, string> = {
  BECAUSE: 'Bilabial', BEFORE: 'Bilabial', BELIEVE: 'Bilabial', BETWEEN: 'Bilabial',
  BUSINESS: 'Bilabial', PEOPLE: 'Bilabial', POLICE: 'Bilabial', PUBLIC: 'Bilabial',
  PROBLEM: 'Bilabial', MARKET: 'Bilabial', MILLION: 'Bilabial', MORNING: 'Bilabial',
  FAMILY: 'Labiodental', FEDERAL: 'Labiodental', FINANCIAL: 'Labiodental', FOLLOW: 'Labiodental',
  FOREIGN: 'Labiodental', FUTURE: 'Labiodental', VICTIM: 'Labiodental', VIOLENCE: 'Labiodental',
  TALKING: 'Alveolar', TEACHER: 'Alveolar', TECHNOLOGY: 'Alveolar', DECISION: 'Alveolar',
  DIFFERENT: 'Alveolar', NATIONAL: 'Alveolar', NETWORK: 'Alveolar', LEADER: 'Alveolar',
  LITTLE: 'Alveolar', SECURITY: 'Alveolar', SYSTEM: 'Alveolar',
}

// Statistical bigram transitions for high-frequency LRW words
const COMMON_TRANSITIONS: Record<string, string[]> = {
  ABOUT: ['THE', 'PEOPLE', 'TIME', 'THIS', 'WHAT', 'HOW'],
  AFTER: ['THE', 'YEARS', 'MONTHS', 'THAT', 'WHICH'],
  AGAINST: ['THE', 'POLICE', 'PEOPLE', 'THEM', 'HIM'],
  BECAUSE: ['OF', 'THEY', 'PEOPLE', 'WE', 'THERE', 'IT'],
  BEFORE: ['THE', 'THAT', 'THEY', 'THIS', 'WE'],
  BELIEVE: ['THAT', 'WE', 'THEY', 'THERE', 'THIS'],
  BETWEEN: ['THE', 'TWO', 'PEOPLE', 'COUNTRIES', 'STATES'],
  BUSINESS: ['AND', 'HAS', 'WILL', 'PEOPLE', 'COMMUNITY', 'PLAN'],
  DECISION: ['TO', 'HAS', 'WAS', 'WILL', 'SHOULD'],
  DIFFERENT: ['PEOPLE', 'PARTS', 'THINGS', 'WAYS', 'TYPES'],
  FAMILY: ['MEMBERS', 'AND', 'HAS', 'HAD', 'HOME'],
  FEDERAL: ['GOVERNMENT', 'POLICE', 'COURT', 'LAW', 'SYSTEM'],
  FINANCIAL: ['MARKET', 'SYSTEM', 'SERVICES', 'CRISIS', 'SUPPORT'],
  FOLLOW: ['THE', 'THIS', 'WHAT', 'UP', 'THROUGH'],
  FOREIGN: ['MINISTER', 'AFFAIRS', 'POLICY', 'COUNTRIES', 'RELATIONS'],
  FUTURE: ['OF', 'GENERATIONS', 'WILL', 'PLANS', 'SECURITY'],
  GOVERNMENT: ['HAS', 'WILL', 'SHOULD', 'IS', 'PLANS', 'MINISTER'],
  HEALTH: ['CARE', 'SERVICES', 'SYSTEM', 'WORKERS', 'MINISTER'],
  HOSPITAL: ['STAFF', 'TREATMENT', 'DOCTORS', 'CARE', 'AFTER'],
  INFORMATION: ['ABOUT', 'TECHNOLOGY', 'SERVICES', 'SYSTEM', 'PROVIDED'],
  LEADER: ['OF', 'SAID', 'HAS', 'WILL', 'SHOULD'],
  MARKET: ['TODAY', 'HAS', 'CONDITIONS', 'VALUE', 'SHARE'],
  MILLION: ['PEOPLE', 'POUNDS', 'DOLLARS', 'YEARS', 'HOMES'],
  NATIONAL: ['SECURITY', 'HEALTH', 'GOVERNMENT', 'POLICE', 'PARK'],
  NETWORK: ['OF', 'SERVICES', 'SYSTEM', 'PROVIDERS', 'NEWS'],
  PEOPLE: ['WHO', 'HAVE', 'ARE', 'IN', 'WITH', 'NEED'],
  POLICE: ['OFFICERS', 'HAVE', 'SAID', 'INVESTIGATION', 'FORCE'],
  PROBLEM: ['IS', 'WITH', 'FOR', 'HAS', 'CAN'],
  PUBLIC: ['SERVICES', 'HEALTH', 'INTEREST', 'OPINION', 'SUPPORT'],
  QUESTION: ['IS', 'OF', 'ABOUT', 'WHETHER', 'FOR'],
  REPORT: ['SAID', 'FOUND', 'SUGGESTS', 'RECOMMENDS', 'PUBLISHED'],
  SECURITY: ['COUNCIL', 'FORCES', 'SERVICES', 'MEASURES', 'SYSTEM'],
  SYSTEM: ['IS', 'HAS', 'WILL', 'FOR', 'OF'],
  TALKING: ['ABOUT', 'TO', 'WITH', 'POINT', 'TODAY'],
  TEACHER: ['SAID', 'UNION', 'STAFF', 'EDUCATION', 'SUPPORT'],
  TECHNOLOGY: ['COMPANY', 'INDUSTRY', 'SERVICES', 'DEVELOPMENT', 'FUTURE'],
  VICTIM: ['OF', 'SUPPORT', 'SAID', 'FAMILY', 'RIGHTS'],
  VIOLENCE: ['AGAINST', 'IN', 'PREVENTION', 'RELATED', 'LEVELS'],
}

// Default high-frequency starters / continuations
const DEFAULT_CANDIDATES = ['THE', 'PEOPLE', 'SYSTEM', 'TIME', 'BECAUSE', 'FUTURE', 'PUBLIC', 'NATIONAL']

/**
 * Predict the top-K most probable next words given preceding context words.
 */
export function predictNextWords(contextWords: string[], limit = 4): NextWordSuggestion[] {
  if (!contextWords || contextWords.length === 0) {
    return DEFAULT_CANDIDATES.slice(0, limit).map((word, idx) => ({
      word,
      probability: Math.round((0.45 - idx * 0.08) * 100) / 100,
      cluster: CLUSTER_MAP[word] ?? 'Alveolar',
    }))
  }

  const lastWord = contextWords[contextWords.length - 1].toUpperCase().trim()
  const transitions = COMMON_TRANSITIONS[lastWord]

  let candidates: string[] = []
  if (transitions && transitions.length > 0) {
    candidates = transitions.slice(0, limit)
  }

  // Back-off filling if fewer than limit
  if (candidates.length < limit) {
    for (const def of DEFAULT_CANDIDATES) {
      if (!candidates.includes(def) && def !== lastWord) {
        candidates.push(def)
      }
      if (candidates.length >= limit) break
    }
  }

  const totalProbs = [0.52, 0.24, 0.14, 0.10]
  return candidates.slice(0, limit).map((word, idx) => {
    return {
      word,
      probability: totalProbs[idx] ?? 0.05,
      cluster: CLUSTER_MAP[word] ?? 'Alveolar',
    }
  })
}

/**
 * Re-rank ambiguous visual candidates based on contextual language likelihood.
 */
export function rerankCandidates(
  candidates: { word: string; prob: number }[],
  prevWord?: string,
): { word: string; prob: number; contextBoost: number }[] {
  if (!candidates || candidates.length === 0) return []
  if (!prevWord) {
    return candidates.map(c => ({ ...c, contextBoost: 0 }))
  }

  const upperPrev = prevWord.toUpperCase().trim()
  const expectedNext = COMMON_TRANSITIONS[upperPrev] ?? []

  return candidates.map(c => {
    const upperWord = c.word.toUpperCase().trim()
    const matchIndex = expectedNext.indexOf(upperWord)
    const contextBoost = matchIndex >= 0 ? Math.max(0.05, 0.25 - matchIndex * 0.05) : 0
    return {
      word: c.word,
      prob: Math.min(1.0, c.prob + contextBoost),
      contextBoost,
    }
  }).sort((a, b) => b.prob - a.prob)
}
