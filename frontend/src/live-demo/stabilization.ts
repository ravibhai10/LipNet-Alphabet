import type { Prediction } from '../lib/api'

export const LIVE_SEQUENCE = { maxFrames: 29, minFrames: 8, stride: 4 } as const
export const ACTIVITY_DEFAULTS = { window: 29, minSamples: 8, minRange: .045, minRecentEnergy: .008, holdMs: 350 } as const

/** A heuristic on measured aperture ratios; thresholds require held-out calibration. */
export class VisualActivityDetector {
  private samples: number[] = []
  private lastMovement = -Infinity
  private lastSample = -Infinity
  reset() { this.samples = []; this.lastMovement = -Infinity; this.lastSample = -Infinity }
  update(aperture: number, now: number) {
    if (!Number.isFinite(aperture) || !Number.isFinite(now)) throw new Error('Invalid measured aperture or timestamp.')
    if (now - this.lastSample > 500 || now < this.lastSample) this.reset()
    this.lastSample = now
    this.samples.push(aperture)
    if (this.samples.length > ACTIVITY_DEFAULTS.window) this.samples.shift()
    const range = Math.max(...this.samples) - Math.min(...this.samples)
    const recent = this.samples.slice(-6)
    const energy = recent.length > 1 ? Math.sqrt(recent.slice(1).reduce((sum, value, index) => sum + (value - recent[index]) ** 2, 0) / (recent.length - 1)) : 0
    const measured = this.samples.length >= ACTIVITY_DEFAULTS.minSamples && range >= ACTIVITY_DEFAULTS.minRange && energy >= ACTIVITY_DEFAULTS.minRecentEnergy
    if (measured) this.lastMovement = now
    return { speaking: now - this.lastMovement <= ACTIVITY_DEFAULTS.holdMs, energy, range, lastMovement: this.lastMovement }
  }
}

/** EMA only uses a complete returned probability vector; older APIs use top-1 consensus. */
export class PredictionStabilizer {
  private average = new Map<string, number>()
  private candidate: string | null = null
  private repeats = 0
  private windows = 0
  private alpha: number
  private confirmations: number
  constructor(alpha = .6, confirmations = 2) {
    if (!(alpha > 0 && alpha <= 1) || confirmations < 1) throw new Error('Invalid stabilization settings.')
    this.alpha = alpha; this.confirmations = confirmations
  }
  reset() { this.average.clear(); this.candidate = null; this.repeats = 0; this.windows = 0 }
  update(raw: Prediction): { confirmed: boolean; prediction: Prediction | null; windows: number } {
    this.windows++
    let prediction = raw
    const probabilities = raw.probabilities
    const total = probabilities?.reduce((sum, entry) => sum + entry.prob, 0) ?? 0
    const valid = probabilities && probabilities.length >= 2 && new Set(probabilities.map(entry => entry.letter)).size === probabilities.length &&
      probabilities.every(entry => Number.isFinite(entry.prob) && entry.prob >= 0 && entry.prob <= 1) && Math.abs(total - 1) < .001
    if (valid && probabilities) {
      if (this.average.size !== probabilities.length || probabilities.some(entry => !this.average.has(entry.letter))) {
        this.average.clear(); this.candidate = null; this.repeats = 0; this.windows = 1
      }
      const scores = probabilities.map(entry => {
        const previous = this.average.get(entry.letter)
        const prob = previous === undefined ? entry.prob : this.alpha * entry.prob + (1 - this.alpha) * previous
        this.average.set(entry.letter, prob)
        return { letter: entry.letter, prob }
      }).sort((a,b) => b.prob - a.prob)
      prediction = { ...raw, letter: scores[0].letter, confidence: scores[0].prob, top3: scores.slice(0,3), probabilities: scores,
        stabilization: { method: 'ema', alpha: this.alpha, windows: this.windows } }
    } else {
      this.average.clear()
      prediction = { ...raw, stabilization: { method: 'consensus', windows: this.windows } }
    }
    this.repeats = this.candidate === prediction.letter ? this.repeats + 1 : 1
    this.candidate = prediction.letter
    const confirmed = this.repeats >= this.confirmations
    return { confirmed, prediction: confirmed ? prediction : null, windows: this.windows }
  }
}
