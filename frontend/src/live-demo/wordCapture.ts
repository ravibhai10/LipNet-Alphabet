/** Anatomical FaceMesh correspondences, not the publisher's landmark detector. */
export const LRW_CONVENTION = 'lrw-meanface-mediapipe68-v1'
export const MP68 = [234,93,132,58,172,136,150,149,152,378,379,365,397,288,361,323,454,
  70,63,105,66,107,336,296,334,293,300,168,6,197,1,98,97,2,326,327,
  33,160,158,133,153,144,362,385,387,263,373,380,
  61,40,37,0,267,270,291,321,314,17,84,91,78,81,13,311,308,402,14,178] as const
export interface WordCapture {
  convention: string
  input_kind: 'webcam'
  landmarks: number[][][]
  timestamps_ms: number[]
  face_counts: number[]
  final: boolean
}
export function compactLandmarks(points: {x: number; y: number}[]): number[][] | null {
  const result = MP68.map(index => points[index]).map(p => p ? [p.x,p.y] : [])
  return result.every(p => p.length === 2 && p.every(n => Number.isFinite(n) && n >= 0 && n <= 1)) ? result : null
}
/** One publication per utterance, even if a later candidate changes. */
export class UtterancePublication {
  private published = new Set<number>()
  reset() { this.published.clear() }
  has(utterance: number) { return this.published.has(utterance) }
  claim(utterance: number): boolean {
    if (this.has(utterance)) return false
    this.published.add(utterance)
    if (this.published.size > 128) this.published.delete(this.published.values().next().value!)
    return true
  }
}

/** Discard a discontinuous capture instead of sending it as a continuous word. */
export function captureGapExceeded(previous: number | undefined, current: number): boolean {
  return previous !== undefined && (!Number.isFinite(current) || current <= previous || current - previous > 160)
}

export function utteranceStartIndex(timestamps: number[], now: number, prerollMs = 240): number {
  const index = timestamps.findIndex(time => time >= now - prerollMs)
  return index < 0 ? Math.max(0,timestamps.length-1) : index
}
