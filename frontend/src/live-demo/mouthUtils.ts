/**
 * Mouth geometry utilities for the live demo.
 * These mirror the REAL pipeline contracts (tracking/aligner.py, repr/geometry.py):
 *   - mouth-centered crop with scale = inner mouth width, 2.2× margin
 *   - the 20 canonical MediaPipe FaceMesh mouth landmark indices
 *   - geometry scalars such as mouth width / openness (subset computed live)
 */

export interface Pt {
  x: number
  y: number
}

/** Canonical mouth landmark indices (configs/alphabet_dataset.yaml). */
export const MOUTH_LANDMARK_INDICES = [
  78, 74, 81, 13, 312, 308, 402, 14, 61, 291,
  39, 37, 0, 267, 269, 391, 61, 0, 17, 291,
]

// FaceMesh endpoints used by the real alignment code
const LEFT_CORNER = 61
const RIGHT_CORNER = 291
const INNER_UPPER = 13
const INNER_LOWER = 14

export function innerMouthWidth(lm: Pt[]): number {
  return Math.hypot(lm[RIGHT_CORNER].x - lm[LEFT_CORNER].x, lm[RIGHT_CORNER].y - lm[LEFT_CORNER].y)
}

/** Mouth aperture (inner opening height, normalized by inner mouth width). */
export function mouthOpenness(lm: Pt[]): number {
  const w = innerMouthWidth(lm)
  if (w <= 0) return 0
  return Math.hypot(lm[INNER_LOWER].x - lm[INNER_UPPER].x, lm[INNER_LOWER].y - lm[INNER_UPPER].y) / w
}

export interface CropResult {
  ok: boolean
  /** square source region in video pixels: [x, y, size] */
  region?: [number, number, number]
}

/**
 * Compute the mouth crop region (2.2× margin, like tracking/aligner.py).
 * Returns normalized [0..1] coords scaled by the caller to video pixels.
 */
export function mouthCropRegion(lm: Pt[], margin = 2.2): CropResult {
  const lx = lm[LEFT_CORNER].x
  const rx = lm[RIGHT_CORNER].x
  const cx = (lx + rx) / 2
  const cy = (lm[INNER_UPPER].y + lm[INNER_LOWER].y) / 2
  const w = Math.hypot(rx - lx, lm[RIGHT_CORNER].y - lm[LEFT_CORNER].y)
  if (w <= 0) return { ok: false }
  const size = w * margin
  const half = size / 2
  // clamp to [0,1]
  const x0 = Math.max(0, cx - half)
  const y0 = Math.max(0, cy - half)
  const s = Math.min(size, 1 - x0, 1 - y0)
  if (s <= 0.01) return { ok: false }
  return { ok: true, region: [x0, y0, s] }
}

/** Clamp helper */
export const clamp01 = (v: number) => Math.min(1, Math.max(0, v))
