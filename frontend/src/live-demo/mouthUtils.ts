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
  if (!lm[RIGHT_CORNER] || !lm[LEFT_CORNER]) return 0
  return Math.hypot(lm[RIGHT_CORNER].x - lm[LEFT_CORNER].x, lm[RIGHT_CORNER].y - lm[LEFT_CORNER].y)
}

/** Mouth aperture (inner opening height, normalized by inner mouth width). */
export function mouthOpenness(lm: Pt[], width = 1, height = 1): number {
  if (!lm[INNER_UPPER] || !lm[INNER_LOWER] || !lm[LEFT_CORNER] || !lm[RIGHT_CORNER]) return 0
  const w = Math.hypot((lm[RIGHT_CORNER].x - lm[LEFT_CORNER].x) * width, (lm[RIGHT_CORNER].y - lm[LEFT_CORNER].y) * height)
  if (w <= 0) return 0
  return Math.hypot((lm[INNER_LOWER].x - lm[INNER_UPPER].x) * width, (lm[INNER_LOWER].y - lm[INNER_UPPER].y) * height) / w
}

export function mouthAlignment(lm: Pt[], width: number, height: number, margin = 2.2) {
  if (![width, height, margin].every(value => Number.isFinite(value) && value > 0)) return null
  if (!lm[61] || !lm[291] || !lm[13] || !lm[14]) return null
  const pixels = (index: number) => ({ x: lm[index].x * width, y: lm[index].y * height })
  const left = pixels(61), right = pixels(291), upper = pixels(13), lower = pixels(14)
  const mouthWidth = Math.hypot(right.x - left.x, right.y - left.y)
  if (!Number.isFinite(mouthWidth) || mouthWidth < 2) return null
  const eyeLeft = lm[33] ? pixels(33) : left, eyeRight = lm[263] ? pixels(263) : right
  if (![left, right, upper, lower, eyeLeft, eyeRight].every(point => Number.isFinite(point.x) && Number.isFinite(point.y))) return null
  const eyeWidth = Math.hypot(eyeRight.y - eyeLeft.y, eyeRight.x - eyeLeft.x)
  const angle = eyeWidth >= 2 ? Math.atan2(eyeRight.y - eyeLeft.y, eyeRight.x - eyeLeft.x) : Math.atan2(right.y - left.y, right.x - left.x)
  const cx = (upper.x + lower.x) / 2, cy = (upper.y + lower.y) / 2, size = mouthWidth * margin
  const extent = size / 2 * (Math.abs(Math.cos(angle)) + Math.abs(Math.sin(angle)))
  // Reject off-image crops instead of introducing transparent/black padding.
  if (cx < extent || cy < extent || cx + extent > width || cy + extent > height) return null
  return { cx, cy, size, angle }
}

export interface CropResult {
  ok: boolean
  /** square source region in video pixels: [x, y, size] */
  region?: [number, number, number]
}

/**
 * Compute the mouth crop region (2.2× margin, like tracking/aligner.py).
 * Accepts the full FaceMesh array. With video dimensions, returns a square
 * crop in pixels and clamps each axis to that video's bounds.
 */
export function mouthCropRegion(lm: Pt[], margin = 2.2, width = 1, height = 1): CropResult {
  if (!lm[LEFT_CORNER] || !lm[RIGHT_CORNER] || !lm[INNER_UPPER] || !lm[INNER_LOWER]) return { ok: false }
  const lx = lm[LEFT_CORNER].x * width
  const rx = lm[RIGHT_CORNER].x * width
  const cx = (lx + rx) / 2
  const cy = (lm[INNER_UPPER].y + lm[INNER_LOWER].y) / 2 * height
  const w = Math.hypot(rx - lx, (lm[RIGHT_CORNER].y - lm[LEFT_CORNER].y) * height)
  if (w <= 0) return { ok: false }
  const size = Math.min(w * margin, width, height)
  const half = size / 2
  // clamp to [0,1]
  const x0 = Math.min(width - size, Math.max(0, cx - half))
  const y0 = Math.min(height - size, Math.max(0, cy - half))
  const s = size
  if (s <= 0.01) return { ok: false }
  return { ok: true, region: [x0, y0, s] }
}

/** Clamp helper */
export const clamp01 = (v: number) => Math.min(1, Math.max(0, v))
