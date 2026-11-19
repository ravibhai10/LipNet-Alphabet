import test from 'node:test'
import assert from 'node:assert/strict'
import { calculateMAR, calculateEyeAxisAngle, evaluateOpticalReadiness } from '../src/live-demo/visionWorker.ts'

test('Vision Worker: calculateMAR correctly returns mouth aperture ratio', () => {
  // Empty landmarks returns 0
  assert.equal(calculateMAR([]), 0)

  // Synthetic 478 points landmarks
  const landmarks = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }))
  landmarks[13] = { x: 0.5, y: 0.45, z: 0 } // Top lip
  landmarks[14] = { x: 0.5, y: 0.55, z: 0 } // Bottom lip: vert dist = 0.10
  landmarks[61] = { x: 0.40, y: 0.50, z: 0 } // Left corner
  landmarks[291] = { x: 0.60, y: 0.50, z: 0 } // Right corner: horiz dist = 0.20

  const mar = calculateMAR(landmarks)
  assert.ok(Math.abs(mar - 0.5) < 0.001)
})

test('Vision Worker: calculateEyeAxisAngle calculates face roll in radians', () => {
  assert.equal(calculateEyeAxisAngle([]), 0)

  const landmarks = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }))
  // Perfectly level eyes
  landmarks[33] = { x: 0.35, y: 0.35, z: 0 }
  landmarks[263] = { x: 0.65, y: 0.35, z: 0 }
  assert.equal(calculateEyeAxisAngle(landmarks), 0)

  // Tilted 45 degrees
  landmarks[263] = { x: 0.65, y: 0.65, z: 0 }
  const angle = calculateEyeAxisAngle(landmarks)
  assert.ok(angle > 0.7) // ~0.785 rad
})

test('Vision Worker: evaluateOpticalReadiness checks centering, distance and score', () => {
  const landmarks = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0.05 }))
  landmarks[1] = { x: 0.50, y: 0.50, z: 0.05 } // Nose centered
  landmarks[33] = { x: 0.35, y: 0.35, z: 0.05 } // Left eye
  landmarks[263] = { x: 0.65, y: 0.35, z: 0.05 } // Right eye: span = 0.30

  const result = evaluateOpticalReadiness(landmarks, 640, 480)
  assert.equal(result.centeringOk, true)
  assert.equal(result.distanceOk, true)
  assert.equal(result.lightingOk, true)
  assert.equal(result.score, 100)
})
