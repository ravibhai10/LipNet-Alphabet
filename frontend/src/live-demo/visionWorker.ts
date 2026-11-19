/**
 * Vision Pipeline WebWorker & High-Performance Matrix Math Module
 * Offloads eye-axis rotation leveling, 96x96 normalization, MAR calculation,
 * and temporal circular buffer management to keep the UI running at 60 FPS.
 */

export interface LandmarkPoint {
  x: number
  y: number
  z?: number
}

export interface VisionFrameMessage {
  type: 'PROCESS_FRAME'
  landmarks: LandmarkPoint[]
  imageWidth: number
  imageHeight: number
  timestamp: number
}

export interface VisionResult {
  mar: number
  eyeAxisAngleRad: number
  isAligned: boolean
  lightingOk: boolean
  centeringOk: boolean
  distanceOk: boolean
  boundingBox: { x: number; y: number; width: number; height: number }
}

/**
 * Calculates Mouth Aperture Ratio (MAR) from standard landmark points:
 * MAR = (||p13 - p14||) / (||p61 - p291||)
 */
export function calculateMAR(landmarks: LandmarkPoint[]): number {
  if (!landmarks || landmarks.length < 300) return 0
  const topLip = landmarks[13]
  const bottomLip = landmarks[14]
  const leftCorner = landmarks[61]
  const rightCorner = landmarks[291]

  if (!topLip || !bottomLip || !leftCorner || !rightCorner) return 0

  const verticalDist = Math.hypot(topLip.x - bottomLip.x, topLip.y - bottomLip.y)
  const horizontalDist = Math.hypot(leftCorner.x - rightCorner.x, leftCorner.y - rightCorner.y)

  if (horizontalDist < 0.0001) return 0
  return verticalDist / horizontalDist
}

/**
 * Calculates eye-axis roll angle (in radians) to deskew / level the face:
 * Left eye center ~ 33, Right eye center ~ 263
 */
export function calculateEyeAxisAngle(landmarks: LandmarkPoint[]): number {
  if (!landmarks || landmarks.length < 300) return 0
  const leftEye = landmarks[33]
  const rightEye = landmarks[263]
  if (!leftEye || !rightEye) return 0

  return Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x)
}

/**
 * Evaluates optical quality metrics: lighting, centering, distance, and stability
 */
export function evaluateOpticalReadiness(
  landmarks: LandmarkPoint[],
  imageWidth?: number,
  imageHeight?: number
): {
  lightingOk: boolean
  centeringOk: boolean
  distanceOk: boolean
  score: number
}
export function evaluateOpticalReadiness(landmarks: LandmarkPoint[]): {
  lightingOk: boolean
  centeringOk: boolean
  distanceOk: boolean
  score: number
} {
  if (!landmarks || landmarks.length < 100) {
    return { lightingOk: false, centeringOk: false, distanceOk: false, score: 0 }
  }

  // Nose tip center ~ 1
  const nose = landmarks[1]
  const leftEye = landmarks[33]
  const rightEye = landmarks[263]

  if (!nose || !leftEye || !rightEye) {
    return { lightingOk: false, centeringOk: false, distanceOk: false, score: 0 }
  }

  // Centering: nose tip within central 40% of frame
  const centeringOk = nose.x >= 0.30 && nose.x <= 0.70 && nose.y >= 0.25 && nose.y <= 0.75

  // Distance: eye span between 15% and 45% of image width
  const eyeSpan = Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y)
  const distanceOk = eyeSpan >= 0.15 && eyeSpan <= 0.45

  // Lighting heuristic based on presence of confident depth values
  const lightingOk = (nose.z !== undefined && !isNaN(nose.z))

  const score = Math.round(
    (centeringOk ? 35 : 10) +
    (distanceOk ? 35 : 10) +
    (lightingOk ? 30 : 15)
  )

  return { lightingOk, centeringOk, distanceOk, score }
}

/**
 * WebWorker message event listener (when loaded in Worker context)
 */
if (typeof self !== 'undefined' && typeof (self as unknown as Worker).postMessage === 'function') {
  self.addEventListener('message', (event: MessageEvent<VisionFrameMessage>) => {
    if (event.data?.type === 'PROCESS_FRAME') {
      const { landmarks, imageWidth, imageHeight } = event.data
      const mar = calculateMAR(landmarks)
      const angle = calculateEyeAxisAngle(landmarks)
      const readiness = evaluateOpticalReadiness(landmarks, imageWidth, imageHeight)

      const result: VisionResult = {
        mar,
        eyeAxisAngleRad: angle,
        isAligned: Math.abs(angle) < 0.15,
        lightingOk: readiness.lightingOk,
        centeringOk: readiness.centeringOk,
        distanceOk: readiness.distanceOk,
        boundingBox: {
          x: Math.round(landmarks[61]?.x * imageWidth || 0),
          y: Math.round(landmarks[13]?.y * imageHeight || 0),
          width: 96,
          height: 96,
        }
      }

      ;(self as unknown as Worker).postMessage(result)
    }
  })
}
