import { useCallback, useRef, useState } from 'react'
import type { FaceLandmarker } from '@mediapipe/tasks-vision'

/**
 * Lazy MediaPipe FaceLandmarker loader.
 *
 * The heavy WASM + model assets are loaded ONLY when the user starts the
 * demo (dynamic import → separate chunk, nothing loaded on page init).
 * The model file is the project's own `models/face_landmarker.task`
 * (the same weights the Python pipeline uses), served from /models/.
 */

const WASM_BASE =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
const MODEL_PATH = '/models/face_landmarker.task'

export type LandmarkerStatus = 'unloaded' | 'loading' | 'ready' | 'failed'

export function useLandmarker() {
  const landmarkerRef = useRef<FaceLandmarker | null>(null)
  const [status, setStatus] = useState<LandmarkerStatus>('unloaded')
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async (): Promise<FaceLandmarker | null> => {
    if (landmarkerRef.current) return landmarkerRef.current
    if (status === 'loading') return null
    setStatus('loading')
    setError(null)
    try {
      const vision = await import('@mediapipe/tasks-vision')
      const fileset = await vision.FilesetResolver.forVisionTasks(WASM_BASE)
      const opts = (delegate: 'GPU' | 'CPU') => ({
        baseOptions: { modelAssetPath: MODEL_PATH, delegate },
        runningMode: 'VIDEO' as const,
        numFaces: 1,
      })
      let lm: FaceLandmarker
      try {
        lm = await vision.FaceLandmarker.createFromOptions(fileset, opts('GPU'))
      } catch {
        lm = await vision.FaceLandmarker.createFromOptions(fileset, opts('CPU'))
      }
      landmarkerRef.current = lm
      setStatus('ready')
      return lm
    } catch (e) {
      setError((e as Error).message || 'Failed to load the landmark model.')
      setStatus('failed')
      return null
    }
  }, [status])

  return { landmarkerRef, status, error, load }
}
