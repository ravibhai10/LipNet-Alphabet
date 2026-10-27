import { useCallback, useEffect, useRef, useState } from 'react'
import type { FaceLandmarker } from '@mediapipe/tasks-vision'
export type LandmarkerStatus = 'unloaded' | 'loading' | 'ready' | 'failed'
export function useLandmarker() {
  const landmarkerRef = useRef<FaceLandmarker | null>(null)
  const pendingRef = useRef<Promise<FaceLandmarker | null> | null>(null)
  const mounted = useRef(true)
  const [status, setStatus] = useState<LandmarkerStatus>('unloaded')
  const [error, setError] = useState<string | null>(null)
  const load = useCallback((): Promise<FaceLandmarker | null> => {
    if (landmarkerRef.current) return Promise.resolve(landmarkerRef.current)
    if (pendingRef.current) return pendingRef.current
    setStatus('loading'); setError(null)
    const work = async () => {
      try {
        const vision = await import('@mediapipe/tasks-vision')
        const fileset = await vision.FilesetResolver.forVisionTasks('/mediapipe/wasm')
        const model = await vision.FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: '/models/face_landmarker.task', delegate: 'CPU' },
          runningMode: 'VIDEO', numFaces: 1,
        })
        if (!mounted.current) { model.close(); return null }
        landmarkerRef.current = model; setStatus('ready'); return model
      } catch (caught) {
        if (mounted.current) { setStatus('failed'); setError(caught instanceof Error ? caught.message : 'Landmark runtime could not load.') }
        return null
      }
    }
    const pending = work().finally(() => { if (pendingRef.current === pending) pendingRef.current = null })
    pendingRef.current = pending
    return pending
  }, [])
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; landmarkerRef.current?.close(); landmarkerRef.current = null } }, [])
  return { landmarkerRef, status, error, load }
}
