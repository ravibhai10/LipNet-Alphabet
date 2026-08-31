/**
 * useMediaPipeFaceMesh.ts
 *
 * Loads MediaPipe FaceMesh via CDN Tasks API (FaceLandmarker).
 * Compatible with MediaPipe 1.0.1 (mp.tasks.vision only).
 */
import { useRef, useState, useCallback } from "react"

export function useMediaPipeFaceMesh() {
  const landmarkerRef = useRef<unknown>(null)
  const [ready, setReady] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const getOrCreate = useCallback(async () => {
    if (landmarkerRef.current) return landmarkerRef.current
    // Dynamic import from a URL is not supported by TypeScript's static analyzer,
    // so we use Function to defer the import to runtime.
    const dynamicImport = new Function("u", "return import(u)") as (u: string) => Promise<any>
    const { FilesetResolver, FaceLandmarker } = await dynamicImport(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/vision_bundle.js"
    )
    const vision = await FilesetResolver.forVisionTasks("/models/face_landmarker.task")
    const lm = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: "/models/face_landmarker.task", delegate: "CPU" },
      runningMode: "VIDEO",
      numFaces: 1,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    })
    landmarkerRef.current = lm
    setReady(true)
    setLoading(false)
    return lm
  }, [])

  const detect = useCallback(async (video: HTMLVideoElement) => {
    try {
      const lm = await getOrCreate()
      if (!lm) return null
      const result = lm.detectForVideo(video, performance.now())
      if (!result.faceLandmarks?.length) return null
      return { landmarks: result.faceLandmarks as { x: number; y: number; z?: number }[][], timestamp: result.timestampMs ?? performance.now() }
    } catch (e) {
      setError((e as Error).message)
      return null
    }
  }, [getOrCreate])

  return { ready, loading, error, detect }
}
