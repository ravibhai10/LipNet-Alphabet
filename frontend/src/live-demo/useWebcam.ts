import { useCallback, useEffect, useRef, useState } from 'react'

export type CamStatus = 'idle' | 'requesting' | 'live' | 'denied' | 'error'

interface WebcamHandle {
  videoRef: React.RefObject<HTMLVideoElement>
  status: CamStatus
  error?: string
  start: () => Promise<CamStatus>
  stop: () => void
}

/**
 * Webcam lifecycle. The video element is owned by the caller; the stream is
 * only attached after an explicit user gesture (no camera access on load).
 */
export function useWebcam(): WebcamHandle {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [status, setStatus] = useState<CamStatus>('idle')
  const [error, setError] = useState<string | undefined>(undefined)

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setStatus('idle')
  }, [])

  const start = useCallback(async (): Promise<CamStatus> => {
    if (streamRef.current) return 'live'
    setStatus('requesting')
    setError(undefined)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 960 }, height: { ideal: 540 }, facingMode: 'user' },
        audio: false,
      })
      streamRef.current = stream
      const v = videoRef.current
      if (v) {
        v.srcObject = stream
        await v.play().catch(() => {})
      }
      setStatus('live')
      return 'live'
    } catch (e) {
      const err = e as DOMException
      if (err.name === 'NotAllowedError') {
        setStatus('denied')
        setError('Camera permission denied.')
        return 'denied'
      }
      setStatus('error')
      setError(err.message || 'Could not access the camera.')
      return 'error'
    }
  }, [])

  useEffect(() => stop, [stop])

  return { videoRef, status, error, start, stop }
}
