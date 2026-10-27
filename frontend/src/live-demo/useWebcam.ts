import { useCallback, useEffect, useRef, useState } from 'react'
export type CamStatus = 'idle' | 'requesting' | 'live' | 'denied' | 'error'
export function useWebcam() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const pendingRef = useRef<Promise<CamStatus> | null>(null)
  const generation = useRef(0)
  const mounted = useRef(true)
  const [status, setStatus] = useState<CamStatus>('idle')
  const [error, setError] = useState<string>()
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])
  const release = useCallback(() => {
    generation.current++
    const stream = streamRef.current
    streamRef.current = null
    stream?.getTracks().forEach(track => track.stop())
    if (videoRef.current) videoRef.current.srcObject = null
  }, [])
  const stop = useCallback(() => { release(); if (mounted.current) { setStatus('idle'); setError(undefined) } }, [release])
  const start = useCallback((deviceId?: string): Promise<CamStatus> => {
    if (streamRef.current?.active) return Promise.resolve('live')
    if (pendingRef.current) return pendingRef.current
    const token = ++generation.current
    setStatus('requesting'); setError(undefined)
    const work = async (): Promise<CamStatus> => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access needs a supported browser on localhost or HTTPS.')
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 960 }, height: { ideal: 540 }, frameRate: { ideal: 15, max: 24 }, ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: 'user' }) }, audio: false,
        })
        if (!mounted.current || token !== generation.current) { stream.getTracks().forEach(track => track.stop()); return 'idle' }
        streamRef.current = stream
        const video = videoRef.current
        if (!video) { release(); return 'idle' }
        video.srcObject = stream
        stream.getVideoTracks().forEach(track => track.addEventListener('ended', () => {
          if (streamRef.current !== stream) return
          release()
          if (mounted.current) { setStatus('error'); setError('The camera disconnected. Reconnect it and retry.') }
        }, { once: true }))
        let playbackTimer: ReturnType<typeof setTimeout> | undefined
        try {
          await Promise.race([
            video.play(),
            new Promise<never>((_resolve, reject) => {
              playbackTimer = setTimeout(() => reject(new Error('The camera did not deliver video. Check the device and retry.')), 8000)
            }),
          ])
        } finally { clearTimeout(playbackTimer) }
        if (!mounted.current || token !== generation.current) return 'idle'
        setStatus('live')
        void navigator.mediaDevices.enumerateDevices().then(list => { if (mounted.current && token === generation.current) setDevices(list.filter(item => item.kind === 'videoinput')) }).catch(() => {})
        return 'live'
      } catch (caught) {
        if (!mounted.current || token !== generation.current) return 'idle'
        const err = caught as Error
        release()
        const messages: Record<string, string> = {
          NotAllowedError: 'Camera permission was denied. Allow camera access in your browser site settings, then retry.',
          NotFoundError: 'No camera was found. Connect a camera and retry.',
          NotReadableError: 'The camera is busy or unavailable. Close other camera apps and retry.',
          OverconstrainedError: 'This camera cannot use the selected settings. Try a different camera.',
        }
        const next = err.name === 'NotAllowedError' ? 'denied' : 'error'
        setStatus(next); setError(messages[err.name] ?? err.message ?? 'Could not start the camera.')
        return next
      }
    }
    const pending = work().finally(() => { if (pendingRef.current === pending) pendingRef.current = null })
    pendingRef.current = pending
    return pending
  }, [release])
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; release() } }, [release])
  return { videoRef, streamRef, status, error, devices, start, stop }
}
