import { useCallback, useEffect, useRef, useState } from 'react'
import { useWebcam, type CamStatus } from './useWebcam'
import { useLandmarker } from './useLandmarker'
import { MOUTH_LANDMARK_INDICES, mouthOpenness, mouthCropRegion, type Pt } from './mouthUtils'
import { useBackend } from '../hooks/useBackend'
import { SEQUENCE } from '../data/architecture'

export interface Top3Entry {
  letter: string
  prob: number
}

export interface DemoReadout {
  frames: number
  faceOk: boolean
  detected: string | null
  confidence: number | null
  top3: Top3Entry[] | null
  uncertaintyState: string | null
  fps: number
}

const DETECT_INTERVAL_MS = 66 // ≈ 15 Hz landmark inference
const BUFFER_INTERVAL_MS = 130 // ≈ 7.7 Hz frame capture
const READOUT_INTERVAL_MS = 250 // UI state sync at 4 Hz — never per frame
const PREDICT_INTERVAL_MS = 2500

/** Minimal polyline sparkline on a canvas. */
function drawSparkline(
  canvas: HTMLCanvasElement | null,
  values: number[],
  min: number,
  max: number,
) {
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  const { width: w, height: h } = canvas
  ctx.clearRect(0, 0, w, h)
  if (values.length < 2) return
  ctx.beginPath()
  const n = values.length
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * w
    const y = h - ((values[i] - min) / (max - min || 1)) * (h - 6) - 3
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  }
  ctx.strokeStyle = 'rgba(143, 223, 203, 0.85)'
  ctx.lineWidth = 1.5
  ctx.stroke()
}

export function useDemoPipeline() {
  const cam = useWebcam()
  const lm = useLandmarker()
  const { status: backendStatus, predict, setLastPrediction } = useBackend()

  const overlayRef = useRef<HTMLCanvasElement>(null)
  const appearanceRef = useRef<HTMLCanvasElement>(null)
  const geometryRef = useRef<HTMLCanvasElement>(null)
  const motionRef = useRef<HTMLCanvasElement>(null)
  const bufferStripRef = useRef<HTMLCanvasElement>(null)

  const lastDetectRef = useRef(0)
  const lastVideoTimeRef = useRef(-1)
  const lastBufferTsRef = useRef(0)
  const lastReadoutRef = useRef(0)
  const lastPredictRef = useRef(0)
  const predictingRef = useRef(false)
  const bufferDataRef = useRef<string[]>([])
  const opennessHist = useRef<number[]>([])
  const detectTimes = useRef<number[]>([])
  const backendStatusRef = useRef(backendStatus)
  backendStatusRef.current = backendStatus

  const [running, setRunning] = useState(false)
  const [predicting, setPredicting] = useState(false)
  predictingRef.current = predicting

  const [readout, setReadout] = useState<DemoReadout>({
    frames: 0,
    faceOk: false,
    detected: null,
    confidence: null,
    top3: null,
    uncertaintyState: null,
    fps: 0,
  })

  const reset = useCallback(() => {
    bufferDataRef.current = []
    opennessHist.current = []
    detectTimes.current = []
    setReadout({
      frames: 0, faceOk: false, detected: null,
      confidence: null, top3: null, uncertaintyState: null, fps: 0,
    })
    setLastPrediction(null)
  }, [setLastPrediction])

  const start = useCallback(async () => {
    reset()
    const result = await cam.start()
    if (result !== 'live') return
    setRunning(true)
    void lm.load()
  }, [cam, lm, reset])

  const stop = useCallback(() => {
    setRunning(false)
    cam.stop()
    reset()
  }, [cam, reset])


  // The single rAF loop — created once per "running" lifecycle; reads only
  // refs and stable callbacks so it never captures stale props.
  useEffect(() => {
    if (!running) return
    let raf = 0

    const loop = () => {
      raf = requestAnimationFrame(loop)
      const video = cam.videoRef.current
      const now = performance.now()
      if (!video || video.readyState < 2) return

      if (video.currentTime === lastVideoTimeRef.current) return
      lastVideoTimeRef.current = video.currentTime
      if (now - lastDetectRef.current < DETECT_INTERVAL_MS) return
      lastDetectRef.current = now

      const landmarker = lm.landmarkerRef.current
      if (!landmarker) return

      let faceOk = false
      try {
        const res = landmarker.detectForVideo(video, now)
        const all = res.faceLandmarks?.[0]
        faceOk = !!all && all.length > 0

        const overlay = overlayRef.current
        const octx = overlay?.getContext('2d')
        if (overlay && octx) {
          const w = overlay.width
          const h = overlay.height
          octx.clearRect(0, 0, w, h)
          octx.save()
          octx.translate(w, 0)
          octx.scale(-1, 1) // mirror to match the CSS-mirrored video

          if (faceOk && all) {
            octx.fillStyle = 'rgba(237, 235, 228, 0.28)'
            for (let i = 0; i < all.length; i += 2) {
              octx.fillRect(all[i].x * w - 0.75, all[i].y * h - 0.75, 1.5, 1.5)
            }
            const mouth: Pt[] = MOUTH_LANDMARK_INDICES.map((idx) => ({ x: all[idx].x, y: all[idx].y }))
            octx.strokeStyle = 'rgba(143, 223, 203, 0.9)'
            octx.lineWidth = 1.2
            octx.beginPath()
            mouth.slice(0, 10).forEach((p, i) =>
              i === 0 ? octx.moveTo(p.x * w, p.y * h) : octx.lineTo(p.x * w, p.y * h),
            )
            octx.closePath()
            octx.stroke()
            octx.beginPath()
            mouth.slice(10).forEach((p, i) =>
              i === 0 ? octx.moveTo(p.x * w, p.y * h) : octx.lineTo(p.x * w, p.y * h),
            )
            octx.stroke()
            octx.fillStyle = 'rgba(143, 223, 203, 1)'
            mouth.forEach((p) => {
              octx.beginPath()
              octx.arc(p.x * w, p.y * h, 1.8, 0, Math.PI * 2)
              octx.fill()
            })

            const crop = mouthCropRegion(mouth)
            const ap = appearanceRef.current
            const actx = ap?.getContext('2d')
            if (crop.ok && crop.region && ap && actx) {
              const [nx, ny, ns] = crop.region
              const sx = nx * video.videoWidth
              const sy = ny * video.videoHeight
              const s = ns * video.videoWidth
              actx.save()
              actx.translate(64, 0)
              actx.scale(-1, 1)
              actx.drawImage(video, sx, sy, s, s, 0, 0, 64, 64)
              actx.restore()

              if (now - lastBufferTsRef.current >= BUFFER_INTERVAL_MS) {
                lastBufferTsRef.current = now
                bufferDataRef.current.push(ap.toDataURL('image/jpeg', 0.6))
                if (bufferDataRef.current.length > SEQUENCE.maxFrames) bufferDataRef.current.shift()

                const strip = bufferStripRef.current
                const sctx = strip?.getContext('2d')
                if (strip && sctx) {
                  const cell = 24
                  const slot = bufferDataRef.current.length - 1
                  sctx.clearRect((slot % SEQUENCE.maxFrames) * cell, 0, cell, cell)
                  sctx.drawImage(ap, 0, 0, 64, 64, (slot % SEQUENCE.maxFrames) * cell, 0, cell, cell)
                }
              }

              const openness = mouthOpenness(mouth)
              const hist = opennessHist.current
              hist.push(openness)
              if (hist.length > 120) hist.shift()
              drawSparkline(geometryRef.current, hist, 0, Math.max(0.6, ...hist))
              const deriv = hist.slice(1).map((v, i) => v - hist[i])
              drawSparkline(motionRef.current, deriv, -0.15, 0.15)

              detectTimes.current.push(now)
              if (detectTimes.current.length > 30) detectTimes.current.shift()
            }
          }
          octx.restore()
        }
      } catch {
        faceOk = false
      }

      if (now - lastReadoutRef.current >= READOUT_INTERVAL_MS) {
        lastReadoutRef.current = now
        const times = detectTimes.current
        const fps =
          times.length > 1
            ? Math.round((1000 * (times.length - 1)) / (times[times.length - 1] - times[0]))
            : 0
        setReadout((r) => ({ ...r, frames: bufferDataRef.current.length, faceOk, fps }))
      }

      if (
        backendStatusRef.current === 'online' &&
        !predictingRef.current &&
        now - lastPredictRef.current >= PREDICT_INTERVAL_MS &&
        bufferDataRef.current.length >= SEQUENCE.minFrames
      ) {
        lastPredictRef.current = now
        setPredicting(true)
        void predict(bufferDataRef.current.slice(-SEQUENCE.maxFrames)).then((p) => {
          setPredicting(false)
          if (p) {
            setLastPrediction(p)
            setReadout((r) => ({
              ...r,
              detected: p.letter,
              confidence: p.confidence,
              top3: p.top3,
              uncertaintyState: p.uncertainty?.state ?? null,
            }))
          }
        })
      }
    }

    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running])

  return {
    videoRef: cam.videoRef,
    camStatus: cam.status as CamStatus,
    camError: cam.error,
    lmStatus: lm.status,
    lmError: lm.error,
    backendStatus,
    predicting,
    running,
    readout,
    overlayRef,
    appearanceRef,
    geometryRef,
    motionRef,
    bufferStripRef,
    start,
    stop,
  }
}

