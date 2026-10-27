import { useCallback, useEffect, useRef, useState } from 'react'
import { useWebcam } from './useWebcam'
import { useLandmarker } from './useLandmarker'
import { MOUTH_LANDMARK_INDICES, mouthOpenness, mouthAlignment } from './mouthUtils'
import { useBackend } from '../hooks/useBackend'
import { LIVE_SEQUENCE, PredictionStabilizer, VisualActivityDetector } from './stabilization'
export interface Top3Entry { letter: string; prob: number }
export interface DemoReadout {
  frames: number; faceOk: boolean; cropOk: boolean; detected: string | null; confidence: number | null
  top3: Top3Entry[] | null; uncertaintyState: string | null; fps: number; latencyMs: number | null
  openness: number | null; motionEnergy: number | null; predictedAt: number | null; elapsedSeconds: number
  speaking: boolean; vadEnergy: number; stabilizing: boolean; smoothed: boolean
}
export interface SessionResult { id: number; letter: string; confidence: number; at: number; latencyMs: number }
const empty = (): DemoReadout => ({ frames: 0, faceOk: false, cropOk: false, detected: null, confidence: null, top3: null, uncertaintyState: null, fps: 0, latencyMs: null, openness: null, motionEnergy: null, predictedAt: null, elapsedSeconds: 0, speaking: false, vadEnergy: 0, stabilizing: false, smoothed: false })
function spark(canvas: HTMLCanvasElement | null, values: number[], maximum: number) {
  const ctx = canvas?.getContext('2d'); if (!canvas || !ctx) return
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  if (values.length < 2) return
  ctx.beginPath()
  values.forEach((v, i) => { const x = i / (values.length - 1) * canvas.width; const y = canvas.height - 4 - v / Math.max(.01, maximum) * (canvas.height - 8); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y) })
  ctx.strokeStyle = '#8fdfcb'; ctx.lineWidth = 1.5; ctx.stroke()
}
export function useDemoPipeline() {
  const cam = useWebcam(), lm = useLandmarker(), backend = useBackend()
  const backendRef = useRef(backend); backendRef.current = backend
  const overlayRef = useRef<HTMLCanvasElement>(null), appearanceRef = useRef<HTMLCanvasElement>(null)
  const geometryRef = useRef<HTMLCanvasElement>(null), motionRef = useRef<HTMLCanvasElement>(null), bufferStripRef = useRef<HTMLCanvasElement>(null)
  const buffer = useRef<string[]>([])
  const request = useRef<AbortController | null>(null), inFlight = useRef(false)
  const session = useRef(0), startedAt = useRef(0)
  const smoother = useRef(new PredictionStabilizer(.6, 2))
  const [readout, setReadout] = useState<DemoReadout>(empty)
  const [history, setHistory] = useState<SessionResult[]>([])
  const [predicting, setPredicting] = useState(false)
  const [pipelineError, setPipelineError] = useState<string | null>(null)
  const running = cam.status === 'live'
  const clear = useCallback(() => {
    session.current++; request.current?.abort(); request.current = null; inFlight.current = false
    buffer.current = []; smoother.current.reset(); setReadout(empty()); setPredicting(false); setPipelineError(null)
    backendRef.current.setLastPrediction(null)
    for (const ref of [overlayRef, appearanceRef, geometryRef, motionRef, bufferStripRef]) {
      const canvas = ref.current; canvas?.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height)
    }
  }, [])
  const start = useCallback(async (deviceId?: string) => {
    clear(); setHistory([]); startedAt.current = performance.now()
    const token = session.current, status = await cam.start(deviceId)
    if (token !== session.current || status !== 'live') return
    void lm.load()
  }, [clear, cam.start, lm.load])
  const stop = useCallback(() => { cam.stop(); clear(); setHistory([]) }, [cam.stop, clear])
  const clearHistory = useCallback(() => setHistory([]), [])
  useEffect(() => {
    if (cam.status === 'idle' || cam.status === 'denied' || cam.status === 'error') { clear(); setHistory([]) }
  }, [cam.status, clear])
  useEffect(() => () => { session.current++; request.current?.abort() }, [])
  useEffect(() => {
    if (backend.status !== 'online' || !backend.inferenceReady) {
      request.current?.abort(); smoother.current.reset(); backend.setLastPrediction(null)
      setReadout(r => ({ ...r, detected: null, confidence: null, top3: null, predictedAt: null, stabilizing: false }))
    }
  }, [backend.status, backend.inferenceReady, backend.setLastPrediction])
  useEffect(() => {
    if (!running || lm.status !== 'ready') return
    const capture = document.createElement('canvas'); capture.width = 96; capture.height = 96
    const captureCtx = capture.getContext('2d', { willReadFrequently: true })
    const geometry: number[] = [], motion: number[] = [], times: number[] = [], frameNumbers: number[] = []
    const activity = new VisualActivityDetector()
    let previous: Uint8ClampedArray | null = null
    let raf = 0, lastDetect = 0, lastBuffer = 0, lastSync = 0, lastVideo = -1, totalFrames = 0
    let utterance = 0, speechStart = 0, lastQueued = 0, lastMovement = -Infinity, lastPublished = 0
    let speaking = false, vadEnergy = 0, idleCleared = true, faceWasVisible = false, disposed = false
    let lastLogged = '', lastLoggedUtterance = -1
    type Window = { frames: string[]; utterance: number }
    let pending: Window | null = null
    const token = session.current
    const clearPrediction = () => {
      smoother.current.reset(); backendRef.current.setLastPrediction(null)
      setReadout(r => ({ ...r, detected: null, confidence: null, top3: null, predictedAt: null, stabilizing: false, smoothed: false }))
    }
    const dispatch = async (window: Window) => {
      if (disposed || token !== session.current || !backendRef.current.inferenceReady || window.utterance !== utterance) return
      if (inFlight.current) { pending = window; return }
      inFlight.current = true; setPredicting(true)
      const controller = new AbortController(); request.current = controller
      const began = performance.now()
      try {
        const raw = await backendRef.current.predict(window.frames, controller.signal)
        if (disposed || controller.signal.aborted || token !== session.current || window.utterance !== utterance) return
        if (!raw) { clearPrediction(); return }
        const result = smoother.current.update(raw)
        setReadout(r => ({ ...r, stabilizing: !result.confirmed }))
        if (!result.prediction) return
        const prediction = result.prediction, at = Date.now(), latencyMs = Math.round(performance.now() - began)
        lastPublished = performance.now(); idleCleared = false
        backendRef.current.setLastPrediction(prediction)
        setReadout(r => ({ ...r, detected: prediction.letter, confidence: prediction.confidence, top3: prediction.top3, uncertaintyState: prediction.uncertainty?.state ?? null, latencyMs, predictedAt: at, stabilizing: false, smoothed: prediction.stabilization?.method === 'ema' }))
        if (prediction.letter !== lastLogged || utterance !== lastLoggedUtterance) {
          lastLogged = prediction.letter; lastLoggedUtterance = utterance
          setHistory(items => [{ id: at, letter: prediction.letter, confidence: prediction.confidence, at, latencyMs }, ...items].slice(0,12))
        }
      } finally {
        if (request.current === controller) {
          request.current = null; inFlight.current = false
          const next = pending; pending = null
          if (!disposed && next && next.utterance === utterance) void dispatch(next)
          else if (!disposed) setPredicting(false)
        }
      }
    }
    const loop = (now: number) => {
      raf = requestAnimationFrame(loop)
      const video = cam.videoRef.current, landmarker = lm.landmarkerRef.current
      if (!video || !landmarker || !captureCtx || video.readyState < 2 || video.currentTime === lastVideo || now - lastDetect < 66) return
      lastVideo = video.currentTime; lastDetect = now
      let faceOk = false, cropOk = false, openness: number | null = null, motionEnergy: number | null = null
      const canvas = overlayRef.current, ctx = canvas?.getContext('2d')
      if (canvas && ctx) {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) { canvas.width = video.videoWidth; canvas.height = video.videoHeight }
        ctx.clearRect(0, 0, canvas.width, canvas.height)
      }
      try {
        const all = landmarker.detectForVideo(video, now).faceLandmarks[0]
        faceOk = !!all?.length
        if (all?.length) {
          if (ctx && canvas) {
            ctx.fillStyle = '#8fdfcb'
            for (const index of new Set(MOUTH_LANDMARK_INDICES)) { const p = all[index]; if (!p) continue; ctx.beginPath(); ctx.arc(p.x * canvas.width, p.y * canvas.height, 2, 0, Math.PI * 2); ctx.fill() }
          }
          const alignment = mouthAlignment(all, video.videoWidth, video.videoHeight)
          if (alignment) {
            cropOk = true
            captureCtx.save()
            captureCtx.clearRect(0,0,96,96)
            captureCtx.translate(48,48)
            captureCtx.scale(96 / alignment.size, 96 / alignment.size)
            captureCtx.rotate(-alignment.angle)
            captureCtx.drawImage(video, -alignment.cx, -alignment.cy)
            captureCtx.restore()
            const appearance = appearanceRef.current
            if (appearance) appearance.getContext('2d')?.drawImage(capture,0,0,appearance.width,appearance.height)
            openness = mouthOpenness(all, video.videoWidth, video.videoHeight)
            const pixels = captureCtx.getImageData(0,0,96,96).data
            if (previous) { let energy = 0; for (let i=0;i<pixels.length;i+=4) energy += Math.abs(pixels[i]-previous[i]); motionEnergy = energy / (96 * 96 * 255) }
            previous = pixels
            geometry.push(openness); if (geometry.length > 90) geometry.shift()
            if (motionEnergy !== null) { motion.push(motionEnergy); if (motion.length > 90) motion.shift() }
            spark(geometryRef.current,geometry,Math.max(.6,...geometry)); spark(motionRef.current,motion,Math.max(.2,...motion))
            if (now - lastBuffer >= 66) {
              lastBuffer = now; totalFrames++
              buffer.current.push(capture.toDataURL('image/jpeg',.75)); frameNumbers.push(totalFrames)
              if (buffer.current.length > LIVE_SEQUENCE.maxFrames) { buffer.current.shift(); frameNumbers.shift() }
              const measured = activity.update(openness,now), wasSpeaking = speaking
              speaking = measured.speaking; vadEnergy = measured.energy
              if (speaking && !wasSpeaking && now - lastMovement > 700) {
                utterance++; speechStart = totalFrames - 7; lastQueued = speechStart
                pending = null; request.current?.abort(); clearPrediction(); idleCleared = false
              }
              lastMovement = measured.lastMovement
              const selected = buffer.current.filter((_frame,index) => frameNumbers[index] >= speechStart)
              const ending = wasSpeaking && !speaking
              if ((speaking || ending) && selected.length >= LIVE_SEQUENCE.minFrames && (totalFrames-lastQueued >= LIVE_SEQUENCE.stride || (ending && totalFrames-lastQueued >= 2))) {
                lastQueued = totalFrames
                void dispatch({ frames: [...selected], utterance })
              }
              const strip = bufferStripRef.current, stripCtx = strip?.getContext('2d')
              if (strip && stripCtx) {
                if (buffer.current.length === LIVE_SEQUENCE.maxFrames) stripCtx.drawImage(strip,24,0,strip.width-24,24,0,0,strip.width-24,24)
                stripCtx.drawImage(capture,(buffer.current.length-1)*24,0,24,24)
              }
            }
          }
        }
      } catch (error) {
        faceOk = false; setPipelineError(error instanceof Error ? error.message : 'Landmark processing failed. Retry the camera.')
      }
      if ((!faceOk || !cropOk) && faceWasVisible) {
        utterance++; pending = null; buffer.current = []; frameNumbers.length = 0
        activity.reset(); speaking = false; vadEnergy = 0; previous = null; request.current?.abort(); clearPrediction()
        for (const ref of [appearanceRef,bufferStripRef]) { const c=ref.current;c?.getContext('2d')?.clearRect(0,0,c.width,c.height) }
      }
      if (!speaking && !inFlight.current && !pending && !idleCleared && now-Math.max(lastMovement,lastPublished)>1800) { clearPrediction(); idleCleared = true }
      faceWasVisible = faceOk && cropOk
      times.push(now); if (times.length>20) times.shift()
      if (now-lastSync>=250) {
        lastSync=now
        setReadout(r=>({...r,frames:buffer.current.length,faceOk,cropOk,openness,motionEnergy,speaking,vadEnergy,elapsedSeconds:Math.floor((now-startedAt.current)/1000),fps:times.length>1?Math.round((times.length-1)*1000/(now-times[0])):0}))
      }
    }
    raf=requestAnimationFrame(loop)
    return()=>{disposed=true;pending=null;cancelAnimationFrame(raf);request.current?.abort();inFlight.current=false;setPredicting(false)}
  },[running,lm.status,lm.landmarkerRef,cam.videoRef])
  return {
    videoRef:cam.videoRef,camStatus:cam.status,camError:cam.error,devices:cam.devices,
    lmStatus:lm.status,lmError:lm.error,backendStatus:backend.status,
    modelLoaded:backend.inferenceReady,inferenceReady:backend.inferenceReady,inferenceReason:backend.inferenceReason,modelKind:backend.modelKind,
    predictionError:backend.predictionError,predicting,running,readout,history,pipelineError,clearHistory,
    overlayRef,appearanceRef,geometryRef,motionRef,bufferStripRef,sequence:LIVE_SEQUENCE,start,stop,
  }
}
