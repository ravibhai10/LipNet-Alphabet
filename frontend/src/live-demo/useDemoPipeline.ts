import { useCallback, useEffect, useRef, useState } from 'react'
import { useWebcam } from './useWebcam'
import { useLandmarker } from './useLandmarker'
import { MOUTH_LANDMARK_INDICES, mouthOpenness, mouthAlignment } from './mouthUtils'
import { useBackend } from '../hooks/useBackend'
import { LIVE_SEQUENCE, PredictionStabilizer, VisualActivityDetector } from './stabilization'
import { WordConsensus } from './wordConsensus'
import { useWordAssistant } from './useWordAssistant'
import { WORD_LIMITS } from './wordAssistant'
import { compactLandmarks, captureGapExceeded, utteranceStartIndex, LRW_CONVENTION, UtterancePublication, type WordCapture } from './wordCapture'
export interface Top3Entry { letter: string; prob: number }
export interface DemoReadout {
  frames: number; faceOk: boolean; cropOk: boolean; detected: string | null; confidence: number | null
  top3: Top3Entry[] | null; uncertaintyState: string | null; fps: number; latencyMs: number | null
  openness: number | null; motionEnergy: number | null; predictedAt: number | null; elapsedSeconds: number
  speaking: boolean; vadEnergy: number; stabilizing: boolean; smoothed: boolean
  quality: string | null; serverLatencyMs: number | null; ramMb: number | null; cpuPercent: number | null; throughput: number | null
}
export interface SessionResult { id: number; letter: string; confidence: number | null; at: number; latencyMs: number }
const empty = (): DemoReadout => ({ frames: 0, faceOk: false, cropOk: false, detected: null, confidence: null, top3: null, uncertaintyState: null, fps: 0, latencyMs: null, openness: null, motionEnergy: null, predictedAt: null, elapsedSeconds: 0, speaking: false, vadEnergy: 0, stabilizing: false, smoothed: false, quality: null, serverLatencyMs: null, ramMb: null, cpuPercent: null, throughput: null })
function spark(canvas: HTMLCanvasElement | null, values: number[], maximum: number) {
  const ctx = canvas?.getContext('2d'); if (!canvas || !ctx) return
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  if (values.length < 2) return
  ctx.beginPath()
  values.forEach((v, i) => { const x = i / (values.length - 1) * canvas.width; const y = canvas.height - 4 - v / Math.max(.01, maximum) * (canvas.height - 8); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y) })
  ctx.strokeStyle = '#8fdfcb'; ctx.lineWidth = 1.5; ctx.stroke()
}
export function useDemoPipeline(options: { manualWords?: boolean } = {}) {
  const cam = useWebcam(), lm = useLandmarker(), backend = useBackend()
  const backendRef = useRef(backend); backendRef.current = backend
  const manualWords = options.manualWords === true && backend.recognitionMode !== 'alphabet'
  const [capturedClip, setCapturedClip] = useState<string[]>([])
  const assistant = useWordAssistant((frames, signal, capture) => {
    setCapturedClip([...frames])
    return backendRef.current.predictWord(frames, signal, capture, true)
  })
  const assistantController = assistant.controller
  const overlayRef = useRef<HTMLCanvasElement>(null), appearanceRef = useRef<HTMLCanvasElement>(null)
  const geometryRef = useRef<HTMLCanvasElement>(null), motionRef = useRef<HTMLCanvasElement>(null), bufferStripRef = useRef<HTMLCanvasElement>(null)
  const buffer = useRef<string[]>([])
  const request = useRef<AbortController | null>(null), inFlight = useRef(false)
  const session = useRef(0), startedAt = useRef(0)
  const smoother = useRef(new PredictionStabilizer(.6, 2))
  const wordConsensus = useRef(new WordConsensus())
  const publication = useRef(new UtterancePublication())
  const [showOverlay, setShowOverlay] = useState(true)
  const overlayEnabled = useRef(true); overlayEnabled.current = showOverlay
  const [readout, setReadout] = useState<DemoReadout>(empty)
  const [history, setHistory] = useState<SessionResult[]>([])
  const [predicting, setPredicting] = useState(false)
  const [pipelineError, setPipelineError] = useState<string | null>(null)
  const running = cam.status === 'live'
  const clear = useCallback(() => {
    session.current++; request.current?.abort(); request.current = null; inFlight.current = false
    buffer.current = []; smoother.current.reset(); wordConsensus.current.reset(); publication.current.reset()
    setReadout(r => manualWords ? { ...empty(), detected: r.detected, confidence: r.confidence, top3: r.top3, predictedAt: r.predictedAt, latencyMs: r.latencyMs, serverLatencyMs: r.serverLatencyMs } : empty())
    setPredicting(false); setPipelineError(null)
    backendRef.current.setLastPrediction(null)
    backendRef.current.setLastWordPrediction(null)
    for (const ref of [overlayRef, appearanceRef, geometryRef, motionRef, bufferStripRef]) {
      const canvas = ref.current; canvas?.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height)
    }
  }, [manualWords])
  const start = useCallback(async (deviceId?: string) => {
    if (manualWords && (cam.streamRef.current?.active || !assistantController.startCamera())) return
    clear(); if (!manualWords) setHistory([]); startedAt.current = performance.now()
    const token = session.current, status = await cam.start(deviceId)
    if (token !== session.current || status !== 'live') return
    const model = await lm.load()
    if (token !== session.current) return
    if (manualWords) {
      if (model) assistantController.cameraReady()
      else assistantController.fail('Landmark tracking could not start. Stop and retry the camera.')
    }
  }, [clear, cam.start, cam.streamRef, lm.load, manualWords, assistantController])
  const stop = useCallback(() => { assistantController.stop(); cam.stop(); clear(); if (!manualWords) setHistory([]) }, [cam.stop, clear, manualWords, assistantController])
  const previousMode = useRef(backend.recognitionMode)
  useEffect(() => {
    if (previousMode.current !== backend.recognitionMode) {
      previousMode.current = backend.recognitionMode
      stop()
    }
  }, [backend.recognitionMode, stop])
  const clearHistory = useCallback(() => { if (manualWords) assistantController.send({ type: 'CLEAR_HISTORY' }); else setHistory([]) }, [manualWords, assistantController])
  const reset = useCallback(() => { stop(); assistantController.reset(); setHistory([]) }, [stop, assistantController])
  useEffect(() => {
    if (cam.status === 'idle' || cam.status === 'denied' || cam.status === 'error') {
      clear()
      if (manualWords && (cam.status === 'denied' || cam.status === 'error')) assistantController.cancel(cam.error || 'The camera is unavailable. Reconnect it and retry.')
      if (!manualWords) setHistory([])
    }
  }, [cam.status, cam.error, clear, manualWords, assistantController])
  useEffect(() => () => { session.current++; request.current?.abort() }, [])
  useEffect(() => {
    if (backend.status !== 'online' || !backend.inferenceReady) {
      if (manualWords) {
        if (['CAPTURING', 'PROCESSING'].includes(assistantController.getSnapshot().phase)) assistantController.cancel('The word backend is unavailable. Check the connection and retry your word.')
        return
      }
      request.current?.abort(); smoother.current.reset(); wordConsensus.current.reset(); backend.setLastPrediction(null); backend.setLastWordPrediction(null)
      setHistory([])
      setReadout(r => ({ ...r, detected: null, confidence: null, top3: null, predictedAt: null, stabilizing: false }))
    }
  }, [backend.status, backend.inferenceReady, backend.setLastPrediction, manualWords, assistantController])
  useEffect(() => {
    if (!manualWords) return
    const result = assistant.state.result
    if (result) setReadout(r => ({ ...r, detected: result.word, confidence: result.confidence,
      top3: result.candidates.map(p => ({ letter: p.word, prob: p.prob })), predictedAt: result.at,
      latencyMs: result.roundTripMs, serverLatencyMs: result.inferenceMs,
      ramMb: result.performance?.memory_after?.rss_bytes != null ? result.performance.memory_after.rss_bytes / 1048576 : null,
      cpuPercent: result.performance?.process_cpu_percent_one_core ?? null,
      throughput: result.performance?.requests_per_second_sequential ?? null,
      uncertaintyState: result.status === 'uncertain' ? 'UNCERTAIN' : 'PREDICTION_AVAILABLE', stabilizing: false, smoothed: false }))
  }, [manualWords, assistant.state.result])
  const canCaptureWord = running && lm.status === 'ready' && backend.inferenceReady && readout.faceOk && readout.cropOk
  const beginWord = useCallback((action: 'start' | 'next' | 'retry' = 'start') => {
    if (!manualWords || !canCaptureWord || backend.recognitionMode === 'alphabet') return
    const mode = backend.recognitionMode
    const changed = action === 'next' ? assistantController.next(mode) : action === 'retry' ? assistantController.retry(mode) : assistantController.begin(mode)
    if (!changed) return
    buffer.current = []; setPipelineError(null)
    backendRef.current.setLastWordPrediction(null)
    setReadout(r => ({ ...r, frames: 0, detected: null, confidence: null, top3: null, predictedAt: null, latencyMs: null, serverLatencyMs: null }))
    const c = bufferStripRef.current; c?.getContext('2d')?.clearRect(0, 0, c.width, c.height)
  }, [manualWords, canCaptureWord, backend.recognitionMode, assistantController])
  useEffect(() => {
    if (!running || lm.status !== 'ready') return
    const capture = document.createElement('canvas'); capture.width = 96; capture.height = 96
    const lrwMode = backendRef.current.recognitionMode === 'lrw500'
    const captureMode = backendRef.current.recognitionMode === 'alphabet' ? 'word' : backendRef.current.recognitionMode
    const sequence = manualWords ? { ...WORD_LIMITS[captureMode], stride: 1 } : lrwMode ? {maxFrames:76,minFrames:12,stride:8} : LIVE_SEQUENCE
    const cadence = lrwMode ? 40 : 66
    const fullFace = document.createElement('canvas')
    const fullCtx = fullFace.getContext('2d')
    const captureTimes: number[] = [], capturedLandmarks: number[][][] = []
    const captureCtx = capture.getContext('2d', { willReadFrequently: true })
    const geometry: number[] = [], motion: number[] = [], times: number[] = [], frameNumbers: number[] = []
    const activity = new VisualActivityDetector()
    let previous: Uint8ClampedArray | null = null
    let raf = 0, lastDetect = 0, lastBuffer = 0, lastSync = 0, lastVideo = -1, totalFrames = 0
    let utterance = 0, speechStart = 0, lastQueued = 0, lastMovement = -Infinity, lastPublished = 0
    let speaking = false, vadEnergy = 0, idleCleared = true, faceWasVisible = false, disposed = false
    let lastLogged = '', lastLoggedUtterance = -1
    type Window = { frames: string[]; utterance: number; capture?: WordCapture }
    let pending: Window | null = null
    const token = session.current
    const clearPrediction = () => {
      smoother.current.reset(); wordConsensus.current.reset(); backendRef.current.setLastPrediction(null); backendRef.current.setLastWordPrediction(null)
      setReadout(r => ({ ...r, detected: null, confidence: null, top3: null, predictedAt: null, stabilizing: false, smoothed: false, quality: null, serverLatencyMs: null, ramMb: null, cpuPercent: null, throughput: null }))
    }
    const dispatch = async (window: Window) => {
      if (disposed || token !== session.current || !backendRef.current.inferenceReady || window.utterance !== utterance) return
      if (backendRef.current.recognitionMode !== 'alphabet' && publication.current.has(window.utterance)) return
      if (inFlight.current) { pending = window; return }
      inFlight.current = true; setPredicting(true)
      setCapturedClip([...window.frames])
      const controller = new AbortController(); request.current = controller
      const began = performance.now()
      try {
        if (backendRef.current.recognitionMode !== 'alphabet') {
          const rawWord = await backendRef.current.predictWord(window.frames, controller.signal, window.capture)
          if (disposed || controller.signal.aborted || token !== session.current || window.utterance !== utterance) return
          if (!rawWord) { clearPrediction(); return }
          const consistent = wordConsensus.current.update(rawWord.predicted_word, rawWord.uncertain, window.utterance)
          const confirmed = lrwMode ? window.capture?.final === true && rawWord.diagnostics?.temporal_consensus?.confirmed === true && !rawWord.uncertain : consistent
          const at = Date.now(), latencyMs = Math.round(performance.now() - began)
          lastPublished = performance.now(); idleCleared = false
          setReadout(r => ({
            ...r,
            detected: confirmed ? rawWord.display_word : null,
            confidence: rawWord.confidence,
            top3: rawWord.top_predictions.map(p => ({ letter: p.word, prob: p.prob })),
            uncertaintyState: rawWord.diagnostics?.rejection_status ?? (rawWord.uncertain ? 'UNCERTAIN' : 'ACCEPTED'),
            latencyMs,
            predictedAt: at,
            stabilizing: !rawWord.uncertain && !confirmed,
            smoothed: false,
            serverLatencyMs: rawWord.latency_ms ?? null,
            ramMb: rawWord.performance?.memory_after?.rss_bytes ? rawWord.performance.memory_after.rss_bytes / 1048576 : null,
            cpuPercent: rawWord.performance?.process_cpu_percent_one_core ?? null,
            throughput: rawWord.performance?.requests_per_second_sequential ?? null
          }))
          if (confirmed && publication.current.claim(utterance)) {
            lastLogged = rawWord.predicted_word; lastLoggedUtterance = utterance
            setHistory(items => [{ id: at, letter: rawWord.display_word, confidence: rawWord.confidence, at, latencyMs }, ...items].slice(0, 12))
          }
          return
        }

        const raw = await backendRef.current.predict(window.frames, controller.signal)
        if (disposed || controller.signal.aborted || token !== session.current || window.utterance !== utterance) return
        if (!raw) { clearPrediction(); return }
        const result = smoother.current.update(raw)
        const at = Date.now(), latencyMs = Math.round(performance.now() - began)
        if (!result.prediction) {
          // Provide instant visual and latency feedback while stabilizing
          setReadout(r => ({
            ...r,
            detected: null,
            confidence: raw.confidence,
            top3: raw.top3,
            latencyMs,
            serverLatencyMs: raw.latency_ms ?? null,
            uncertaintyState: raw.uncertain ? 'UNCERTAIN' : 'STABILIZING',
            stabilizing: !raw.uncertain
          }))
          return
        }
        const prediction = result.prediction
        lastPublished = performance.now(); idleCleared = false
        backendRef.current.setLastPrediction(prediction)
        setReadout(r => ({ ...r, detected: prediction.letter, confidence: prediction.confidence, top3: prediction.top3, uncertaintyState: prediction.uncertainty?.state ?? null, latencyMs, predictedAt: at, stabilizing: false, smoothed: prediction.stabilization?.method === 'ema', serverLatencyMs: raw.latency_ms ?? null }))
        if (prediction.letter !== lastLogged || utterance !== lastLoggedUtterance) {
          lastLogged = prediction.letter; lastLoggedUtterance = utterance
          setHistory(items => [{ id: at, letter: prediction.letter, confidence: prediction.confidence, at, latencyMs }, ...items].slice(0,12))
        }
      } finally {
        if (request.current === controller) {
          request.current = null; inFlight.current = false
          const next = pending; pending = null
          if (!disposed && next && next.utterance === utterance && !(backendRef.current.recognitionMode !== 'alphabet' && publication.current.has(next.utterance))) void dispatch(next)
          else if (!disposed) setPredicting(false)
        }
      }
    }
    const loop = (now: number) => {
      raf = requestAnimationFrame(loop)
      const video = cam.videoRef.current, landmarker = lm.landmarkerRef.current
      if (!video || !landmarker || !captureCtx || video.readyState < 2 || video.currentTime === lastVideo || now - lastDetect < cadence) return
      lastVideo = video.currentTime; lastDetect = now
      let quality: string | null = null
      let faceOk = false, cropOk = false, openness: number | null = null, motionEnergy: number | null = null
      const canvas = overlayRef.current, ctx = canvas?.getContext('2d')
      if (canvas && ctx) {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) { canvas.width = video.videoWidth; canvas.height = video.videoHeight }
        ctx.clearRect(0, 0, canvas.width, canvas.height)
      }
      try {
        const faces = landmarker.detectForVideo(video, now).faceLandmarks
        const all = faces.length === 1 ? faces[0] : undefined
        quality = faces.length > 1 ? 'Multiple faces detected. Keep one person in view.' : !faces.length ? 'No face detected.' : null
        faceOk = !!all?.length
        if (all?.length) {
          if (ctx && canvas && overlayEnabled.current) {
            ctx.fillStyle = '#8fdfcb'
            for (const index of new Set(MOUTH_LANDMARK_INDICES)) { const p = all[index]; if (!p) continue; ctx.beginPath(); ctx.arc(p.x * canvas.width, p.y * canvas.height, 2, 0, Math.PI * 2); ctx.fill() }
          }
          const alignment = mouthAlignment(all, video.videoWidth, video.videoHeight)
          if (alignment) {
            cropOk = true
            setPipelineError(null)
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
            let sum=0, squared=0
            for(let i=0;i<pixels.length;i+=4){const g=(pixels[i]+pixels[i+1]+pixels[i+2])/3;sum+=g;squared+=g*g}
            const brightness=sum/(96*96), contrast=Math.sqrt(Math.max(0,squared/(96*96)-brightness*brightness))
            if(brightness<15 || brightness>240 || contrast<8){cropOk=false;quality='Improve front lighting and avoid glare.'}
            geometry.push(openness); if (geometry.length > 90) geometry.shift()
            if (motionEnergy !== null) { motion.push(motionEnergy); if (motion.length > 90) motion.shift() }
            spark(geometryRef.current,geometry,Math.max(.6,...geometry)); spark(motionRef.current,motion,Math.max(.2,...motion))
            if (cropOk && now - lastBuffer >= cadence) {
              lastBuffer = now; totalFrames++
              if (manualWords) {
                if (assistantController.getSnapshot().phase === 'CAPTURING') {
                  let encoded: string
                  let measured: number[][] | undefined
                  if (lrwMode && fullCtx) {
                    const scale = Math.min(1, 512 / Math.max(video.videoWidth, video.videoHeight))
                    fullFace.width = Math.round(video.videoWidth * scale); fullFace.height = Math.round(video.videoHeight * scale)
                    fullCtx.drawImage(video, 0, 0, fullFace.width, fullFace.height)
                    measured = compactLandmarks(all) ?? undefined
                    encoded = fullFace.toDataURL('image/jpeg', .85)
                  } else encoded = capture.toDataURL('image/jpeg', .75)
                  if (assistantController.sample({ image: encoded, time: now, landmarks: measured })) {
                    buffer.current.push(encoded)
                    const strip = bufferStripRef.current
                    strip?.getContext('2d')?.drawImage(capture, (buffer.current.length - 1) * 24, 0, 24, 24)
                  }
                }
              } else {
              if(lrwMode && captureGapExceeded(captureTimes[captureTimes.length-1],now)){
                utterance++;pending=null;buffer.current=[];frameNumbers.length=0;captureTimes.length=0;capturedLandmarks.length=0
                activity.reset();speaking=false;vadEnergy=0;previous=null;clearPrediction()
                quality='Capture paused for over 160 ms. Hold still while a fresh word window builds.'
              }
              let encoded: string
              if(lrwMode && fullCtx){
                const scale=Math.min(1,512/Math.max(video.videoWidth,video.videoHeight))
                fullFace.width=Math.round(video.videoWidth*scale);fullFace.height=Math.round(video.videoHeight*scale)
                fullCtx.drawImage(video,0,0,fullFace.width,fullFace.height)
                const measured=compactLandmarks(all)
                if(!measured){quality='Landmarks extend outside the frame. Center your face.';cropOk=false;throw new Error(quality)}
                encoded=fullFace.toDataURL('image/jpeg',.85);capturedLandmarks.push(measured);captureTimes.push(now)
              } else encoded=capture.toDataURL('image/jpeg',.75)
              buffer.current.push(encoded); frameNumbers.push(totalFrames)
              if (buffer.current.length > sequence.maxFrames) { buffer.current.shift(); frameNumbers.shift(); if(lrwMode){capturedLandmarks.shift();captureTimes.shift()} }
              const measured = activity.update(openness,now), wasSpeaking = speaking
              speaking = measured.speaking; vadEnergy = measured.energy
              if (speaking && !wasSpeaking && now - lastMovement > 700) {
                utterance++; speechStart = lrwMode ? frameNumbers[utteranceStartIndex(captureTimes,now)] : totalFrames - 7; lastQueued = speechStart
                // Aborting fetch cannot cancel a running CPU model. Await it,
                // discard its stale response, and keep only the newest window.
                pending = null; if(!lrwMode)request.current?.abort(); clearPrediction(); idleCleared = false
              }
              lastMovement = measured.lastMovement
              const selected = buffer.current.filter((_frame,index) => frameNumbers[index] >= speechStart)
              const ending = wasSpeaking && !speaking
              const tooLong=lrwMode && (speaking || ending) && totalFrames-speechStart>sequence.maxFrames
              if(tooLong){quality='Utterance is too long. Pause, then say one word.';clearPrediction()}
              if (!tooLong && (lrwMode ? ending : speaking || ending) && selected.length >= sequence.minFrames && (lrwMode || totalFrames-lastQueued >= sequence.stride || (ending && totalFrames-lastQueued >= 2))) {
                lastQueued = totalFrames
                const indexes=frameNumbers.map((n,i)=>n>=speechStart?i:-1).filter(i=>i>=0)
                const metadata: WordCapture | undefined=lrwMode ? {input_kind:'webcam',convention:LRW_CONVENTION,landmarks:indexes.map(i=>capturedLandmarks[i]),timestamps_ms:indexes.map(i=>captureTimes[i]),face_counts:indexes.map(()=>1),final:ending} : undefined
                void dispatch({ frames: [...selected], utterance, capture:metadata })
              }
              const strip = bufferStripRef.current, stripCtx = strip?.getContext('2d')
              if (strip && stripCtx) {
                if (buffer.current.length === sequence.maxFrames) stripCtx.drawImage(strip,24,0,strip.width-24,24,0,0,strip.width-24,24)
                stripCtx.drawImage(capture,(buffer.current.length-1)*24,0,24,24)
              }
              }
            }
          }
        }
      } catch (error) {
        faceOk = false; setPipelineError(error instanceof Error ? error.message : 'Landmark processing failed. Retry the camera.')
      }
      if ((!faceOk || !cropOk) && faceWasVisible) {
        if (manualWords) {
          if (assistantController.getSnapshot().phase === 'CAPTURING') assistantController.fail(quality || 'Face tracking was lost. Center your face and retry this word.')
        } else {
        utterance++; pending = null; buffer.current = []; frameNumbers.length = 0;captureTimes.length=0;capturedLandmarks.length=0
        activity.reset(); speaking = false; vadEnergy = 0; previous = null; if(!lrwMode)request.current?.abort(); clearPrediction()
        for (const ref of [appearanceRef,bufferStripRef]) { const c=ref.current;c?.getContext('2d')?.clearRect(0,0,c.width,c.height) }
        }
      }
      if (!manualWords && !speaking && !inFlight.current && !pending && !idleCleared && now-Math.max(lastMovement,lastPublished)>1800) { clearPrediction(); idleCleared = true }
      faceWasVisible = faceOk && cropOk
      times.push(now); if (times.length>20) times.shift()
      if (now-lastSync>=250) {
        lastSync=now
        setReadout(r=>({...r,frames:buffer.current.length,faceOk,cropOk,quality,openness,motionEnergy,speaking,vadEnergy,elapsedSeconds:Math.floor((now-startedAt.current)/1000),fps:times.length>1?Math.round((times.length-1)*1000/(now-times[0])):0}))
      }
    }
    raf=requestAnimationFrame(loop)
    return()=>{disposed=true;pending=null;cancelAnimationFrame(raf);request.current?.abort();inFlight.current=false}
  },[running,lm.status,lm.landmarkerRef,cam.videoRef,manualWords,assistantController])
  return {
    videoRef:cam.videoRef,camStatus:cam.status,camError:cam.error,devices:cam.devices,
    lmStatus:lm.status,lmError:lm.error,backendStatus:backend.status,
    modelLoaded:backend.inferenceReady,inferenceReady:backend.inferenceReady,inferenceReason:backend.inferenceReason,modelKind:backend.modelKind,
    predictionError:backend.predictionError,predicting:manualWords ? assistant.state.phase === 'PROCESSING' : predicting,running,readout,history,pipelineError,clearHistory,
    assistant:manualWords ? assistant : null, canCaptureWord, beginWord, capturedFrames: capturedClip,
    recognitionMode:backend.recognitionMode,setRecognitionMode:backend.setRecognitionMode,wordHealth:backend.wordHealth,expandedHealth:backend.expandedHealth,personalHealth:backend.personalHealth,lrwHealth:backend.lrwHealth,showOverlay,setShowOverlay,
    overlayRef,appearanceRef,geometryRef,motionRef,bufferStripRef,sequence:manualWords ? { ...WORD_LIMITS[backend.recognitionMode === 'alphabet' ? 'word' : backend.recognitionMode], stride: 1 } : backend.recognitionMode === 'lrw500' ? {maxFrames:76,minFrames:12,stride:8} : LIVE_SEQUENCE,start,stop,reset,
  }
}
