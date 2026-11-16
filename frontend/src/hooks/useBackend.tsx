import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import {
  probeHealth,
  fetchDatasetStats,
  requestPrediction,
  requestWordPrediction,
  probeWordHealth,
  type BackendStatus,
  type Prediction,
  type WordPrediction,
  type WordHealth,
  type HealthInfo,
  type RecognitionMode,
  type ModelAvailability
} from '../lib/api'

interface BackendContextValue {
  status: BackendStatus
  modelLoaded?: boolean
  inferenceReady: boolean
  inferenceReason?: string
  modelKind?: string
  capabilities?: ModelAvailability
  predictionError: string | null
  datasetStats: Record<string, number> | null
  lastPrediction: Prediction | null
  setLastPrediction: (p: Prediction | null) => void
  lastWordPrediction: WordPrediction | null
  setLastWordPrediction: (p: WordPrediction | null) => void
  recognitionMode: RecognitionMode
  setRecognitionMode: (m: RecognitionMode) => void
  wordHealth: WordHealth
  expandedHealth: WordHealth
  lrwHealth: WordHealth
  refresh: () => void
  predict: (frames: string[], signal?: AbortSignal) => Promise<Prediction | null>
  predictWord: (frames: string[], signal?: AbortSignal, capture?: import('../live-demo/wordCapture').WordCapture, throwErrors?: boolean) => Promise<WordPrediction | null>
}
const BackendContext = createContext<BackendContextValue>({
  status: 'idle', inferenceReady: false, predictionError: null,
  datasetStats: null, lastPrediction: null, setLastPrediction: () => {},
  lastWordPrediction: null, setLastWordPrediction: () => {},
  recognitionMode: 'alphabet', setRecognitionMode: () => {},
  lrwHealth: { model_loaded: false, inference_ready: false, model_kind: 'LRW 500', vocabulary: [] },
  wordHealth: { model_loaded: false, inference_ready: false, model_kind: 'Unavailable', vocabulary: [] },
  expandedHealth: { model_loaded: false, inference_ready: false, model_kind: 'Combined 20 words', vocabulary: [] },
  refresh: () => {}, predict: async () => null, predictWord: async () => null,
})
export function BackendProvider({ children }: { children: ReactNode }) {
  const [health, setHealth] = useState<HealthInfo>({ status: 'idle', inferenceReady: false })
  const [wordHealth, setWordHealth] = useState<WordHealth>({
    model_loaded: false, inference_ready: false, model_kind: 'Unavailable', vocabulary: []
  })
  const [lrwHealth, setLrwHealth] = useState<WordHealth>({ model_loaded: false, inference_ready: false, model_kind: 'LRW 500', vocabulary: [] })
  const [expandedHealth, setExpandedHealth] = useState<WordHealth>({ model_loaded: false, inference_ready: false, model_kind: 'Combined 20 words', vocabulary: [] })
  const [recognitionMode, setRecognitionModeState] = useState<RecognitionMode>('alphabet')
  const modeRef = useRef<RecognitionMode>('alphabet'); modeRef.current = recognitionMode
  const modeChosen = useRef(false)
  const [datasetStats, setDatasetStats] = useState<Record<string, number> | null>(null)
  const [lastPrediction, setLastPrediction] = useState<Prediction | null>(null)
  const [lastWordPrediction, setLastWordPrediction] = useState<WordPrediction | null>(null)
  const [predictionError, setPredictionError] = useState<string | null>(null)
  const healthRef = useRef(health)
  const activeRef = useRef(false)
  const refreshRef = useRef<Promise<void> | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const setRecognitionMode = useCallback((mode: RecognitionMode) => {
    modeChosen.current = true
    modeRef.current = mode
    setRecognitionModeState(mode)
    setLastPrediction(null); setLastWordPrediction(null); setPredictionError(null)
  }, [])

  const refreshAsync = useCallback(() => {
    if (refreshRef.current) return refreshRef.current
    const controller = new AbortController()
    abortRef.current = controller
    const work = async () => {
      setHealth(current => current.status === 'idle' ? { ...current, status: 'checking' } : current)
      const [info, wHealth] = await Promise.all([
        probeHealth(controller.signal),
        probeWordHealth(controller.signal)
      ])
      if (controller.signal.aborted || !activeRef.current) return
      healthRef.current = info
      setHealth(info)
      setWordHealth(wHealth)
      if (!modeChosen.current && wHealth.inference_ready && wHealth.research_activation_gate_passed === true) {
        modeChosen.current = true
        setRecognitionModeState('word')
      }
      // Load separately after the fallback health call to avoid overlapping cold encoder loads.
      const lrw = await probeWordHealth(controller.signal, 'lrw500')
      if (controller.signal.aborted || !activeRef.current) return
      setLrwHealth(lrw)
      const expanded = await probeWordHealth(controller.signal, 'expanded')
      if (controller.signal.aborted || !activeRef.current) return
      setExpandedHealth(expanded)
      if (info.status === 'online') {
        const stats = await fetchDatasetStats(controller.signal)
        if (!controller.signal.aborted && activeRef.current) setDatasetStats(stats)
      } else {
        setDatasetStats(null)
        setLastPrediction(null)
        setLastWordPrediction(null)
      }
    }
    const pending = work().finally(() => { if (refreshRef.current === pending) refreshRef.current = null })
    refreshRef.current = pending
    return pending
  }, [])

  const refresh = useCallback(() => { void refreshAsync() }, [refreshAsync])
  useEffect(() => {
    activeRef.current = true
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    let delay = 1000
    const poll = async () => {
      if (cancelled) return
      await refreshAsync()
      if (cancelled) return
      delay = healthRef.current.status === 'online' ? 5000 : Math.min(delay * 2, 30000)
      timer = setTimeout(() => void poll(), delay)
    }
    void poll()
    return () => {
      cancelled = true; activeRef.current = false
      clearTimeout(timer); abortRef.current?.abort(); refreshRef.current = null
    }
  }, [refreshAsync])

  const predict = useCallback(async (frames: string[], signal?: AbortSignal) => {
    if (signal?.aborted) return null
    if (!healthRef.current.inferenceReady) {
      setPredictionError(healthRef.current.inferenceReason ?? 'Inference is not ready.')
      return null
    }
    setPredictionError(null)
    try {
      const prediction = await requestPrediction(frames, signal)
      return signal?.aborted ? null : prediction
    } catch (error) {
      if (!signal?.aborted && activeRef.current) {
        setPredictionError(error instanceof Error && error.name !== 'AbortError'
          ? error.message : 'Inference timed out. Try a shorter capture.')
      }
      return null
    }
  }, [])

  const predictWord = useCallback(async (frames: string[], signal?: AbortSignal, capture?: import('../live-demo/wordCapture').WordCapture, throwErrors = false) => {
    if (signal?.aborted) return null
    const requestedMode = modeRef.current
    if (requestedMode === 'alphabet') return null
    setPredictionError(null)
    try {
      const prediction = await requestWordPrediction(frames, signal, requestedMode, capture)
      if (!signal?.aborted && activeRef.current && modeRef.current === requestedMode) {
        setLastWordPrediction(prediction)
      }
      return signal?.aborted || modeRef.current !== requestedMode ? null : prediction
    } catch (error) {
      if (!signal?.aborted && activeRef.current) {
        setPredictionError(error instanceof Error && error.name !== 'AbortError'
          ? error.message : 'Word inference timed out. Try again.')
      }
      if (throwErrors) throw error
      return null
    }
  }, [])

  const selectedWordHealth = recognitionMode === 'lrw500' ? lrwHealth : recognitionMode === 'expanded' ? expandedHealth : wordHealth
  const value = useMemo(() => ({
    ...health,
    inferenceReady: recognitionMode !== 'alphabet' ? selectedWordHealth.inference_ready : health.inferenceReady,
    modelKind: recognitionMode !== 'alphabet' ? selectedWordHealth.model_kind : health.modelKind,
    inferenceReason: recognitionMode !== 'alphabet' ? selectedWordHealth.inference_ready ? undefined : selectedWordHealth.error || (selectedWordHealth.state === 'loading' ? 'Loading the word model on CPU…' : 'The selected word model is not ready.') : health.inferenceReason,
    predictionError,
    datasetStats,
    lastPrediction,
    setLastPrediction,
    lastWordPrediction,
    setLastWordPrediction,
    recognitionMode,
    setRecognitionMode,
    wordHealth,
    expandedHealth,
    lrwHealth,
    refresh,
    predict,
    predictWord,
  }), [
    health,
    predictionError,
    datasetStats,
    lastPrediction,
    lastWordPrediction,
    recognitionMode,
    wordHealth,
    expandedHealth,
    lrwHealth,
    refresh,
    predict,
    predictWord,
  ])
  return <BackendContext.Provider value={value}>{children}</BackendContext.Provider>
}
export function useBackend(): BackendContextValue { return useContext(BackendContext) }
