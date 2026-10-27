import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { probeHealth, fetchDatasetStats, requestPrediction, type BackendStatus, type Prediction, type HealthInfo, type ModelAvailability } from '../lib/api'

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
  refresh: () => void
  predict: (frames: string[], signal?: AbortSignal) => Promise<Prediction | null>
}
const BackendContext = createContext<BackendContextValue>({
  status: 'idle', inferenceReady: false, predictionError: null,
  datasetStats: null, lastPrediction: null, setLastPrediction: () => {},
  refresh: () => {}, predict: async () => null,
})
export function BackendProvider({ children }: { children: ReactNode }) {
  const [health, setHealth] = useState<HealthInfo>({ status: 'idle', inferenceReady: false })
  const [datasetStats, setDatasetStats] = useState<Record<string, number> | null>(null)
  const [lastPrediction, setLastPrediction] = useState<Prediction | null>(null)
  const [predictionError, setPredictionError] = useState<string | null>(null)
  const healthRef = useRef(health)
  const activeRef = useRef(false)
  const refreshRef = useRef<Promise<void> | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const refreshAsync = useCallback(() => {
    if (refreshRef.current) return refreshRef.current
    const controller = new AbortController()
    abortRef.current = controller
    const work = async () => {
      setHealth(current => current.status === 'idle' ? { ...current, status: 'checking' } : current)
      const info = await probeHealth(controller.signal)
      if (controller.signal.aborted || !activeRef.current) return
      healthRef.current = info
      setHealth(info)
      if (info.status === 'online') {
        const stats = await fetchDatasetStats(controller.signal)
        if (!controller.signal.aborted && activeRef.current) setDatasetStats(stats)
      } else {
        setDatasetStats(null)
        setLastPrediction(null)
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
      delay = healthRef.current.status === 'online' ? 10000 : Math.min(delay * 2, 30000)
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
  const value = useMemo(() => ({
    ...health, predictionError, datasetStats, lastPrediction, setLastPrediction, refresh, predict,
  }), [health, predictionError, datasetStats, lastPrediction, refresh, predict])
  return <BackendContext.Provider value={value}>{children}</BackendContext.Provider>
}
export function useBackend(): BackendContextValue { return useContext(BackendContext) }
