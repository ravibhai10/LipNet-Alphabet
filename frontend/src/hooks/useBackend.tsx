import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import {
  probeHealth,
  fetchDatasetStats,
  requestPrediction,
  type BackendStatus,
  type Prediction,
} from '../lib/api'

interface BackendContextValue {
  status: BackendStatus
  modelLoaded?: boolean
  datasetStats: Record<string, number> | null
  /** Latest real prediction returned by the backend (null when offline). */
  lastPrediction: Prediction | null
  setLastPrediction: (p: Prediction | null) => void
  refresh: () => void
  /** Send mouth crops (base64 JPEG, temporal order) to the backend. Null when offline. */
  predict: (frames: string[]) => Promise<Prediction | null>
}

const BackendContext = createContext<BackendContextValue>({
  status: 'idle',
  datasetStats: null,
  lastPrediction: null,
  setLastPrediction: () => {},
  refresh: () => {},
  predict: async () => null,
})

export function BackendProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<BackendStatus>('idle')
  const [modelLoaded, setModelLoaded] = useState<boolean | undefined>(undefined)
  const [datasetStats, setDatasetStats] = useState<Record<string, number> | null>(null)
  const [lastPrediction, setLastPrediction] = useState<Prediction | null>(null)
  const onlineRef = useRef(false)

  const refresh = useCallback(() => {
    setStatus('checking')
    void probeHealth().then((info) => {
      onlineRef.current = info.status === 'online'
      setStatus(info.status)
      setModelLoaded(info.modelLoaded)
      if (onlineRef.current) {
        void fetchDatasetStats().then(setDatasetStats)
      }
    })
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  const predict = useCallback(async (frames: string[]) => {
    if (!onlineRef.current) return null
    return requestPrediction(frames)
  }, [])

  const value = useMemo(
    () => ({
      status,
      modelLoaded,
      datasetStats,
      lastPrediction,
      setLastPrediction,
      refresh,
      predict,
    }),
    [status, modelLoaded, datasetStats, lastPrediction, refresh, predict],
  )

  return <BackendContext.Provider value={value}>{children}</BackendContext.Provider>
}

export function useBackend(): BackendContextValue {
  return useContext(BackendContext)
}
