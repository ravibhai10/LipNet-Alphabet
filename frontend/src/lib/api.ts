/** Typed browser API boundary. Predictions always come from the backend. */
const configuredBase = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim()
export const API_BASE = configuredBase?.replace(/\/$/, '') ?? (
  typeof window !== 'undefined' && ['5173', '5174', '3000'].includes(window.location.port)
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : ''
)
export type BackendStatus = 'idle' | 'checking' | 'online' | 'offline'
export interface ModelAvailability {
  active_backend: string
  alphabet_classifier: { present: boolean; ready: boolean }
  avhubert_encoder: { present: boolean; loaded: boolean }
  avhubert_head: { present: boolean; ready: boolean; compatibility: string }
  sentence_recognition: { ready: boolean }
}
export interface HealthInfo {
  status: BackendStatus
  modelLoaded?: boolean
  inferenceReady: boolean
  inferenceReason?: string
  modelKind?: string
  capabilities?: ModelAvailability
}
export interface Prediction {
  letter: string
  confidence: number
  top3: { letter: string; prob: number }[]
  probabilities?: { letter: string; prob: number }[]
  stabilization?: { method: 'ema' | 'consensus'; windows: number; alpha?: number }
  gates?: { appearance: number; geometry: number; motion: number }
  uncertainty?: { margin?: number; entropy?: number; state?: string }
  temporal?: { frames: number }
  model?: string
  latency_ms?: number
}
export class ApiError extends Error {
  constructor(message: string, public status?: number) { super(message); this.name = 'ApiError' }
}
export async function fetchWithTimeout(url: string, init: RequestInit = {}, ms = 5000): Promise<Response> {
  const ctrl = new AbortController()
  const abort = () => ctrl.abort(init.signal?.reason)
  if (init.signal?.aborted) abort()
  else init.signal?.addEventListener('abort', abort, { once: true })
  const timer = setTimeout(() => ctrl.abort(), ms)
  try { return await fetch(url, { ...init, signal: ctrl.signal }) }
  finally { clearTimeout(timer); init.signal?.removeEventListener('abort', abort) }
}
export async function responseError(res: Response): Promise<ApiError> {
  let detail = `Request failed (${res.status}).`
  try {
    const data = await res.json() as { detail?: unknown; message?: string }
    if (typeof data.detail === 'string') detail = data.detail
    else if (data.message) detail = data.message
    else if (Array.isArray(data.detail)) detail = 'The request contains invalid values.'
  } catch { /* Preserve the HTTP status if the body is not JSON. */ }
  return new ApiError(detail, res.status)
}
export async function probeHealth(signal?: AbortSignal): Promise<HealthInfo> {
  for (const path of ['/health', '/api/health']) {
    try {
      const res = await fetchWithTimeout(`${API_BASE}${path}`, { signal }, 10000)
      if (!res.ok) continue
      const data = await res.json() as Record<string, unknown>
      if (data.status !== 'ok' && data.status !== 'ready') continue
      return {
        status: 'online', modelLoaded: data.model_loaded === true,
        inferenceReady: data.inference_ready === true,
        inferenceReason: typeof data.inference_reason === 'string' ? data.inference_reason : 'Browser inference readiness has not been reported.',
        modelKind: typeof data.model_kind === 'string' ? data.model_kind : undefined,
        capabilities: data.capabilities as ModelAvailability | undefined,
      }
    } catch { if (signal?.aborted) break }
  }
  return { status: 'offline', inferenceReady: false, inferenceReason: 'Cannot reach the local backend.' }
}
export async function fetchDatasetStats(signal?: AbortSignal): Promise<Record<string, number> | null> {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/api/dataset/stats`, { signal })
    if (!res.ok) return null
    const data = await res.json() as Record<string, unknown>
    return Object.fromEntries(Object.entries(data).filter((entry): entry is [string, number] => typeof entry[1] === 'number' && Number.isFinite(entry[1])))
  } catch { return null }
}
export async function requestPrediction(frames: string[], signal?: AbortSignal): Promise<Prediction> {
  const res = await fetchWithTimeout(`${API_BASE}/api/predict`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ frames }), signal,
  }, 20000)
  if (!res.ok) throw await responseError(res)
  const data = await res.json() as Partial<Prediction>
  if (!data || typeof data.letter !== 'string' || typeof data.confidence !== 'number' ||
      !Number.isFinite(data.confidence) || !Array.isArray(data.top3)) {
    throw new ApiError('The backend returned an invalid prediction.')
  }
  return {
    ...data, letter: data.letter, confidence: data.confidence,
    top3: data.top3.filter(item => typeof item?.letter === 'string' && Number.isFinite(item.prob)),
  }
}
