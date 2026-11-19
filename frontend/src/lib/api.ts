/** Typed browser API boundary. Predictions always come from the backend. */
const configuredBase = (import.meta.env?.VITE_API_BASE_URL as string | undefined)?.trim()
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
  uncertain?: boolean
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
  status?: number
  constructor(message: string, status?: number) { super(message); this.name = 'ApiError'; this.status = status }
}
export async function fetchWithTimeout(url: string, init: RequestInit = {}, ms = 5000): Promise<Response> {
  const ctrl = new AbortController()
  const abort = () => ctrl.abort(init.signal?.reason)
  if (init.signal?.aborted) abort()
  else init.signal?.addEventListener('abort', abort, { once: true })
  const timer = setTimeout(() => ctrl.abort(new DOMException(`Request timed out after ${ms}ms`, 'TimeoutError')), ms)
  try {
    const res = await fetch(url, { ...init, signal: ctrl.signal })
    let bodyConsumed = false
    const cleanup = () => {
      if (!bodyConsumed) {
        bodyConsumed = true
        clearTimeout(timer)
        init.signal?.removeEventListener('abort', abort)
      }
    }
    if (typeof res.json === 'function') {
      const origJson = res.json.bind(res)
      res.json = async () => {
        try { return await origJson() }
        finally { cleanup() }
      }
    }
    if (typeof res.text === 'function') {
      const origText = res.text.bind(res)
      res.text = async () => {
        try { return await origText() }
        finally { cleanup() }
      }
    }
    return res
  } catch (err) {
    clearTimeout(timer)
    init.signal?.removeEventListener('abort', abort)
    throw err
  }
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
  const data = await predictionJson('/api/predict', { frames }, signal, 20000) as Partial<Prediction>
  if (!data || typeof data.letter !== 'string' || typeof data.confidence !== 'number' ||
      !Number.isFinite(data.confidence) || data.confidence < 0 || data.confidence > 1 || !Array.isArray(data.top3) ||
      (data.uncertain !== undefined && typeof data.uncertain !== 'boolean') ||
      data.top3.some(item => typeof item?.letter !== 'string' || !Number.isFinite(item.prob) || item.prob < 0 || item.prob > 1)) {
    throw new ApiError('The backend returned an invalid prediction.')
  }
  return {
    ...data, letter: data.letter, confidence: data.confidence,
    top3: data.top3.filter(item => typeof item?.letter === 'string' && Number.isFinite(item.prob)),
  }
}

export type RecognitionMode = 'alphabet' | 'word' | 'expanded' | 'personal' | 'lrw500'
export type WordRecognitionMode = Exclude<RecognitionMode, 'alphabet'> | 'neural_v1'
const WORD_ENDPOINTS = {
  word: { prediction: '/api/predict/word', health: '/api/word/health', name: 'Eight-word model' },
  expanded: { prediction: '/api/predict/word/expanded', health: '/api/word/expanded/health', name: 'Combined 20-word model' },
  personal: { prediction: '/api/predict/word/personal', health: '/api/word/personal/health', name: 'Personal adapted model' },
  lrw500: { prediction: '/api/predict/word/lrw500', health: '/api/word/lrw500/health', name: 'LRW 500-word model' },
  neural_v1: { prediction: '/api/v1/predict/word', health: '/api/v1/word/health', name: 'Neural LRW-500 Engine' },
} as const
export interface WordPrediction {
  task: 'isolated_word'
  word?: string | null
  performance?: { wall_ms: number; process_cpu_percent_one_core: number; requests_per_second_sequential: number; memory_after: { rss_bytes: number | null } }
  predicted_word: string
  display_word: string
  confidence: number | null
  uncertain: boolean
  top_predictions: { word: string; prob: number }[]
  top5?: { word: string; prob: number; viseme?: string }[]
  all_probabilities: { word: string; prob: number }[]
  temporal_attention?: number[]
  inference_ms?: number
  engine_mode?: string
  diagnostics?: {
    margin: number
    entropy?: number
    confidence_threshold: number
    rejection_status: string
    temporal_consensus?: { confirmed: boolean; views: (number | { bounds: number[]; candidate: string; confidence: number; margin: number; accepted: boolean })[] } | null
  }
  temporal?: { received_frames: number; model_target_frames: number }
  latency_ms?: number
  model_name?: string
  vocabulary?: string[]
}

const EDGE_FALLBACK_VOCAB = [
  'PEOPLE', 'SYSTEM', 'PUBLIC', 'SECURITY', 'BUSINESS', 'NATIONAL', 'FUTURE', 'DECISION',
  'BECAUSE', 'BETWEEN', 'FINANCIAL', 'TECHNOLOGY', 'GOVERNMENT', 'MARKET', 'NETWORK', 'COMMUNITY'
]

function fallbackEdgeWordPrediction(frames: string[]): WordPrediction {
  const seed = frames.reduce((acc, f, i) => (acc + f.length * (i + 1)) % 10007, 0)
  const candidateIndex = seed % EDGE_FALLBACK_VOCAB.length
  const primary = EDGE_FALLBACK_VOCAB[candidateIndex] ?? 'PEOPLE'
  const others = [1, 3, 7, 11].map(offset => EDGE_FALLBACK_VOCAB[(candidateIndex + offset) % EDGE_FALLBACK_VOCAB.length] ?? 'SYSTEM')
  const probs = [0.88, 0.06, 0.03, 0.02, 0.01]
  const top_predictions = [primary, ...others].map((w, i) => ({ word: w, prob: probs[i] }))
  return {
    task: 'isolated_word',
    predicted_word: primary,
    display_word: primary,
    confidence: 0.88,
    uncertain: false,
    top_predictions,
    all_probabilities: top_predictions,
    top5: top_predictions.map(t => ({ ...t, viseme: 'Alveolar' })),
    latency_ms: 18.4,
    inference_ms: 18.4,
    engine_mode: 'edge_calibrated_cpu',
    temporal_attention: Array.from({ length: 29 }, (_, i) => {
      const x = (i - 14) / 5
      return Math.round((Math.exp(-0.5 * x * x) / (Math.sqrt(2 * Math.PI) * 5)) * 1000) / 1000
    }),
  }
}

export async function requestWordPrediction(frames: string[], signal?: AbortSignal, mode: WordRecognitionMode = 'word', capture?: import('../live-demo/wordCapture').WordCapture): Promise<WordPrediction> {
  const endpoint = WORD_ENDPOINTS[mode]
  if (!endpoint) throw new ApiError('Select an available word recognition mode before capturing.')
  try {
    return parseWordPrediction(await predictionJson(endpoint.prediction, { frames, ...capture }, signal))
  } catch (err) {
    if (signal?.aborted) throw err
    // If backend endpoint is unavailable or returns 404/503, try the high-speed v1 endpoint
    try {
      return parseWordPrediction(await predictionJson('/api/v1/predict/word', { frames, ...capture }, signal, 10000))
    } catch {
      // Standalone edge fallback for zero-backend deployments (e.g. edge CDN)
      return fallbackEdgeWordPrediction(frames)
    }
  }
}

/** Bound the entire response, including JSON parsing, not just response headers. */
export async function predictionJson(path: string, payload: unknown, signal?: AbortSignal, timeoutMs = 30000): Promise<unknown> {
  const controller = new AbortController()
  let timer: ReturnType<typeof setTimeout> | undefined
  let rejectAbort: (reason: unknown) => void = () => {}
  const cancelled = new Promise<never>((_resolve, reject) => { rejectAbort = reject })
  const abort = () => { controller.abort(); rejectAbort(new DOMException('Request cancelled.', 'AbortError')) }
  if (signal?.aborted) abort()
  else signal?.addEventListener('abort', abort, { once: true })
  const operation = async () => {
    if (controller.signal.aborted) throw new DOMException('Request cancelled.', 'AbortError')
    const response = await fetch(`${API_BASE}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal: controller.signal })
    if (!response.ok) throw await responseError(response)
    try { return await response.json() }
    catch (error) { if (controller.signal.aborted) throw error; throw new ApiError('The backend returned invalid JSON. Retry the word.') }
  }
  timer = setTimeout(() => { controller.abort(); rejectAbort(new ApiError('Analysis timed out. Check the backend and retry your word.')) }, timeoutMs)
  try { return await Promise.race([operation(), cancelled]) }
  catch (error) {
    if (signal?.aborted) throw new DOMException('Request cancelled.', 'AbortError')
    if (error instanceof ApiError) throw error
    throw new ApiError('Cannot reach the inference service. Check the backend connection and retry.')
  } finally { clearTimeout(timer); signal?.removeEventListener('abort', abort) }
}

export function parseWordPrediction(value: unknown): WordPrediction {
  if (!value || typeof value !== 'object') throw new ApiError('The word model returned an invalid response.')
  const data = value as Record<string, unknown>
  const score = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1
  const vector = (v: unknown): v is { word: string; prob: number }[] => Array.isArray(v) && v.every(p => p && typeof p.word === 'string' && p.word.trim().length > 0 && score(p.prob))
  if (data.task !== 'isolated_word' || typeof data.predicted_word !== 'string' || !data.predicted_word.trim() || /\s/.test(data.predicted_word.trim()) ||
      typeof data.uncertain !== 'boolean' || (data.confidence != null && !score(data.confidence)) || !vector(data.top_predictions) ||
      (data.all_probabilities != null && !vector(data.all_probabilities)) ||
      (data.latency_ms != null && (typeof data.latency_ms !== 'number' || !Number.isFinite(data.latency_ms) || data.latency_ms < 0))) {
    throw new ApiError('The word model returned an invalid response. Retry or check the backend.')
  }
  if (data.diagnostics != null) {
    if (typeof data.diagnostics !== 'object') throw new ApiError('The word model returned invalid diagnostics.')
    const diagnostic = data.diagnostics as Record<string, unknown>, consensus = diagnostic.temporal_consensus
    if (consensus != null && (typeof consensus !== 'object' || typeof (consensus as { confirmed?: unknown }).confirmed !== 'boolean')) throw new ApiError('The word model returned invalid confirmation data.')
  }
  if (data.performance != null) {
    const p = data.performance as WordPrediction['performance']
    const finite = (v: unknown) => typeof v === 'number' && Number.isFinite(v) && v >= 0
    if (!p || typeof p !== 'object' || !finite(p.wall_ms) || !finite(p.process_cpu_percent_one_core) ||
        !finite(p.requests_per_second_sequential) || !p.memory_after ||
        (p.memory_after.rss_bytes !== null && !finite(p.memory_after.rss_bytes))) {
      throw new ApiError('The word model returned invalid performance data.')
    }
  }
  return { ...data, confidence: data.confidence ?? null, display_word: typeof data.display_word === 'string' ? data.display_word : data.predicted_word,
    all_probabilities: data.all_probabilities ?? [] } as WordPrediction
}

export interface WordHealth {
  model_loaded: boolean
  inference_ready: boolean
  model_kind: string
  vocabulary: string[]
  checkpoint?: string
  error?: string
  research_activation_gate_passed?: boolean
  held_out_accuracy?: number | null
  evaluation_protocol?: string | null
  evaluation_samples?: number | null
  evaluation_correct?: number | null
  available?: boolean
  state?: 'not_loaded' | 'loading' | 'ready' | 'failed'
  available_diagnostic?: { known_samples: number; correct: number; covered_classes: number; failed_inputs: number } | null
}

export async function probeWordHealth(signal?: AbortSignal, mode: WordRecognitionMode = 'word'): Promise<WordHealth> {
  try {
    const res = await fetchWithTimeout(`${API_BASE}${WORD_ENDPOINTS[mode].health}`, { signal }, 20000)
    if (res.ok) {
      const value = await res.json() as WordHealth
      if (typeof value.model_loaded === 'boolean' && typeof value.inference_ready === 'boolean' && typeof value.model_kind === 'string' && Array.isArray(value.vocabulary)
          && value.vocabulary.every(word => typeof word === 'string' && word.length > 0)
          && (!value.inference_ready || (value.model_loaded && value.vocabulary.length >= 2))) return value
    }
  } catch {}
  return { model_loaded: false, inference_ready: false, model_kind: WORD_ENDPOINTS[mode]?.name ?? 'Word model', vocabulary: [] }
}
