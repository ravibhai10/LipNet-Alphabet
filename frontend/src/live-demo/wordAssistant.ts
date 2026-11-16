import type { WordPrediction } from '../lib/api'
import type { WordCapture } from './wordCapture'

export type WordMode = 'word' | 'expanded' | 'lrw500'
export type AssistantPhase = 'IDLE' | 'CAMERA_STARTING' | 'READY' | 'CAPTURING' | 'PROCESSING' | 'RESULT' | 'WAITING_NEXT' | 'RETRY' | 'ERROR' | 'STOPPING'
export type Feedback = 'correct' | 'incorrect' | null
export interface WordResult {
  id: number; mode: WordMode; candidate: string; word: string | null
  confidence: number | null; inferenceMs: number | null; roundTripMs: number
  at: number; status: 'available' | 'uncertain'; feedback: Feedback
  captureMs: number; frameCount: number; retryOf: number | null
  candidates: { word: string; prob: number }[]
  performance?: WordPrediction['performance']
}
export interface SessionTotals {
  attempts: number; predictions: number; available: number; uncertain: number
  failedRequests: number; retries: number; inferenceTotalMs: number; timedPredictions: number
}
export interface AssistantState {
  phase: AssistantPhase; mode: WordMode; result: WordResult | null; history: WordResult[]
  error: string | null; attemptId: number; retryOf: number | null
  captureStartedAt: number | null; processingStartedAt: number | null
  captureMs: number; processingMs: number; frames: number; totals: SessionTotals
  minimumScore: number | null
}
export const WORD_LIMITS = {
  word: { minFrames: 8, maxFrames: 76, maxDurationMs: 5000 },
  expanded: { minFrames: 8, maxFrames: 76, maxDurationMs: 5000 },
  lrw500: { minFrames: 12, maxFrames: 76, maxDurationMs: 3000 },
} as const
export const initialAssistant = (): AssistantState => ({
  phase: 'IDLE', mode: 'word', result: null, history: [], error: null, attemptId: 0, retryOf: null,
  captureStartedAt: null, processingStartedAt: null, captureMs: 0, processingMs: 0, frames: 0,
  totals: { attempts: 0, predictions: 0, available: 0, uncertain: 0, failedRequests: 0, retries: 0, inferenceTotalMs: 0, timedPredictions: 0 },
  minimumScore: null,
})
export type AssistantEvent =
  | { type: 'CAMERA_START' | 'CAMERA_READY' | 'NEXT' | 'RETRY' | 'WAIT' | 'STOP' | 'STOPPED' | 'RECOVER' | 'RESET' | 'CLEAR_HISTORY' }
  | { type: 'CAPTURE'; now: number; mode: WordMode; id: number; retryOf: number | null }
  | { type: 'SAMPLE'; now: number; frames: number }
  | { type: 'PROCESS'; now: number }
  | { type: 'TICK'; now: number }
  | { type: 'RESOLVED'; result: WordResult }
  | { type: 'FAILED'; message: string; requestFailed?: boolean }
  | { type: 'FEEDBACK'; id: number; feedback: Feedback }
  | { type: 'THRESHOLD'; value: number | null }

/** Invalid transitions return the same state, including rapid duplicate clicks. */
export function assistantReducer(s: AssistantState, e: AssistantEvent): AssistantState {
  switch (e.type) {
    case 'CAMERA_START': return ['IDLE', 'ERROR'].includes(s.phase) ? { ...s, phase: 'CAMERA_STARTING', error: null } : s
    case 'CAMERA_READY': return s.phase === 'CAMERA_STARTING' ? { ...s, phase: 'READY', error: null } : s
    case 'NEXT': return ['RESULT', 'WAITING_NEXT'].includes(s.phase) ? { ...s, phase: 'READY', error: null } : s
    case 'RETRY': return ['RESULT', 'WAITING_NEXT', 'ERROR'].includes(s.phase) ? { ...s, phase: 'RETRY', error: null } : s
    case 'CAPTURE': return ['READY', 'RETRY'].includes(s.phase) ? {
      ...s, phase: 'CAPTURING', mode: e.mode, attemptId: e.id, retryOf: e.retryOf, error: null,
      result: null, frames: 0, captureStartedAt: e.now, processingStartedAt: null, captureMs: 0, processingMs: 0,
      totals: { ...s.totals, attempts: s.totals.attempts + 1, retries: s.totals.retries + (s.phase === 'RETRY' ? 1 : 0) },
    } : s
    case 'SAMPLE': return s.phase === 'CAPTURING' ? { ...s, frames: e.frames, captureMs: Math.max(0, e.now - (s.captureStartedAt ?? e.now)) } : s
    case 'PROCESS': return s.phase === 'CAPTURING' ? { ...s, phase: 'PROCESSING', processingStartedAt: e.now, captureMs: Math.max(0, e.now - (s.captureStartedAt ?? e.now)) } : s
    case 'TICK': return s.phase === 'CAPTURING' ? { ...s, captureMs: Math.max(0, e.now - (s.captureStartedAt ?? e.now)) }
      : s.phase === 'PROCESSING' ? { ...s, processingMs: Math.max(0, e.now - (s.processingStartedAt ?? e.now)) } : s
    case 'RESOLVED': return s.phase === 'PROCESSING' && e.result.id === s.attemptId ? {
      ...s, phase: 'RESULT', result: e.result, error: null, processingMs: e.result.roundTripMs,
      history: [e.result, ...s.history].slice(0, 100),
      totals: { ...s.totals, predictions: s.totals.predictions + 1,
        available: s.totals.available + (e.result.status === 'available' ? 1 : 0),
        uncertain: s.totals.uncertain + (e.result.status === 'uncertain' ? 1 : 0),
        inferenceTotalMs: s.totals.inferenceTotalMs + (e.result.inferenceMs ?? 0),
        timedPredictions: s.totals.timedPredictions + (e.result.inferenceMs !== null ? 1 : 0) },
    } : s
    case 'WAIT': return s.phase === 'RESULT' ? { ...s, phase: 'WAITING_NEXT' } : s
    case 'FAILED': return !['IDLE', 'STOPPING'].includes(s.phase) ? { ...s, phase: 'ERROR', error: e.message,
      totals: { ...s.totals, failedRequests: s.totals.failedRequests + (e.requestFailed ? 1 : 0) } } : s
    case 'STOP': return s.phase !== 'IDLE' && s.phase !== 'STOPPING' ? { ...s, phase: 'STOPPING' } : s
    case 'STOPPED': return s.phase === 'STOPPING' ? { ...s, phase: 'IDLE', error: null, frames: 0, captureStartedAt: null, processingStartedAt: null } : s
    case 'RECOVER': return s.phase === 'ERROR' ? { ...s, phase: 'READY', error: null, frames: 0 } : s
    case 'RESET': return s.phase === 'IDLE' ? initialAssistant() : s
    case 'CLEAR_HISTORY': return !['CAPTURING', 'PROCESSING', 'CAMERA_STARTING', 'STOPPING'].includes(s.phase) ? { ...s, history: [] } : s
    case 'FEEDBACK': {
      const update = (r: WordResult) => r.id === e.id ? { ...r, feedback: e.feedback } : r
      return { ...s, history: s.history.map(update), result: s.result ? update(s.result) : null }
    }
    case 'THRESHOLD': return !['CAPTURING', 'PROCESSING'].includes(s.phase) && (e.value === null || Number.isFinite(e.value) && e.value >= 0 && e.value <= 1) ? { ...s, minimumScore: e.value } : s
  }
}

export interface CapturedWordFrame { image: string; time: number; landmarks?: number[][] }
interface Dependencies {
  request: (frames: string[], signal: AbortSignal, capture: WordCapture | undefined) => Promise<WordPrediction | null>
  now?: () => number; wallTime?: () => number; timeoutMs?: number
}
/** Owns one independent clip and one request; no media is retained in history. */
export class WordAssistant {
  private state = initialAssistant()
  private listeners = new Set<() => void>()
  private frames: CapturedWordFrame[] = []
  private abort: AbortController | null = null
  private epoch = 0
  private nextId = 0
  private deadline: ReturnType<typeof setTimeout> | undefined
  private now: () => number
  private deps: Dependencies
  constructor(deps: Dependencies) { this.deps = deps; this.now = deps.now ?? (() => performance.now()) }
  getSnapshot = () => this.state
  subscribe = (listener: () => void) => { this.listeners.add(listener); return () => { this.listeners.delete(listener) } }
  send = (e: AssistantEvent) => {
    const next = assistantReducer(this.state, e)
    if (next === this.state) return false
    this.state = next; this.listeners.forEach(listener => listener()); return true
  }
  startCamera = () => this.send({ type: 'CAMERA_START' })
  cameraReady = () => this.send({ type: 'CAMERA_READY' })
  recover = () => this.send({ type: 'RECOVER' })
  begin = (mode: WordMode, retryOf: number | null = null) => {
    if (this.abort || !['READY', 'RETRY'].includes(this.state.phase)) return false
    this.frames = []
    return this.send({ type: 'CAPTURE', now: this.now(), mode, id: ++this.nextId, retryOf })
  }
  next = (mode: WordMode) => this.send({ type: 'NEXT' }) && this.begin(mode)
  retry = (mode: WordMode) => {
    const retryOf = (this.state.result?.id ?? this.state.attemptId) || null
    return this.send({ type: 'RETRY' }) && this.begin(mode, retryOf)
  }
  sample = (frame: CapturedWordFrame) => {
    if (this.state.phase !== 'CAPTURING') return false
    const limits = WORD_LIMITS[this.state.mode], previous = this.frames[this.frames.length - 1]
    // Only the publisher LRW contract imposes a 160 ms maximum gap. The
    // eight-word head interpolates ordered ROIs and has no native-FPS contract.
    if (!frame.image || !Number.isFinite(frame.time) || previous && (frame.time <= previous.time || this.state.mode === 'lrw500' && frame.time - previous.time > 160)) {
      this.fail('Capture was interrupted. Keep your mouth visible and retry this word.'); return false
    }
    if (this.frames.length >= limits.maxFrames || frame.time - (this.state.captureStartedAt ?? frame.time) > limits.maxDurationMs) {
      this.fail(`The word exceeded the capture limit. Retry a single word within ${limits.maxDurationMs / 1000} seconds.`); return false
    }
    if (this.state.mode === 'lrw500' && (!frame.landmarks || frame.landmarks.length !== 68)) {
      this.fail('Mouth landmarks are incomplete. Center your face and retry.'); return false
    }
    this.frames.push({ ...frame, landmarks: frame.landmarks?.map(p => [...p]) })
    this.send({ type: 'SAMPLE', now: this.now(), frames: this.frames.length }); return true
  }
  tick = () => {
    if (this.state.phase === 'CAPTURING' && this.now() - (this.state.captureStartedAt ?? this.now()) > WORD_LIMITS[this.state.mode].maxDurationMs) {
      this.fail('Capture timed out. Retry one complete word and click Finish Word.'); return
    }
    this.send({ type: 'TICK', now: this.now() })
  }
  fail = (message: string, requestFailed = false) => {
    this.frames = []; this.send({ type: 'FAILED', message, requestFailed })
  }
  cancel = (message: string) => {
    const requestFailed = this.state.phase === 'PROCESSING'
    this.epoch++; clearTimeout(this.deadline); this.abort?.abort(); this.abort = null
    this.fail(message, requestFailed)
  }
  finish = () => {
    if (this.state.phase !== 'CAPTURING' || this.abort) return false
    const clip = this.frames, mode = this.state.mode, limits = WORD_LIMITS[mode]
    if (clip.length < limits.minFrames) { this.fail(`Not enough mouth frames. Retry and capture at least ${limits.minFrames} frames.`); return false }
    const duration = clip[clip.length - 1].time - clip[0].time
    if (mode === 'lrw500' && (duration < 320 || duration > 3000 || (clip.length - 1) * 1000 / duration < 10)) {
      this.fail('Capture one word lasting 0.32–3 seconds with steady video, then retry.'); return false
    }
    const captured = this.state, token = ++this.epoch, controller = new AbortController(), began = this.now()
    this.abort = controller; this.frames = []
    this.send({ type: 'PROCESS', now: began })
    const metadata: WordCapture | undefined = mode === 'lrw500' ? { input_kind: 'webcam', convention: 'lrw-meanface-mediapipe68-v1',
      landmarks: clip.map(f => f.landmarks!), timestamps_ms: clip.map(f => f.time), face_counts: clip.map(() => 1), final: true } : undefined
    // A separate deadline also covers a noncooperative transport/body parser.
    this.deadline = setTimeout(() => {
      if (token !== this.epoch || this.state.phase !== 'PROCESSING') return
      this.epoch++; controller.abort(); this.abort = null
      this.fail('Analysis timed out. Check the backend and retry your word.', true)
    }, this.deps.timeoutMs ?? 30000)
    void Promise.resolve().then(() => controller.signal.aborted ? null : this.deps.request(clip.map(f => f.image), controller.signal, metadata)).then(response => {
      if (token !== this.epoch || controller.signal.aborted) return
      if (!response) throw new Error('No prediction was returned. Check the backend and retry.')
      const confidence = Number.isFinite(response.confidence) ? response.confidence : null
      const uncertain = response.uncertain || (captured.minimumScore !== null && confidence !== null && confidence < captured.minimumScore)
      const consensus = response.diagnostics?.temporal_consensus
      const rejected = uncertain || (mode === 'lrw500' && consensus?.confirmed !== true)
      const result: WordResult = { id: captured.attemptId, mode, candidate: response.predicted_word,
        word: rejected ? null : response.predicted_word, confidence,
        inferenceMs: Number.isFinite(response.latency_ms) && response.latency_ms! >= 0 ? response.latency_ms! : null,
        roundTripMs: Math.max(0, this.now() - began), at: (this.deps.wallTime ?? Date.now)(),
        status: rejected ? 'uncertain' : 'available', feedback: null,
        captureMs: Math.max(0, began - (captured.captureStartedAt ?? began)), frameCount: clip.length,
        retryOf: captured.retryOf, candidates: response.top_predictions.slice(0, 5), performance: response.performance }
      this.send({ type: 'RESOLVED', result }); this.send({ type: 'WAIT' })
    }).catch(error => {
      if (token === this.epoch && !controller.signal.aborted) this.fail(error instanceof Error ? error.message : 'Analysis failed. Retry your word.', true)
    }).finally(() => {
      if (token !== this.epoch) return
      clearTimeout(this.deadline); this.deadline = undefined; this.abort = null
    })
    return true
  }
  stop = () => {
    this.epoch++; clearTimeout(this.deadline); this.deadline = undefined
    this.abort?.abort(); this.abort = null; this.frames = []
    this.send({ type: 'STOP' }); this.send({ type: 'STOPPED' })
  }
  reset = () => { this.stop(); this.send({ type: 'RESET' }) }
  dispose = () => { this.stop(); this.listeners.clear() }
}
