import { API_BASE, fetchWithTimeout, responseError } from '../lib/api'
export interface DatasetValidation { ok: boolean; checks: { name: string; pass: boolean; detail?: string }[] }
export interface TrainingCheckpoint {
  name: string; path: string; created?: string; epoch?: number; best_val?: number;
  config_hash?: string; repr_version?: string; vocab?: string; size_bytes?: number; model_kind?: string
}
export interface InferenceCheckpoint {
  name: string; path: string; compatible: boolean; note?: string; loaded?: boolean; model_kind?: string
}
export interface TrainingJobStatus {
  job_id: string; state: 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED' | 'UNKNOWN'
  epoch?: number; total_epochs?: number; train_loss?: number; val_loss?: number; best_val?: number; message?: string
}
export interface TrainingPrecheck {
  can_start: boolean; reasons: string[]; warnings: string[]
  dataset_summary?: { recordings?: number; speakers?: number; letters?: number }
  config?: Record<string, unknown>
}
export interface LiveState {
  running: boolean; model_loaded: boolean; buffer_size?: number; last_letter?: string
  last_confidence?: number; last_top3?: { letter: string; prob: number }[]; state?: string; temporal_frames?: number
}
export interface CamerasProbe { cameras: { index: number; name?: string; available: boolean | null }[] }
export interface DatasetOverview {
  total_recordings: number; total_speakers: number; total_sessions: number
  letters: { letter: string; total_recordings: number }[]
}
export interface DatasetRecording { path: string; speaker_id: string; session_id: string; recording_id: string; duration_ms: number }
export interface ExperimentRun {
  name: string; path: string; epochs: number; best_val_accuracy?: number; test_accuracy?: number; device?: string
  history: { epoch: number; train_loss?: number; val_loss?: number; val_accuracy?: number }[]
}
let lastError: string | null = null
export function getWorkspaceError(): string | null { return lastError }
async function json<T>(path: string, init: RequestInit = {}, ms = 10000): Promise<T | null> {
  lastError = null
  try {
    const response = await fetchWithTimeout(`${API_BASE}${path}`, init, ms)
    if (!response.ok) throw await responseError(response)
    return await response.json() as T
  } catch (error) {
    lastError = error instanceof Error
      ? error.name === 'AbortError' ? 'The request timed out. Please try again.' : error.message
      : 'Cannot reach the local backend.'
    return null
  }
}
export function datasetOverview() { return json<DatasetOverview>('/api/overview') }
export function datasetRecordings(letter: string) { return json<{ letter: string; recordings: DatasetRecording[] }>(`/api/recordings/${encodeURIComponent(letter)}`) }
export async function validateDataset(): Promise<DatasetValidation | null> {
  const data = await json<{ passed: boolean; results?: { check: string; passed: boolean; message?: string; details?: unknown }[] }>('/api/dataset/validate', {}, 30000)
  return data ? { ok: data.passed === true, checks: (data.results ?? []).map(item => ({ name: item.check, pass: item.passed, detail: item.message })) } : null
}
export async function buildManifest(): Promise<{ ok: boolean; note?: string; entries?: number } | null> {
  const data = await json<{ entries: number }>('/api/dataset/build-manifest', { method: 'POST' }, 30000)
  return data ? { ok: Number.isFinite(data.entries), entries: data.entries, note: `${data.entries} recordings indexed.` } : null
}
export async function createSplits(): Promise<{ ok: boolean; note?: string; counts?: Record<string, number> } | null> {
  const data = await json<Record<string, number>>('/api/dataset/create-splits', { method: 'POST' }, 30000)
  return data ? { ok: true, counts: data, note: Object.entries(data).map(([name, count]) => `${name}: ${count}`).join(' · ') } : null
}
export async function trainingPrecheck(): Promise<TrainingPrecheck | null> {
  const data = await json<{ passed: boolean; errors?: string[]; warnings?: string[]; recordings?: number; speakers?: number; letters?: number; config?: Record<string, unknown> }>('/api/training/check', {}, 30000)
  return data ? {
    can_start: data.passed === true, reasons: data.errors ?? [], warnings: data.warnings ?? [],
    dataset_summary: { recordings: data.recordings, speakers: data.speakers, letters: data.letters }, config: data.config,
  } : null
}
export function startTraining(config?: { epochs?: number; batch_size?: number; lr?: number; resume?: string }): Promise<{ job_id: string } | null> {
  return json('/api/training/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(config ?? {}) }, 30000)
}
export async function trainingStatus(jobId: string): Promise<TrainingJobStatus | null> {
  const data = await json<{ job_id: string; state: TrainingJobStatus['state']; progress?: Record<string, unknown>; error?: string }>(`/api/training/status/${encodeURIComponent(jobId)}`)
  if (!data) return null
  const progress = data.progress ?? {}
  const number = (key: string) => typeof progress[key] === 'number' ? progress[key] as number : undefined
  return {
    job_id: data.job_id, state: data.state, epoch: number('epoch'), total_epochs: number('total_epochs'),
    train_loss: number('train_loss'), val_loss: number('val_loss'), best_val: number('best_val'),
    message: data.error ?? (typeof progress.note === 'string' ? progress.note : undefined),
  }
}
export async function cancelTraining(jobId: string): Promise<{ ok: boolean; note?: string } | null> {
  const data = await json<{ cancelled: boolean }>(`/api/training/cancel/${encodeURIComponent(jobId)}`, { method: 'POST' })
  return data ? { ok: data.cancelled === true } : { ok: false, note: lastError ?? 'Could not cancel this job.' }
}
export async function trainingCheckpoints(): Promise<{ checkpoints: TrainingCheckpoint[] } | null> {
  const data = await json<TrainingCheckpoint[] | { checkpoints: TrainingCheckpoint[] }>('/api/training/checkpoints')
  return data ? { checkpoints: Array.isArray(data) ? data : data.checkpoints ?? [] } : null
}
export function inferenceCheckpoints() { return json<{ checkpoints: InferenceCheckpoint[]; model_loaded?: boolean }>('/api/inference/checkpoints') }
export async function liveState(): Promise<LiveState | null> {
  const data = await json<{ state: string; running?: boolean; model_loaded?: boolean; buffer_size?: number; letter?: string; confidence?: number; top_k?: [string, number][]; frames_inferred?: number }>('/api/live/state')
  return data ? {
    state: data.state, running: data.running === true, model_loaded: data.model_loaded === true,
    buffer_size: data.buffer_size, last_letter: data.letter, last_confidence: data.confidence,
    last_top3: data.top_k?.map(([letter, prob]) => ({ letter, prob })), temporal_frames: data.frames_inferred,
  } : null
}
export function camerasProbe() { return json<CamerasProbe>('/api/live/cameras') }
export function experimentSummaries() { return json<{ runs: ExperimentRun[] }>('/api/experiments') }
export interface EvaluationMetrics {
  accuracy?: number; macro_f1?: number; top3?: number; top5?: number; cer?: number; ece?: number; evaluated_on?: string; evaluated_at?: string
}
export function readEvaluation(sidecarPath: string) { return json<EvaluationMetrics>(`/api/evaluation/${encodeURIComponent(sidecarPath)}`) }
