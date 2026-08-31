/**
 * Workspace API client — extends the existing lib/api.ts.
 *
 * Reuses the existing endpoints in app/main.py:
 *   /api/dataset/validate, /api/dataset/build-manifest,
 *   /api/dataset/create-splits, /api/dataset/stats
 *   /api/training/check, /api/training/start, /api/training/status/{id},
 *   /api/training/cancel/{id}, /api/training/checkpoints
 *   /api/inference/checkpoints
 *   /api/live/state, /api/live/cameras
 *
 * All requests are time-bounded. On failure, returns null (never fabricates).
 */
import { API_BASE, fetchWithTimeout } from "../lib/api"

export interface DatasetValidation {
  ok: boolean
  checks: { name: string; pass: boolean; detail?: string }[]
}

export interface TrainingCheckpoint {
  name: string
  path: string
  created: string
  epoch?: number
  best_val?: number
  config_hash?: string
  repr_version?: string
  vocab?: string
}

export interface InferenceCheckpoint {
  name: string
  path: string
  compatible: boolean
  note?: string
}

export interface TrainingJobStatus {
  job_id: string
  state: "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED" | "UNKNOWN"
  epoch?: number
  total_epochs?: number
  train_loss?: number
  val_loss?: number
  best_val?: number
  message?: string
}

export interface TrainingPrecheck {
  can_start: boolean
  reasons: string[]
  dataset_summary?: { recordings?: number; speakers?: number; letters?: number }
  config?: Record<string, unknown>
}

export interface LiveState {
  running: boolean
  model_loaded: boolean
  buffer_size?: number
  last_letter?: string
  last_confidence?: number
  last_top3?: { letter: string; prob: number }[]
  state?: string
  temporal_frames?: number
}

export interface CamerasProbe {
  cameras: { index: number; name?: string; available: boolean }[]
}

async function getJson<T>(path: string, ms = 2500): Promise<T | null> {
  try {
    const res = await fetchWithTimeout(`${API_BASE}${path}`, {}, ms)
    if (!res.ok) return null
    return (await res.json()) as T
  } catch {
    return null
  }
}

export async function validateDataset(): Promise<DatasetValidation | null> {
  return getJson<DatasetValidation>("/api/dataset/validate")
}

export async function buildManifest(): Promise<{ ok: boolean; note?: string } | null> {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/api/dataset/build-manifest`, { method: "POST" }, 30000)
    if (!res.ok) return null
    return (await res.json()) as { ok: boolean; note?: string }
  } catch {
    return null
  }
}

export async function createSplits(): Promise<{ ok: boolean; note?: string } | null> {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/api/dataset/create-splits`, { method: "POST" }, 10000)
    if (!res.ok) return null
    return (await res.json()) as { ok: boolean; note?: string }
  } catch {
    return null
  }
}

export async function trainingPrecheck(): Promise<TrainingPrecheck | null> {
  return getJson<TrainingPrecheck>("/api/training/check")
}

export async function startTraining(config?: { epochs?: number; batch_size?: number; lr?: number; resume?: string }): Promise<{ job_id: string } | null> {
  try {
    const res = await fetchWithTimeout(
      `${API_BASE}/api/training/start`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(config ?? {}) },
      5000
    )
    if (!res.ok) return null
    return (await res.json()) as { job_id: string }
  } catch {
    return null
  }
}

export async function trainingStatus(jobId: string): Promise<TrainingJobStatus | null> {
  return getJson<TrainingJobStatus>(`/api/training/status/${encodeURIComponent(jobId)}`)
}

export async function cancelTraining(jobId: string): Promise<{ ok: boolean } | null> {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/api/training/cancel/${encodeURIComponent(jobId)}`, {}, 3000)
    if (!res.ok) return null
    return (await res.json()) as { ok: boolean }
  } catch {
    return null
  }
}

export async function trainingCheckpoints(): Promise<{ checkpoints: TrainingCheckpoint[] } | null> {
  return getJson<{ checkpoints: TrainingCheckpoint[] }>("/api/training/checkpoints")
}

export async function inferenceCheckpoints(): Promise<{ checkpoints: InferenceCheckpoint[]; model_loaded?: boolean } | null> {
  return getJson<{ checkpoints: InferenceCheckpoint[]; model_loaded?: boolean }>("/api/inference/checkpoints")
}

export async function liveState(): Promise<LiveState | null> {
  return getJson<LiveState>("/api/live/state")
}

export async function camerasProbe(): Promise<CamerasProbe | null> {
  return getJson<CamerasProbe>("/api/live/cameras")
}

export interface EvaluationMetrics {
  accuracy?: number
  macro_f1?: number
  top3?: number
  top5?: number
  cer?: number
  ece?: number
  evaluated_on?: string
  evaluated_at?: string
}

/** Evaluation is read from a sidecar file the backend writes next to a checkpoint. */
export async function readEvaluation(sidecarPath: string): Promise<EvaluationMetrics | null> {
  // The backend serves it under /api/training/checkpoints/{name}/eval — fall back to a generic
  // /api/evaluation/{name} if available, otherwise return null (NEVER fabricate).
  return getJson<EvaluationMetrics>(`/api/evaluation/${encodeURIComponent(sidecarPath)}`)
}
