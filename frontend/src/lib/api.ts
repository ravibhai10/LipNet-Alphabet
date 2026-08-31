/**
 * Backend client.
 *
 * The frontend never fabricates model output. It probes the optional
 * inference backend and, when reachable, consumes these endpoints:
 *
 *   GET  /health            -> { status: string, model_loaded?: boolean }
 *   POST /api/predict       -> body: { frames: string[] }   (base64 JPEG mouth crops, temporal order)
 *                              resp: {
 *                                letter: string,            // "A".."Z" | "ABSTAIN"
 *                                confidence: number,        // 0..1
 *                                top3: { letter: string, prob: number }[],
 *                                gates: { appearance: number, geometry: number, motion: number },
 *                                uncertainty?: { margin?: number, entropy?: number, state?: string },
 *                                temporal?: { frames: number }
 *                              }
 *   GET  /api/dataset/stats -> { recordings?: number, speakers?: number, letters?: number }
 *
 * If none of these are reachable the UI shows "Model connection unavailable"
 * instead of invented values.
 */

const BASE: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000'

export type BackendStatus = 'idle' | 'checking' | 'online' | 'offline'

export interface HealthInfo {
  status: BackendStatus
  modelLoaded?: boolean
}

export interface Prediction {
  letter: string
  confidence: number
  top3: { letter: string; prob: number }[]
  gates: { appearance: number; geometry: number; motion: number }
  uncertainty?: { margin?: number; entropy?: number; state?: string }
  temporal?: { frames: number }
}

export async function fetchWithTimeout(url: string, init: RequestInit = {}, ms = 2500): Promise<Response> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), ms)
  try {
    return await fetch(url, { ...init, signal: ctrl.signal })
  } finally {
    clearTimeout(timer)
  }
}

export async function probeHealth(): Promise<HealthInfo> {
  for (const path of ['/health', '/api/health']) {
    try {
      const res = await fetchWithTimeout(`${BASE}${path}`)
      if (res.ok) {
        const data = (await res.json().catch(() => ({}))) as {
          status?: string
          model_loaded?: boolean
        }
        return { status: 'online', modelLoaded: data.model_loaded }
      }
    } catch {
      // try next path
    }
  }
  return { status: 'offline' }
}

export async function fetchDatasetStats(): Promise<Record<string, number> | null> {
  try {
    const res = await fetchWithTimeout(`${BASE}/api/dataset/stats`, {}, 2000)
    if (!res.ok) return null
    const data = (await res.json()) as Record<string, unknown>
    const out: Record<string, number> = {}
    for (const [k, v] of Object.entries(data)) {
      if (typeof v === 'number') out[k] = v
    }
    return Object.keys(out).length ? out : null
  } catch {
    return null
  }
}

export async function requestPrediction(frames: string[]): Promise<Prediction | null> {
  try {
    const res = await fetchWithTimeout(
      `${BASE}/api/predict`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frames }),
      },
      6000,
    )
    if (!res.ok) return null
    const data = (await res.json()) as Partial<Prediction>
    if (!data || typeof data.letter !== 'string') return null
    return {
      letter: data.letter,
      confidence: typeof data.confidence === 'number' ? data.confidence : 0,
      top3: Array.isArray(data.top3) ? data.top3 : [],
      gates: data.gates ?? { appearance: 0, geometry: 0, motion: 0 },
      uncertainty: data.uncertainty,
      temporal: data.temporal,
    }
  } catch {
    return null
  }
}

export const API_BASE = BASE
