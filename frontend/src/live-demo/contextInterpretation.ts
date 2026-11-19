import { API_BASE } from '../lib/api.ts'

export interface ContextReply {
  event_id: string; visual_candidate: string; original_word: string | null; safe_prediction: string | null
  visual_confidence: number; suggested_word: string | null; suggested_visual_score: number | null
  applied: false; requires_confirmation: boolean; status: string; latency_ms: number; cached: boolean
}
export interface ContextEvent {
  mode: string; event_id: string; visual_candidate: string; original_word: string | null
  visual_uncertain: boolean; candidates: { word: string; prob: number }[]
  context: string; history: string[]; consent_text: boolean
}

export function validateContextReply(data: unknown, event: ContextEvent): ContextReply {
  const d = data as ContextReply
  if (!d || d.event_id !== event.event_id || d.visual_candidate !== event.visual_candidate ||
      d.original_word !== event.original_word || d.safe_prediction !== event.original_word || d.applied !== false ||
      typeof d.status !== 'string' || typeof d.requires_confirmation !== 'boolean' || typeof d.cached !== 'boolean' ||
      d.visual_confidence !== event.candidates[0]?.prob ||
      (d.suggested_word === null ? d.suggested_visual_score !== null || d.requires_confirmation :
        d.suggested_visual_score !== event.candidates.find(c => c.word === d.suggested_word)?.prob ||
        d.requires_confirmation !== (d.suggested_word !== event.original_word)) ||
      !Number.isFinite(d.latency_ms) || d.latency_ms < 0 ||
      (d.suggested_word !== null && !event.candidates.some(c => c.word === d.suggested_word))) {
    throw new Error('Invalid contextual suggestion; visual result retained.')
  }
  return d
}

/** One latest request, bounded deadline; a provider cannot delay visual inference. */
export class ContextClient {
  private epoch = 0
  private controller: AbortController | null = null
  private sessionToken: string | null = null
  cancel() { this.epoch++; this.controller?.abort(); this.controller = null }
  private headers(key: string, json = false) {
    return { ...(json ? { 'Content-Type': 'application/json' } : {}), Authorization: `Bearer ${key}`,
      ...(this.sessionToken ? { 'X-Interpretation-Session': this.sessionToken } : {}) }
  }
  async createSession(key: string): Promise<void> {
    const response = await fetch(`${API_BASE}/api/context/sessions`, { method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` }, body: JSON.stringify({}) })
    if (!response.ok) throw new Error('Could not create an interpretation session.')
    const data = await response.json() as { session_token?: unknown }
    if (typeof data.session_token !== 'string' || data.session_token.length < 32) throw new Error('Invalid interpretation session response.')
    this.sessionToken = data.session_token
  }
  async grantConsent(key: string): Promise<void> {
    if (!this.sessionToken) throw new Error('Create an interpretation session first.')
    const response = await fetch(`${API_BASE}/api/context/sessions/consent`, { method: 'POST',
      headers: this.headers(key, true), body: JSON.stringify({ purpose: 'text_only_llm_interpretation', notice_version: 'llm-text-v1', granted: true }) })
    if (!response.ok) throw new Error('Could not record interpretation consent.')
  }
  async revokeConsent(key: string): Promise<void> {
    if (!this.sessionToken) return
    const response = await fetch(`${API_BASE}/api/context/sessions/consent/revoke`, { method: 'POST', headers: this.headers(key) })
    if (!response.ok && response.status !== 401) throw new Error('Could not revoke interpretation consent.')
  }
  async terminate(key: string): Promise<void> {
    if (!this.sessionToken) return
    const token = this.sessionToken
    this.cancel()
    const response = await fetch(`${API_BASE}/api/context/sessions/terminate`, { method: 'POST',
      headers: { Authorization: `Bearer ${key}`, 'X-Interpretation-Session': token } })
    if (!response.ok) throw new Error('Could not terminate interpretation session; retry is available.')
    if (this.sessionToken === token) this.sessionToken = null
  }
  async interpret(event: ContextEvent, key: string): Promise<ContextReply | null> {
    if (!this.sessionToken) throw new Error('An active, consented interpretation session is required.')
    this.cancel()
    const epoch = this.epoch, controller = new AbortController()
    this.controller = controller
    let rejectDeadline: (reason: Error) => void = () => {}
    const deadline = new Promise<never>((_resolve, reject) => { rejectDeadline = reject })
    const timer = setTimeout(() => { controller.abort(); rejectDeadline(new Error('Interpretation timed out; visual result retained.')) }, 2200)
    const operation = async () => {
      const response = await fetch(`${API_BASE}/api/context/interpret`, { method: 'POST', signal: controller.signal,
        headers: { ...this.headers(key, true) }, body: JSON.stringify(event) })
      if (!response.ok) throw new Error(response.status === 401 || response.status === 503
        ? 'Context service unavailable or private access is not configured. Visual result retained.' : 'Interpretation unavailable; visual result retained.')
      return validateContextReply(await response.json(), event)
    }
    try {
      const reply = await Promise.race([operation(), deadline])
      return epoch === this.epoch ? reply : null
    } catch (error) {
      if (epoch !== this.epoch) return null
      throw error
    } finally { clearTimeout(timer); if (epoch === this.epoch) this.controller = null }
  }
}

/** Human feedback is audit-only; never feeds back into model training. */
export async function recordInterpretation(mode: string, reply: ContextReply, accepted: boolean, key: string) {
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 1000)
  try {
    const response = await fetch(`${API_BASE}/api/context/feedback`, { method: 'POST', signal: controller.signal,
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
      body: JSON.stringify({mode, event_id:reply.event_id, word:reply.suggested_word, accepted}) })
    if (!response.ok) throw new Error('Could not record interpretation feedback.')
  } finally { clearTimeout(timer) }
}
