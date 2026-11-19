import { useCallback, useEffect, useRef, useState } from 'react'
import type { WordResult, WordMode } from '../../live-demo/wordAssistant'
import { ContextClient, recordInterpretation, type ContextReply } from '../../live-demo/contextInterpretation'

interface Props { result: WordResult | null; history: WordResult[]; mode: WordMode; vocabulary: string[]; active: boolean }

export function ContextInterpretation({ result, history, mode, vocabulary, active }: Props) {
  const [enabled, setEnabled] = useState(false), [consent, setConsent] = useState(false)
  const [context, setContext] = useState(''), [key, setKey] = useState('')
  const [reply, setReply] = useState<ContextReply | null>(null), [message, setMessage] = useState('')
  const [accepted, setAccepted] = useState<string | null>(null), [busy, setBusy] = useState(false)
  const [terminationPending, setTerminationPending] = useState(false)
  const client = useRef(new ContextClient()), generation = useRef(0)
  const terminateSession = async () => {
    try { await client.current.terminate(key); setTerminationPending(false) }
    catch {
      setConsent(false); setTerminationPending(true)
      setMessage('Session termination failed. Retry termination before starting another interpretation session.')
    }
  }
  const setTextConsent = async (checked: boolean) => {
    if (!checked) {
      invalidate()
      try { await client.current.revokeConsent(key) }
      catch { setMessage('Consent revocation could not be confirmed; terminating this session.') }
      finally { setConsent(false); await terminateSession() }
      return
    }
    if (!key) { setMessage('Enter private access before granting text-sharing consent.'); return }
    setBusy(true); setMessage('Creating a private interpretation session…')
    try {
      await client.current.createSession(key)
      await client.current.grantConsent(key)
      setConsent(true); setMessage('Session and text-only consent are active.')
    } catch (error) {
      await terminateSession()
      setConsent(false); setMessage(error instanceof Error ? error.message : 'Could not start interpretation session.')
    } finally { setBusy(false) }
  }
  const run = useCallback(async () => {
    if (!enabled || !active || !result || mode === 'personal') return
    if (!consent || !context.trim() || !key) { setMessage('Enter context and private access, and consent to text sharing.'); return }
    const token = ++generation.current
    setBusy(true); setReply(null); setAccepted(null); setMessage('Requesting text-only interpretation…')
    try {
      const response = await client.current.interpret({ mode, event_id: `${mode}_${result.id}`,
        visual_candidate: result.candidate, original_word: result.word, visual_uncertain: !result.word,
        candidates: result.candidates.slice(0, 5), context,
        history: history.filter(r => r.id !== result.id && r.mode === mode && r.word && vocabulary.includes(r.word)).slice(0, 6).reverse().map(r => r.word!),
        consent_text: consent }, key)
      if (token !== generation.current) return
      setReply(response); setMessage(response ? response.status.replace(/_/g, ' ') : 'Superseded request discarded.')
    } catch (error) {
      if (token === generation.current) setMessage(error instanceof Error ? error.message : 'Interpretation unavailable; visual result retained.')
    } finally { if (token === generation.current) setBusy(false) }
  }, [enabled, active, result, mode, consent, context, key, history, vocabulary])
  const runRef = useRef(run); runRef.current = run
  const invalidate = () => { generation.current++; client.current.cancel(); setBusy(false); setReply(null); setAccepted(null); setMessage('') }
  const record = (value: ContextReply, choice: boolean) => {
    const token = generation.current
    void recordInterpretation(mode, value, choice, key).catch(() => {
      if (token === generation.current) setMessage('Interpretation selected locally; feedback log unavailable.')
    })
  }
  useEffect(() => {
    generation.current++; client.current.cancel(); setReply(null); setAccepted(null); setBusy(false); setMessage('')
    if (enabled && active && result) void runRef.current()
    return () => { generation.current++; client.current.cancel() }
  }, [result?.id, mode, enabled, active]) // Completed capture only; stop/next also invalidate pending work.
  useEffect(() => () => { void client.current.terminate(key) }, [key])
  useEffect(() => {
    if (!active) {
      invalidate(); void terminateSession(); setConsent(false)
    }
  }, [active])

  return <section className="ws-assistant-feedback-box" aria-label="Contextual interpretation">
    <label><input type="checkbox" checked={enabled} onChange={e => {
      const next = e.target.checked; setEnabled(next)
      if (!next) { invalidate(); void terminateSession(); setConsent(false) }
    }} /> Enable optional LLM interpretation</label>
     {terminationPending && <button type="button" className="ws-btn ws-btn-sm" onClick={() => void terminateSession()}>Retry session termination</button>}
    {enabled && <>
      <p>Visual recognition stays unchanged. Suggestions need your confirmation; accuracy benefits are unverified.</p>
      <label>Application context <input maxLength={240} value={context} onChange={e => { invalidate(); setContext(e.target.value) }} placeholder="For example: navigating a menu" /></label>
      <label>Private access credential <input type="password" autoComplete="off" value={key} onChange={e => { invalidate(); setKey(e.target.value) }} /></label>
      <label><input type="checkbox" checked={consent} disabled={busy || !key || terminationPending} onChange={e => void setTextConsent(e.target.checked)} /> Share candidate labels, scores, recent words and this text with OpenAI. No video. Consent is stored for this server-managed session.</label>
      <button type="button" className="ws-btn ws-btn-sm" disabled={!active || !result || busy || !consent || !context.trim() || !key || mode === 'personal'} onClick={() => void run()}>Interpret this result</button>
      <p role="status">{message}</p>
      {result && <p>Visual: <strong>{result.word ?? 'Uncertain'}</strong> · candidate {result.candidate} · score {result.confidence === null ? 'unavailable' : `${(result.confidence * 100).toFixed(1)}%`}</p>}
      {reply?.suggested_word && <>
        <p>LLM suggestion: <strong>{reply.suggested_word}</strong> · visual score {((reply.suggested_visual_score ?? 0) * 100).toFixed(1)}% · {reply.latency_ms.toFixed(0)} ms {reply.cached ? '(cached)' : ''}</p>
        <button type="button" className="ws-btn ws-btn-sm" onClick={() => { setAccepted(reply.suggested_word); record(reply, true) }}>Use as interpretation</button>
        <button type="button" className="ws-btn ws-btn-sm" onClick={() => { setAccepted(null); record(reply, false); setReply(null) }}>Reject suggestion</button>
      </>}
      {accepted && <p>User-confirmed interpretation: <strong>{accepted}</strong>. This is not a changed visual prediction.</p>}
    </>}
  </section>
}
