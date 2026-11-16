import type { AssistantState, WordAssistant as Controller } from '../../live-demo/wordAssistant'
import { WORD_LIMITS } from '../../live-demo/wordAssistant'
import { Icon } from '../Icon'

interface Props {
  state: AssistantState; controller: Controller; canCapture: boolean; cameraActive: boolean
  begin: (action?: 'start' | 'next' | 'retry') => void
}
const scores = (value: number | null) => value === null ? 'Confidence unavailable' : `${(value * 100).toFixed(1)}%`
const duration = (value: number | null) => value === null ? 'Unavailable' : `${Math.round(value)} ms`
const prompts: Record<AssistantState['phase'], string> = {
  IDLE: 'Start your camera to prepare a word.', CAMERA_STARTING: 'Starting camera and landmark tracking…',
  READY: 'Ready. Click Start Word, say one word, then click Finish Word.', CAPTURING: 'Capturing your word…',
  PROCESSING: 'Analyzing lip movements…', RESULT: 'Prediction complete.', WAITING_NEXT: 'Prediction complete. Ready for your next word.',
  RETRY: 'Preparing a fresh capture…', ERROR: 'This attempt needs attention.', STOPPING: 'Stopping the camera…',
}

export function WordCaptureStatus({ state }: Pick<Props, 'state'>) {
  const limits = WORD_LIMITS[state.mode]
  return <div className={`ws-assistant-capture state-${state.phase.toLowerCase()}`}>
    <div aria-live="polite" aria-atomic="true"><span className="ws-assistant-indicator"/><strong>{prompts[state.phase]}</strong></div>
    {state.phase === 'CAPTURING' && <>
      <div className="ws-assistant-clock"><span>{(state.captureMs / 1000).toFixed(1)} s captured</span><span>{state.frames} frames · minimum {limits.minFrames}</span></div>
      <progress value={state.frames} max={limits.maxFrames} aria-label="Capture buffer used"/>
      <p>Click Finish Word when you finish speaking. Limit: {limits.maxDurationMs / 1000} s. The timer does not detect word boundaries.</p>
    </>}
    {state.phase === 'PROCESSING' && <p className="ws-assistant-processing"><span className="ws-assistant-spinner" aria-hidden="true"/>{(state.processingMs / 1000).toFixed(1)} s processing · waiting for the backend</p>}
  </div>
}

export function WordAssistantPanel({ state, controller, canCapture, cameraActive, begin }: Props) {
  const result = state.result, busy = ['CAPTURING', 'PROCESSING', 'CAMERA_STARTING', 'STOPPING'].includes(state.phase)
  return <div className="ws-assistant" aria-busy={state.phase === 'PROCESSING'}>
    <div className={`ws-assistant-result ${result?.status === 'uncertain' ? 'is-uncertain' : result ? 'is-available' : ''}`}>
      <span className="ws-eyebrow">{state.phase === 'PROCESSING' ? 'ANALYZING LIP MOVEMENTS' : result ? 'LAST PREDICTION' : 'ONE WORD AT A TIME'}</span>
      <p className="ws-assistant-word" aria-live="polite">{result ? result.word ?? 'Uncertain' : '—'}</p>
      <p>{result?.status === 'uncertain' ? 'Prediction uncertain. Please try this word again.' : result ? 'Prediction available · awaiting your next action.' : prompts[state.phase]}</p>
      {result?.status === 'uncertain' && <p className="ws-assistant-candidate">Possible match: <strong>{result.candidate}</strong></p>}
      {state.phase === 'PROCESSING' && <div className="ws-assistant-processing" role="status"><span className="ws-assistant-spinner" aria-hidden="true"/>{(state.processingMs / 1000).toFixed(1)} s elapsed</div>}
    </div>
    <dl className="ws-assistant-result-data">
      <div><dt>Model confidence</dt><dd>{result ? scores(result.confidence) : 'No result yet'}</dd></div>
      <div><dt>Server inference</dt><dd>{result ? duration(result.inferenceMs) : 'No measurement yet'}</dd></div>
      <div><dt>Request round trip</dt><dd>{result ? duration(result.roundTripMs) : 'No measurement yet'}</dd></div>
    </dl>
    <div className="ws-assistant-actions">
      {state.phase === 'READY' && <button className="ws-btn ws-btn-primary" disabled={!canCapture} onClick={() => begin('start')}><Icon name="camera" size={14}/>Start Word</button>}
      {state.phase === 'CAPTURING' && <button className="ws-btn ws-btn-primary" disabled={state.frames < WORD_LIMITS[state.mode].minFrames} onClick={controller.finish}><Icon name="stop" size={14}/>Finish Word</button>}
      {state.phase === 'PROCESSING' && <button className="ws-btn ws-btn-primary" disabled><span className="ws-assistant-spinner" aria-hidden="true"/>Analyzing…</button>}
      {['RESULT', 'WAITING_NEXT'].includes(state.phase) && <>
        <button className="ws-btn ws-btn-primary" disabled={!canCapture} onClick={() => begin('next')}>{result?.status === 'uncertain' ? 'Continue · Next Word' : 'Next Word'}<Icon name="arrow" size={14}/></button>
        <button className="ws-btn ws-btn-ghost" disabled={!canCapture} onClick={() => begin('retry')}><Icon name="refresh" size={14}/>Retry Word</button>
      </>}
      {state.phase === 'ERROR' && cameraActive && <>
        <button className="ws-btn ws-btn-primary" disabled={!canCapture} onClick={() => begin('retry')}>Retry Word</button>
        <button className="ws-btn ws-btn-ghost" onClick={controller.recover}>Back to Ready</button>
      </>}
    </div>
    {cameraActive && !canCapture && !busy && <p className="ws-assistant-help">Position one face inside the frame, keep your mouth visible, and check model readiness.</p>}
    {!cameraActive && result && <p className="ws-assistant-help">Camera stopped. Your result and history are retained. Start the camera to continue.</p>}
    {state.phase === 'WAITING_NEXT' && cameraActive && <p className="ws-assistant-waiting" role="status">Ready for your next word. Capture stays paused until you choose Next Word or Retry Word.</p>}
    <p className="ws-assistant-help">Model scores are uncalibrated. They are not a probability that the word is correct.</p>
    <details className="ws-assistant-settings"><summary>Optional score filter</summary>
      <label><input type="checkbox" disabled={busy} checked={state.minimumScore !== null} onChange={e => controller.send({ type: 'THRESHOLD', value: e.target.checked ? .8 : null })}/>Apply an extra minimum model score</label>
      {state.minimumScore !== null && <label>Minimum score: {(state.minimumScore * 100).toFixed(0)}%<input aria-label="Minimum model score" disabled={busy} type="range" min="0" max="1" step="0.01" value={state.minimumScore} onChange={e => controller.send({ type: 'THRESHOLD', value: Number(e.target.value) })}/></label>}
      <p>This optional filter can reject additional results; it never overrides the backend’s rejection decision.</p>
    </details>
    {!!result?.candidates.length && <details className="ws-assistant-settings"><summary>Returned candidates</summary><ul>{result.candidates.map(c => <li key={c.word}><span>{c.word}</span><span>{(c.prob * 100).toFixed(1)}%</span></li>)}</ul></details>}
  </div>
}

export function WordSession({ state, controller }: Pick<Props, 'state' | 'controller'>) {
  const { totals } = state
  const busy = ['CAPTURING', 'PROCESSING', 'CAMERA_STARTING', 'STOPPING'].includes(state.phase)
  const metrics = [['Words attempted', totals.attempts], ['Predictions generated', totals.predictions], ['Available predictions', totals.available],
    ['Uncertain predictions', totals.uncertain], ['Failed requests', totals.failedRequests], ['Retries', totals.retries],
    ['Average server inference', totals.timedPredictions ? duration(totals.inferenceTotalMs / totals.timedPredictions) : 'Not measured']]
  return <section className="ws-card ws-assistant-session" aria-label="Word session summary">
    <div className="ws-card-head"><h2>Session summary & history</h2><button className="ws-inline-link" disabled={!state.history.length || busy} onClick={() => controller.send({ type: 'CLEAR_HISTORY' })}>Clear History</button></div>
    <dl className="ws-assistant-stats">{metrics.map(([name, value]) => <div key={name}><dt>{name}</dt><dd>{value}</dd></div>)}</dl>
    <p className="ws-assistant-help">Counts describe attempts and model responses, not accuracy. Feedback stays in this session and does not train the model. Clear History removes entries; Reset session clears all totals.</p>
    {!state.history.length ? <p className="ws-assistant-empty">Your real predictions will appear here, most recent first.</p> : <ol className="ws-assistant-history">{state.history.map(r => <li key={r.id} data-attempt-id={r.id}>
      <div className="ws-assistant-history-word"><strong>{r.word ?? r.candidate}</strong><span className={r.status === 'uncertain' ? 'is-uncertain' : ''}>{r.status === 'uncertain' ? 'Uncertain candidate' : 'Prediction available'}</span></div>
      <div className="ws-assistant-history-meta"><time dateTime={new Date(r.at).toISOString()}>{new Date(r.at).toLocaleTimeString()}</time><span>{r.mode === 'word' ? '8-word model' : r.mode === 'expanded' ? '20-word model' : '500-word model'}</span><span>{scores(r.confidence)}</span><span>Inference: {duration(r.inferenceMs)}</span>{r.retryOf !== null && <span>Retry of attempt {r.retryOf}</span>}</div>
      <div className="ws-assistant-feedback" aria-label={`Feedback for attempt ${r.id}`}><span>Your feedback:</span><button aria-pressed={r.feedback === 'correct'} onClick={() => controller.send({ type: 'FEEDBACK', id: r.id, feedback: r.feedback === 'correct' ? null : 'correct' })}>Correct</button><button aria-pressed={r.feedback === 'incorrect'} onClick={() => controller.send({ type: 'FEEDBACK', id: r.id, feedback: r.feedback === 'incorrect' ? null : 'incorrect' })}>Incorrect</button></div>
    </li>)}</ol>}
    {!!state.history.length && <p className="ws-assistant-help">Up to 100 responses retained in session memory. Stopping the camera keeps them; leaving this page clears them.</p>}
  </section>
}
