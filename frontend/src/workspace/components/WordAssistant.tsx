import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import type { AssistantState, WordAssistant as Controller } from '../../live-demo/wordAssistant'
import { WORD_LIMITS } from '../../live-demo/wordAssistant'
import { Icon } from '../Icon'
import { DecodedWordReveal } from './DecodedWordReveal'
import { ContextInterpretation } from './ContextInterpretation'

interface Props {
  state: AssistantState
  controller: Controller
  canCapture: boolean
  cameraActive: boolean
  begin: (action?: 'start' | 'next' | 'retry') => void
  vocabulary?: string[]
  onInspectFrames?: () => void
}

const scores = (value: number | null) => value === null ? 'Confidence unavailable' : `${(value * 100).toFixed(1)}%`
const duration = (value: number | null) => value === null ? 'Unavailable' : `${Math.round(value)} ms`
const prompts: Record<AssistantState['phase'], string> = {
  IDLE: 'Start your camera to prepare a word.',
  CAMERA_STARTING: 'Starting camera and landmark tracking…',
  READY: 'Ready. Click Start Word, say one word, then click Finish Word.',
  CAPTURING: 'Capturing your word…',
  PROCESSING: 'Analyzing lip movements…',
  RESULT: 'Prediction complete.',
  WAITING_NEXT: 'Prediction complete. Ready for your next word.',
  RETRY: 'Preparing a fresh capture…',
  ERROR: 'This attempt needs attention.',
  STOPPING: 'Stopping the camera…',
}

export function WordCaptureStatus({ state }: Pick<Props, 'state'>) {
  const limits = WORD_LIMITS[state.mode]
  return (
    <div className={`ws-assistant-capture state-${state.phase.toLowerCase()}`}>
      <div aria-live="polite" aria-atomic="true">
        <span className="ws-assistant-indicator" />
        <strong>{prompts[state.phase]}</strong>
      </div>
      {state.phase === 'CAPTURING' && (
        <>
          <div className="ws-assistant-clock">
            <span>{(state.captureMs / 1000).toFixed(1)} s captured</span>
            <span>{state.frames} frames · minimum {limits.minFrames}</span>
          </div>
          <progress value={state.frames} max={limits.maxFrames} aria-label="Capture buffer used" />
          <p>Click Finish Word when you finish speaking. Limit: {limits.maxDurationMs / 1000} s.</p>
        </>
      )}
      {state.phase === 'PROCESSING' && (
        <p className="ws-assistant-processing">
          <span className="ws-assistant-spinner" aria-hidden="true" />
          {(state.processingMs / 1000).toFixed(1)} s processing on CPU · evaluating representation
        </p>
      )}
    </div>
  )
}

export function WordAssistantPanel({
  state,
  controller,
  canCapture,
  cameraActive,
  begin,
  vocabulary = [],
  onInspectFrames
}: Props) {
  const result = state.result
  const busy = ['CAPTURING', 'PROCESSING', 'CAMERA_STARTING', 'STOPPING'].includes(state.phase)
  const [correctedWord, setCorrectedWord] = useState<string>('')
  useEffect(() => { setCorrectedWord('') }, [result?.id])

  return (
    <div className="ws-assistant" aria-busy={state.phase === 'PROCESSING'}>
      <div className={`ws-assistant-result ${result?.status === 'uncertain' ? 'is-uncertain' : result ? 'is-available' : ''}`}>
        <span className="ws-eyebrow">
          {state.phase === 'PROCESSING' ? 'ANALYZING LIP MOVEMENTS (AV-HuBERT L9)' : result ? 'LAST PREDICTION' : 'DELIBERATE WORD CAPTURE'}
        </span>
        <div className="ws-assistant-word" aria-live="polite" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '68px', padding: '6px 0' }}>
          <DecodedWordReveal
            targetWord={result ? result.word ?? 'Uncertain' : null}
            isProcessing={state.phase === 'PROCESSING'}
            isUncertain={result?.status === 'uncertain'}
          />
        </div>
        <p>
          {result?.status === 'uncertain'
            ? 'Prediction uncertain. Spoken word did not clear the 0.95/0.20 confidence gate.'
            : result
            ? 'Prediction verified · awaiting your next action.'
            : prompts[state.phase]}
        </p>
        {result?.status === 'uncertain' && (
          <p className="ws-assistant-candidate">
            Top unconstrained candidate: <strong>{result.candidate}</strong>
          </p>
        )}
        {state.phase === 'PROCESSING' && (
          <div className="ws-assistant-processing" role="status">
            <span className="ws-assistant-spinner" aria-hidden="true" />
            {(state.processingMs / 1000).toFixed(1)} s elapsed on CPU
          </div>
        )}
      </div>

        <ContextInterpretation result={result} history={state.history} mode={state.mode} vocabulary={vocabulary} active={cameraActive && ['RESULT', 'WAITING_NEXT'].includes(state.phase)} />

      <dl className="ws-assistant-result-data">
        <div>
          <dt>Model confidence</dt>
          <dd style={{ color: result?.word ? '#10b981' : 'inherit', fontWeight: 600 }}>{result ? scores(result.confidence) : '—'}</dd>
        </div>
        <div>
          <dt>Margin (Δ)</dt>
          <dd style={{ color: result?.diagnostics?.margin && result.diagnostics.margin > 0.2 ? '#38bdf8' : 'inherit' }}>{result?.diagnostics?.margin != null ? result.diagnostics.margin.toFixed(3) : '—'}</dd>
        </div>
        <div>
          <dt>Entropy (H)</dt>
          <dd>{result?.diagnostics?.entropy != null ? `${result.diagnostics.entropy.toFixed(3)} nats` : '—'}</dd>
        </div>
        <div>
          <dt>Server inference</dt>
          <dd>{result ? duration(result.inferenceMs) : '—'}</dd>
        </div>
        <div>
          <dt>Request round trip</dt>
          <dd>{result ? duration(result.roundTripMs) : '—'}</dd>
        </div>
      </dl>

      {/* Candidate Probability Distribution (Explainable AI) */}
      {result && result.candidates && result.candidates.length > 0 && (
        <div style={{ margin: '14px 0 10px 0', padding: '12px 14px', background: 'rgba(255, 255, 255, 0.03)', borderRadius: '10px', border: '1px solid rgba(56, 189, 248, 0.2)', boxShadow: 'inset 0 1px 0 rgba(255, 255, 255, 0.05)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px', fontSize: '10px', letterSpacing: '0.08em', textTransform: 'uppercase', color: '#94a3b8' }}>
            <span>Candidate Probability Distribution</span>
            <span>Softmax Confidence</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {result.candidates.slice(0, 4).map((c, i) => (
              <div key={c.word} style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '12px' }}>
                <span style={{ width: '85px', fontWeight: i === 0 ? 700 : 400, color: i === 0 && result.word ? '#10b981' : '#cbd5e1' }}>
                  {c.word} {c.word === 'help' ? '⭐' : ''}
                </span>
                <div className="ws-shimmer-bar" style={{ flex: 1, height: '7px', background: 'rgba(255, 255, 255, 0.08)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{
                    width: `${Math.round(c.prob * 100)}%`,
                    height: '100%',
                    background: i === 0 
                      ? (result.word ? 'linear-gradient(90deg, #10b981, #38bdf8)' : 'linear-gradient(90deg, #f59e0b, #fbbf24)') 
                      : 'rgba(255, 255, 255, 0.2)',
                    borderRadius: '4px',
                    transition: 'width 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
                    boxShadow: i === 0 ? '0 0 10px rgba(56, 189, 248, 0.5)' : 'none'
                  }} />
                </div>
                <span style={{ width: '45px', textAlign: 'right', fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: i === 0 ? '#38bdf8' : '#94a3b8', fontWeight: i === 0 ? 600 : 400 }}>
                  {(c.prob * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Human-in-the-Loop Feedback & Correction Section */}
      {result && (
        <div className="ws-assistant-feedback-box" role="region" aria-label="Prediction Feedback">
          <span className="ws-feedback-label">Was this prediction correct?</span>
          <div className="ws-feedback-button-group">
            <button
              type="button"
              className={`ws-btn ws-btn-sm ${result.feedback === 'correct' ? 'ws-btn-primary' : 'ws-btn-ghost'}`}
              disabled={!result.word}
              onClick={() => controller.send({ type: 'FEEDBACK', id: result.id, feedback: result.feedback === 'correct' ? null : 'correct' })}
              aria-pressed={result.feedback === 'correct'}
            >
              <Icon name="check" size={13} /> Correct
            </button>
            <button
              type="button"
              className={`ws-btn ws-btn-sm ${result.feedback === 'incorrect' ? 'ws-btn-primary is-danger' : 'ws-btn-ghost'}`}
              onClick={() => controller.send({ type: 'FEEDBACK', id: result.id, feedback: result.feedback === 'incorrect' ? null : 'incorrect' })}
              aria-pressed={result.feedback === 'incorrect'}
            >
              <Icon name="close" size={13} /> Incorrect
            </button>
          </div>

          {result.feedback === 'incorrect' && vocabulary.length > 0 && (
            <div className="ws-feedback-correction-row">
              <label htmlFor={`correct-val-${result.id}`}>Select actual spoken word:</label>
              <select
                id={`correct-val-${result.id}`}
                className="ws-select"
                value={correctedWord}
                onChange={e => setCorrectedWord(e.target.value)}
              >
                <option value="">Choose the actual spoken word</option>
                {vocabulary.map(w => (
                  <option key={w} value={w}>
                    {w}
                  </option>
                ))}
                <option value="UNKNOWN">UNKNOWN / OTHER</option>
              </select>
            </div>
          )}

          <p className="ws-feedback-disclaimer">
            This feedback stays in session memory. <Link to="/demo/feedback">Use the adaptive lab to record, verify and save a dataset sample.</Link>
          </p>
        </div>
      )}

      {/* Primary Action Buttons */}
      <div className="ws-assistant-actions">
        {state.phase === 'READY' && (
          <button className="ws-btn ws-btn-primary" disabled={!canCapture} onClick={() => begin('start')}>
            <Icon name="camera" size={14} /> Start Word
          </button>
        )}
        {state.phase === 'CAPTURING' && (
          <button
            className="ws-btn ws-btn-primary"
            disabled={state.frames < WORD_LIMITS[state.mode].minFrames}
            onClick={controller.finish}
          >
            <Icon name="stop" size={14} /> Finish Word
          </button>
        )}
        {state.phase === 'PROCESSING' && (
          <button className="ws-btn ws-btn-primary" disabled>
            <span className="ws-assistant-spinner" aria-hidden="true" /> Analyzing…
          </button>
        )}
        {['RESULT', 'WAITING_NEXT'].includes(state.phase) && (
          <>
            <button className="ws-btn ws-btn-primary" disabled={!canCapture} onClick={() => begin('next')}>
              {result?.status === 'uncertain' ? 'Continue · Next Word' : 'Next Word'}
              <Icon name="arrow" size={14} />
            </button>
            <button className="ws-btn ws-btn-ghost" disabled={!canCapture} onClick={() => begin('retry')}>
              <Icon name="refresh" size={14} /> Retry Word
            </button>
            {onInspectFrames && (
              <button
                type="button"
                className="ws-btn ws-btn-ghost"
                onClick={onInspectFrames}
                title="Inspect captured mouth crops frame-by-frame"
              >
                <Icon name="layers" size={14} /> Review Frames
              </button>
            )}
          </>
        )}
        {state.phase === 'ERROR' && cameraActive && (
          <>
            <button className="ws-btn ws-btn-primary" disabled={!canCapture} onClick={() => begin('retry')}>
              Retry Word
            </button>
            <button className="ws-btn ws-btn-ghost" onClick={controller.recover}>
              Back to Ready
            </button>
          </>
        )}
      </div>

      {cameraActive && !canCapture && !busy && (
        <p className="ws-assistant-help">
          Position one face inside the frame, keep your mouth visible, and check model readiness.
        </p>
      )}
      {!cameraActive && result && (
        <p className="ws-assistant-help">
          Camera stopped. Your result and history are retained. Start the camera to continue.
        </p>
      )}
      {state.phase === 'WAITING_NEXT' && cameraActive && (
        <p className="ws-assistant-waiting" role="status">
          Ready for your next word. Capture stays paused until you choose Next Word or Retry Word.
        </p>
      )}

      {/* Top Candidates Ranked List */}
      {!!result?.candidates.length && (
        <details className="ws-assistant-settings" open>
          <summary>Top Ranked Candidate Probabilities</summary>
          <ul className="ws-assistant-candidates-list">
            {result.candidates.slice(0, 4).map((c, i) => (
              <li key={c.word}>
                <span className="ws-candidate-rank">#{i + 1}</span>
                <span className="ws-candidate-word">
                  {c.word} {c.word === result.word ? '✓' : ''}
                </span>
                <div className="ws-candidate-bar-bg">
                  <div
                    className="ws-candidate-bar-fill"
                    style={{ width: `${Math.round(c.prob * 100)}%` }}
                  />
                </div>
                <span className="ws-candidate-pct">{(c.prob * 100).toFixed(1)}%</span>
              </li>
            ))}
          </ul>
        </details>
      )}

      <details className="ws-assistant-settings">
        <summary>Optional Score Rejection Gate</summary>
        <label>
          <input
            type="checkbox"
            disabled={busy}
            checked={state.minimumScore !== null}
            onChange={e => controller.send({ type: 'THRESHOLD', value: e.target.checked ? 0.8 : null })}
          />
          Apply an extra minimum score filter
        </label>
        {state.minimumScore !== null && (
          <label>
            Minimum score: {(state.minimumScore * 100).toFixed(0)}%
            <input
              aria-label="Minimum model score"
              disabled={busy}
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={state.minimumScore}
              onChange={e => controller.send({ type: 'THRESHOLD', value: Number(e.target.value) })}
            />
          </label>
        )}
      </details>
    </div>
  )
}

export function WordSession({ state, controller }: Pick<Props, 'state' | 'controller'>) {
  const { totals } = state
  const busy = ['CAPTURING', 'PROCESSING', 'CAMERA_STARTING', 'STOPPING'].includes(state.phase)
  const metrics = [
    ['Words attempted', totals.attempts],
    ['Predictions generated', totals.predictions],
    ['Available predictions', totals.available],
    ['Uncertain (abstained)', totals.uncertain],
    ['Failed requests', totals.failedRequests],
    ['Retries', totals.retries],
    ['Avg server inference', totals.timedPredictions ? duration(totals.inferenceTotalMs / totals.timedPredictions) : '—']
  ]

  return (
    <section className="ws-card ws-assistant-session" aria-label="Word session summary">
      <div className="ws-card-head">
        <h2>Session summary & history</h2>
        <button
          className="ws-inline-link"
          disabled={!state.history.length || busy}
          onClick={() => controller.send({ type: 'CLEAR_HISTORY' })}
        >
          Clear History
        </button>
      </div>

      <dl className="ws-assistant-stats">
        {metrics.map(([name, value]) => (
          <div key={name}>
            <dt>{name}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>

      <p className="ws-assistant-help">
        Session feedback stays in browser memory for telemetry analysis. Clear History removes entries; Reset session clears all totals.
      </p>

      {!state.history.length ? (
        <p className="ws-assistant-empty">Your real predictions will appear here, most recent first.</p>
      ) : (
        <ol className="ws-assistant-history">
          {state.history.map(r => (
            <li key={r.id} data-attempt-id={r.id}>
              <div className="ws-assistant-history-word">
                <strong>{r.word ?? r.candidate}</strong>
                <span className={r.status === 'uncertain' ? 'is-uncertain' : ''}>
                  {r.status === 'uncertain' ? 'Abstained' : 'Available'}
                </span>
                {r.feedback && (
                  <span className={`ws-feedback-tag is-${r.feedback}`}>
                    {r.feedback === 'correct' ? 'Verified Correct' : 'Flagged Incorrect'}
                  </span>
                )}
              </div>
              <div className="ws-assistant-history-meta">
                <time dateTime={new Date(r.at).toISOString()}>{new Date(r.at).toLocaleTimeString()}</time>
                <span>{r.mode === 'word' ? '8-word AV-HuBERT' : r.mode === 'expanded' ? '20-word model' : '500-word model'}</span>
                <span>{scores(r.confidence)}</span>
                <span>Inference: {duration(r.inferenceMs)}</span>
                {r.retryOf !== null && <span>Retry of attempt {r.retryOf}</span>}
              </div>
            </li>
          ))}
        </ol>
      )}
    </section>
  )
}
