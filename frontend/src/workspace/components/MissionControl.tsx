import type { DemoReadout, SessionResult } from '../../live-demo/useDemoPipeline'
import type { AssistantState } from '../../live-demo/wordAssistant'

interface Props {
  readout: DemoReadout
  running: boolean
  backendOnline: boolean
  inferenceReady: boolean
  history: SessionResult[]
  assistant?: AssistantState
}

function qualityScore(readout: DemoReadout) {
  return Math.round((readout.faceOk ? 35 : 0) + (readout.cropOk ? 35 : 0) + (readout.speaking ? 20 : 10) + (readout.fps >= 12 ? 10 : readout.fps > 0 ? 5 : 0))
}

export function MissionControl({ readout, running, backendOnline, inferenceReady, history, assistant }: Props) {
  const wordHistory = assistant?.history ?? []
  const recent = assistant ? wordHistory.slice(0, 4).map(item => ({ label: item.word ?? item.candidate, confidence: item.confidence, at: item.at })) : history.slice(0, 4).map(item => ({ label: item.letter, confidence: item.confidence, at: item.at }))
  const score = qualityScore(readout)
  const evidence: Array<[string, string, boolean]> = [
    ['Face tracking', readout.faceOk ? 'Locked' : 'Searching', readout.faceOk],
    ['Mouth crop', readout.cropOk ? 'Stable' : 'Waiting', readout.cropOk],
    ['Movement', readout.speaking ? 'Detected' : 'Quiet', readout.speaking],
  ]
  return <section className="ws-mission-control" aria-label="Live mission control">
    <div className="ws-mission-heading"><div><p className="ws-kicker">MISSION CONTROL</p><h2>Readiness at a glance</h2></div><span className={`ws-live-state ${running ? 'is-live' : ''}`}><i />{running ? 'LIVE SESSION' : 'STANDBY'}</span></div>
    <div className="ws-mission-grid">
      <article><span>Capture quality</span><strong>{score}%</strong><div className="ws-mission-meter"><i style={{ width: `${score}%` }} /></div><small>{readout.quality ?? (running ? 'Measuring camera signal…' : 'Start camera to measure')}</small></article>
      <article><span>Pipeline</span><strong>{backendOnline && inferenceReady ? 'READY' : 'BLOCKED'}</strong><small>{backendOnline ? inferenceReady ? `${readout.latencyMs ? Math.round(readout.latencyMs) + ' ms last round trip' : 'Waiting for first response'}` : 'Model unavailable' : 'Backend disconnected'}</small></article>
      <article><span>Frame rate</span><strong>{readout.fps ? `${readout.fps} Hz` : '—'}</strong><small>Landmark processing cadence</small></article>
      <article><span>Evidence</span><strong>{readout.confidence !== null ? `${(readout.confidence * 100).toFixed(0)}%` : '—'}</strong><small>{readout.stabilizing ? 'Second consistent window required' : readout.detected ? 'Prediction confirmed' : 'No confirmed result'}</small></article>
    </div>
    <div className="ws-mission-bottom">
      <div className="ws-mission-evidence"><h3>Why this result?</h3>{evidence.map(([label, value, ready]) => <div key={label}><i className={ready ? 'ready' : ''} /><span>{label}</span><strong>{value}</strong></div>)}<p>Scores are model outputs, not calibrated probabilities. Low evidence should be retried rather than treated as a fact.</p></div>
      <div className="ws-mission-history"><h3>Recent predictions <span>{recent.length}</span></h3>{recent.length ? recent.map(item => <div key={`${item.at}-${item.label}`}><strong>{item.label ?? '—'}</strong><span>{item.confidence === null ? 'unavailable' : `${(item.confidence * 100).toFixed(0)}%`}</span></div>) : <p>No predictions in this session yet.</p>}</div>
    </div>
  </section>
}
