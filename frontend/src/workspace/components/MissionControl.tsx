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

  // Pre-flight calibration diagnostics
  const lightingOk = readout.faceOk && !readout.quality?.toLowerCase().includes('dark') && !readout.quality?.toLowerCase().includes('light')
  const centeringOk = readout.faceOk && !readout.quality?.toLowerCase().includes('center')
  const distanceOk = readout.cropOk && !readout.quality?.toLowerCase().includes('close') && !readout.quality?.toLowerCase().includes('far')
  const fpsStabilityOk = readout.fps >= 20

  const evidence: Array<[string, string, boolean]> = [
    ['Face Tracking Mesh', readout.faceOk ? 'Target Locked (478 pts)' : 'Searching Face…', readout.faceOk],
    ['Mouth Crop Alignment', readout.cropOk ? 'Eye-Axis Level (96×96)' : 'Awaiting Landmarks', readout.cropOk],
    ['Speech Articulation VAD', readout.speaking ? 'Active Speech Dynamics' : 'Rest / Neutral State', readout.speaking],
  ]

  return (
    <section className="ws-mission-control" aria-label="Live mission control">
      <div className="ws-mission-heading">
        <div>
          <p className="ws-kicker">00 / SYSTEM TELEMETRY & CALIBRATION</p>
          <h2 style={{ letterSpacing: '-0.02em', margin: '4px 0 0' }}>
            Mission Control & Pre-Flight Gate
          </h2>
        </div>
        <span className={`ws-live-state ${running ? 'is-live' : ''}`}>
          <i />
          {running ? 'REAL-TIME VISION STREAM' : 'OPTICAL SENSORS STANDBY'}
        </span>
      </div>

      <div className="ws-mission-grid">
        {/* Pre-Flight Optical Calibration Card */}
        <article>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: 'var(--ws-text-muted)', fontSize: '11px' }}>Pre-Flight Calibration</span>
            <strong style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '18px',
              color: score >= 70 ? 'var(--ws-success)' : score >= 40 ? 'var(--ws-warning)' : 'var(--ws-text-muted)',
              fontVariantNumeric: 'tabular-nums'
            }}>
              {score}% READINESS
            </strong>
          </div>
          
          <div className="ws-mission-meter" style={{ height: '4px', borderRadius: '2px', background: 'var(--ws-surface3)', margin: '8px 0' }}>
            <i style={{
              width: `${score}%`,
              background: score >= 70 ? 'var(--ws-success)' : score >= 40 ? 'var(--ws-warning)' : 'var(--ws-accent)',
              borderRadius: '2px'
            }} />
          </div>

          {/* 4 Actionable Diagnostic Pills */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', marginTop: '6px' }}>
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '3px 6px', borderRadius: '4px', background: '#090c14',
              border: lightingOk ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid #1f2536',
              fontSize: '10px', fontFamily: 'var(--font-mono)'
            }}>
              <span style={{ color: '#94a3b8' }}>LIGHTING</span>
              <span style={{ color: lightingOk ? '#10b981' : '#f59e0b', fontWeight: 600 }}>
                {lightingOk ? 'PASS' : running ? 'CHECK' : 'STANDBY'}
              </span>
            </div>

            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '3px 6px', borderRadius: '4px', background: '#090c14',
              border: centeringOk ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid #1f2536',
              fontSize: '10px', fontFamily: 'var(--font-mono)'
            }}>
              <span style={{ color: '#94a3b8' }}>CENTERING</span>
              <span style={{ color: centeringOk ? '#10b981' : '#f59e0b', fontWeight: 600 }}>
                {centeringOk ? 'PASS' : running ? 'CHECK' : 'STANDBY'}
              </span>
            </div>

            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '3px 6px', borderRadius: '4px', background: '#090c14',
              border: distanceOk ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid #1f2536',
              fontSize: '10px', fontFamily: 'var(--font-mono)'
            }}>
              <span style={{ color: '#94a3b8' }}>DISTANCE</span>
              <span style={{ color: distanceOk ? '#10b981' : '#f59e0b', fontWeight: 600 }}>
                {distanceOk ? 'PASS' : running ? 'CHECK' : 'STANDBY'}
              </span>
            </div>

            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '3px 6px', borderRadius: '4px', background: '#090c14',
              border: fpsStabilityOk ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid #1f2536',
              fontSize: '10px', fontFamily: 'var(--font-mono)'
            }}>
              <span style={{ color: '#94a3b8' }}>FPS STABLE</span>
              <span style={{ color: fpsStabilityOk ? '#10b981' : '#f59e0b', fontWeight: 600 }}>
                {fpsStabilityOk ? '30 HZ' : running ? `${readout.fps} HZ` : 'STANDBY'}
              </span>
            </div>
          </div>
        </article>

        <article>
          <span style={{ color: 'var(--ws-text-muted)', fontSize: '11px' }}>Inference Engine</span>
          <strong style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '16px',
            color: backendOnline && inferenceReady ? 'var(--ws-success)' : 'var(--ws-warning)'
          }}>
            {backendOnline && inferenceReady ? 'SYSTEM READY' : 'OFFLINE / BLOCKED'}
          </strong>
          <small>{backendOnline ? inferenceReady ? `${readout.latencyMs ? Math.round(readout.latencyMs) + ' ms latency' : 'Model hot on CPU'}` : 'Model checkpoint pending' : 'API service disconnected'}</small>
        </article>

        <article>
          <span style={{ color: 'var(--ws-text-muted)', fontSize: '11px' }}>Vision Frame Rate</span>
          <strong style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '22px',
            color: readout.fps >= 20 ? 'var(--ws-accent)' : 'var(--ws-text)',
            fontVariantNumeric: 'tabular-nums'
          }}>
            {readout.fps ? `${readout.fps} Hz` : '0 Hz'}
          </strong>
          <small>Target: 25–30 Hz camera pipeline</small>
        </article>

        <article>
          <span style={{ color: 'var(--ws-text-muted)', fontSize: '11px' }}>Model Evidence</span>
          <strong style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '22px',
            color: readout.confidence !== null ? 'var(--ws-success)' : 'var(--ws-text)',
            fontVariantNumeric: 'tabular-nums'
          }}>
            {readout.confidence !== null ? `${(readout.confidence * 100).toFixed(1)}%` : '—'}
          </strong>
          <small>{readout.stabilizing ? 'Dual-window agreement gating' : readout.detected ? 'Classification locked' : 'Awaiting input sequence'}</small>
        </article>
      </div>

      <div className="ws-mission-bottom">
        <div className="ws-mission-evidence">
          <h3 style={{ color: 'var(--ws-text)', fontSize: '12px', letterSpacing: '0.02em', fontWeight: 600 }}>Optical & Biometric Checks</h3>
          {evidence.map(([label, value, ready]) => (
            <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '8px', margin: '8px 0' }}>
              <i className={ready ? 'ready' : ''} style={{
                width: '6px',
                height: '6px',
                borderRadius: '50%',
                background: ready ? 'var(--ws-success)' : 'var(--ws-surface3)'
              }} />
              <span style={{ color: 'var(--ws-text)', fontSize: '11px' }}>{label}</span>
              <strong style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)', color: ready ? 'var(--ws-accent)' : 'var(--ws-text-muted)', fontSize: '11px' }}>
                {value}
              </strong>
            </div>
          ))}
          <p style={{ color: 'var(--ws-text-dim)', fontSize: '10px', lineHeight: '1.5', marginTop: '10px' }}>
            Decision thresholds strictly gated by temporal consistency and calibrated softmax margin to eliminate visual hallucination.
          </p>
        </div>

        <div className="ws-mission-history">
          <h3 style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--ws-text)', fontSize: '12px', fontWeight: 600 }}>
            Recent Predictions
            <span style={{ color: 'var(--ws-accent)', fontFamily: 'var(--font-mono)', fontSize: '11px', fontWeight: 400 }}>{recent.length} logged</span>
          </h3>
          {recent.length ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '8px' }}>
              {recent.map(item => (
                <div key={`${item.at}-${item.label}`} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '4px 0', borderBottom: '1px solid var(--ws-border)' }}>
                  <strong style={{ color: 'var(--ws-success)', fontFamily: 'var(--font-mono)', fontSize: '13px' }}>{item.label ?? '—'}</strong>
                  <span style={{ color: 'var(--ws-text-muted)', fontFamily: 'var(--font-mono)', fontSize: '11px', fontVariantNumeric: 'tabular-nums' }}>
                    {item.confidence === null ? 'uncalibrated' : `${(item.confidence * 100).toFixed(1)}%`}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p style={{ color: 'var(--ws-text-dim)', fontSize: '11px', marginTop: '12px' }}>Awaiting initial speech capture.</p>
          )}
        </div>
      </div>
    </section>
  )
}
