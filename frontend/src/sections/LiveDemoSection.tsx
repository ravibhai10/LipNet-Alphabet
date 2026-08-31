import { SectionHead } from '../components/ui'
import { useDemoPipeline } from '../live-demo/useDemoPipeline'
import { SEQUENCE } from '../data/architecture'

/**
 * SECTION 09 — LIVE DEMO
 * Real webcam + real MediaPipe landmarks (the project's own
 * face_landmarker.task, lazy-loaded). Letter decoding requires the optional
 * Python inference backend; when it is unreachable the panel reads
 * "Model connection unavailable" — never a fabricated prediction.
 */

export function LiveDemoSection() {
  const d = useDemoPipeline()

  return (
    <section className="section demo" id="demo" aria-labelledby="demo-title">
      <div className="container">
        <SectionHead
          id="demo-title"
          kicker="13 — Live Visual Speech"
          title={
            <>
              See what <em>the model sees.</em>
            </>
          }
          lede="Real landmark detection runs locally in your browser — nothing is uploaded. Letter decoding uses the project inference backend when connected."
        />

        <div className="demo-grid">
          {/* ---------- viewport ---------- */}
          <div className="demo-stage">
            <div className="demo-video-wrap">
              <video
                ref={d.videoRef}
                className="demo-video"
                playsInline
                muted
                aria-label="Live camera feed"
              />
              <canvas ref={d.overlayRef} width={960} height={540} className="demo-overlay" aria-hidden="true" />

              {d.camStatus !== 'live' && (
                <div className="demo-idle">
                  {d.camStatus === 'requesting' ? (
                    <p>Requesting camera…</p>
                  ) : d.camStatus === 'denied' || d.camStatus === 'error' ? (
                    <p>{d.camError ?? 'Camera unavailable.'}</p>
                  ) : (
                    <p className="demo-idle-hint">The camera stays off until you start it.</p>
                  )}
                </div>
              )}
            </div>

            <div className="demo-controls">
              {d.running ? (
                <button className="btn btn-ghost" onClick={d.stop}>
                  Stop camera
                </button>
              ) : (
                <button className="btn btn-solid" onClick={() => void d.start()}>
                  Start camera
                </button>
              )}
              <p className="mono-note" role="status">
                {d.lmStatus === 'loading' && 'Loading landmark model…'}
                {d.lmStatus === 'failed' && `Landmark model failed to load — ${d.lmError ?? 'unknown error'}`}
                {d.lmStatus === 'ready' && d.readout.faceOk && 'Face tracked · 20 mouth landmarks'}
                {d.lmStatus === 'ready' && !d.readout.faceOk && 'Looking for a face…'}
              </p>
            </div>

            <div className="demo-buffer">
              <div className="demo-buffer-head">
                <p className="mono-note">Temporal frame buffer</p>
                <p className="mono-note">
                  {d.readout.frames} / {SEQUENCE.maxFrames} frames · min {SEQUENCE.minFrames}
                </p>
              </div>
              <canvas
                ref={d.bufferStripRef}
                width={SEQUENCE.maxFrames * 24}
                height={24}
                className="demo-buffer-strip"
                aria-label={`Frame buffer with ${d.readout.frames} of ${SEQUENCE.maxFrames} frames`}
              />
            </div>
          </div>

          {/* ---------- side panel ---------- */}
          <aside className="demo-panel" aria-label="Live inference readout">
            <div className="demo-panel-block demo-panel-block--primary">
              <p className="mono-note">Detected letter</p>
              <p className="demo-letter" aria-live="polite">
                {d.backendStatus !== 'online' ? '—' : d.readout.detected ?? (d.predicting ? '…' : '—')}
              </p>
              <p className="mono-note demo-status" role="status">
                {d.backendStatus === 'online'
                  ? 'Connected'
                  : d.backendStatus === 'checking'
                    ? 'Checking…'
                    : 'Model connection unavailable'}
              </p>
            </div>

            <div className="demo-panel-block">
              <p className="mono-note">Confidence</p>
              <p className="demo-value">
                {d.readout.confidence != null ? d.readout.confidence.toFixed(2) : '—'}
              </p>
            </div>

            <div className="demo-panel-block">
              <p className="mono-note">Top 3</p>
              <ul className="demo-top3">
                {[0, 1, 2].map((i) => {
                  const t = d.readout.top3?.[i]
                  return (
                    <li key={i}>
                      <span>{t ? t.letter : '—'}</span>
                      <span className="demo-top3-bar" aria-hidden="true">
                        <i style={{ width: t ? `${Math.round(t.prob * 100)}%` : '0%' }} />
                      </span>
                      <span className="mono-note">{t ? t.prob.toFixed(2) : '—'}</span>
                    </li>
                  )
                })}
              </ul>
            </div>

            <div className="demo-panel-block">
              <p className="mono-note">Temporal status</p>
              <p className="demo-value">
                {d.readout.frames}/{SEQUENCE.maxFrames}
              </p>
              <p className="mono-note">
                {d.readout.frames >= SEQUENCE.minFrames
                  ? 'sequence ready'
                  : `need ${SEQUENCE.minFrames - d.readout.frames} more`}
              </p>
            </div>

            <div className="demo-panel-block">
              <p className="mono-note">Model status</p>
              <p className={`demo-value demo-model ${d.backendStatus === 'online' ? 'is-online' : ''}`}>
                {d.backendStatus === 'online' ? 'Ready' : 'Unavailable'}
              </p>
            </div>

            <div className="demo-streams">
              <div className="demo-stream">
                <p className="mono-note">Appearance</p>
                <canvas ref={d.appearanceRef} width={64} height={64} className="demo-appearance" aria-label="Live 64×64 mouth crop" />
                <p className="mono-note">64 × 64 crop</p>
              </div>
              <div className="demo-stream">
                <p className="mono-note">Geometry</p>
                <canvas ref={d.geometryRef} width={120} height={40} className="demo-spark" aria-label="Mouth openness over time" />
                <p className="mono-note">20 landmarks</p>
              </div>
              <div className="demo-stream">
                <p className="mono-note">Motion</p>
                <canvas ref={d.motionRef} width={120} height={40} className="demo-spark" aria-label="Mouth motion energy over time" />
                <p className="mono-note">Δ signal</p>
              </div>
            </div>

            <p className="mono-note demo-fps">
              {d.running ? `Landmark inference ≈ ${d.readout.fps} Hz · runs locally` : 'Camera idle'}
            </p>
          </aside>
        </div>
      </div>
    </section>
  )
}
