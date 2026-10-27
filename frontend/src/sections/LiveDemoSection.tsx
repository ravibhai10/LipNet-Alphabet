import { SectionHead } from '../components/ui'
import { useDemoPipeline } from '../live-demo/useDemoPipeline'

/**
 * SECTION 13 — LIVE VISUAL SPEECH
 * Premium live demo with deliberate composition, clear hierarchy,
 * and real backend integration. Never fabricates predictions.
 */

export function LiveDemoSection() {
  const d = useDemoPipeline()

  return (
    <section
      className="section demo premium"
      id="demo"
      aria-labelledby="demo-title"
    >
      <div className="container">
        <SectionHead
          id="demo-title"
          kicker="13 — Live Visual Speech"
          title={
            <>
              See what <em>the model sees.</em>
            </>
          }
          lede="Landmark detection runs locally in your browser. When inference is ready, mouth crops are sent to your configured backend for experimental alphabet classification."
        />

        <div className="demo-layout">
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
              <canvas
                ref={d.overlayRef}
                width={960}
                height={540}
                className="demo-overlay"
                aria-hidden="true"
              />

              {d.camStatus !== 'live' && (
                <div className="demo-idle-overlay">
                  {d.camStatus === 'requesting' ? (
                    <p>Requesting camera…</p>
                  ) : d.camStatus === 'denied' || d.camStatus === 'error' ? (
                    <div className="demo-camera-help">
                      <p>{d.camError ?? 'Camera unavailable.'}</p>
                      <ol>
                        <li>Allow camera access for this site.</li>
                        <li>Close other apps using the webcam.</li>
                        <li>Reload the page and try again.</li>
                      </ol>
                    </div>
                  ) : (
                    <p className="demo-idle-hint">The camera stays off until you start it.</p>
                  )}
                </div>
              )}
              {d.camStatus === 'live' && (
                <div className="demo-hud" aria-label="Live camera metrics">
                  <span>{d.readout.fps} FPS</span>
                  <span>{d.readout.latencyMs ?? '—'} ms</span>
                  <span className={d.readout.cropOk ? 'is-good' : ''}>
                    {d.readout.cropOk ? 'LIP CROP OK' : 'SEARCHING'}
                  </span>
                </div>
              )}
            </div>

            <div className="demo-controls">
              {d.running ? (
                <button className="btn btn-ghost" onClick={d.stop}>
                  Stop camera
                </button>
              ) : (
                <button className="btn btn-solid" disabled={d.camStatus === 'requesting'} onClick={() => void d.start()}>
                  {d.camStatus === 'requesting' ? 'Requesting camera…' : 'Start camera'}
                </button>
              )}
              <p className="mono-note" role="status">
                {d.lmStatus === 'loading' && 'Loading landmark model…'}
                {d.lmStatus === 'failed' && `Landmark model failed to load — ${d.lmError ?? 'unknown error'}`}
                {d.lmStatus === 'ready' && d.readout.faceOk && 'Face tracked · 20 mouth landmarks'}
                {d.running && d.lmStatus === 'ready' && !d.readout.faceOk && 'Looking for a face…'}
              </p>
            </div>

            <div className="demo-buffer">
              <div className="demo-buffer-head">
                <p className="mono-note">Temporal frame buffer</p>
                <p className="mono-note">
                  {d.predicting
                    ? `Processing ${d.readout.frames}/${d.sequence.maxFrames} frames…`
                    : `${d.readout.frames} / ${d.sequence.maxFrames} frames · min ${d.sequence.minFrames}`}
                </p>
              </div>
              <canvas
                ref={d.bufferStripRef}
                width={d.sequence.maxFrames * 24}
                height={24}
                className="demo-buffer-strip"
                aria-label={`Frame buffer with ${d.readout.frames} of ${d.sequence.maxFrames} frames`}
              />
            </div>
          </div>

          {/* ---------- side panel ---------- */}
          <aside className="demo-panel" aria-label="Live inference readout">
            {(d.predictionError || d.pipelineError) && <p className="demo-status" role="alert">{d.predictionError || d.pipelineError}</p>}
            <div className="demo-panel-block demo-panel-block--primary">
              <p className="mono-note">Detected letter</p>
              <p className="demo-letter" aria-live="polite">
                {d.backendStatus !== 'online' || !d.modelLoaded
                  ? '—'
                  : d.readout.detected ?? (d.predicting ? '…' : '—')}
              </p>
              <p className="mono-note demo-status" role="status">
                {d.backendStatus === 'online' && d.modelLoaded
                  ? 'Model ready'
                  : d.backendStatus === 'online'
                    ? 'Backend online · model unavailable'
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
                {d.readout.frames}/{d.sequence.maxFrames}
              </p>
              <p className="mono-note">
                {d.readout.frames >= d.sequence.minFrames
                  ? 'sequence ready'
                  : `need ${d.sequence.minFrames - d.readout.frames} more`}
              </p>
            </div>

            <div className="demo-panel-block">
              <p className="mono-note">Model status</p>
              <p className={`demo-value demo-model ${d.backendStatus === 'online' && d.modelLoaded ? 'is-online' : ''}`}>
                {d.backendStatus === 'online' && d.modelLoaded
                  ? 'Ready'
                  : d.backendStatus === 'online'
                    ? 'Backend only'
                    : 'Unavailable'}
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
