/**
 * Workspace — Live Demo
 *
 * Reuses the production live-demo pipeline (useDemoPipeline) so the
 * /demo/live route is byte-identical to the #demo section on the
 * landing page. No fabricated predictions: if the backend is offline
 * the panel reads "Model connection unavailable".
 */
import { useDemoPipeline } from "../../live-demo/useDemoPipeline"
import { SEQUENCE } from "../../data/architecture"
import { useBackend } from "../../hooks/useBackend"
import { useEffect } from "react"

export function LiveDemoPage() {
  const d = useDemoPipeline()
  const { status, refresh } = useBackend()
  useEffect(() => { refresh() }, [refresh])

  return (
    <div className="ws-page-inner">
      <header className="ws-page-head">
        <h1>Live Demo</h1>
        <p className="ws-page-sub">Real webcam → MediaPipe FaceMesh → 20 mouth landmarks → 64×64 crop → backend /api/predict</p>
      </header>

      <div className="ws-demo-layout">
        <div className="ws-demo-stage">
          <div className="ws-video-wrap">
            <video
              ref={d.videoRef}
              className="ws-video"
              playsInline
              muted
              aria-label="Live camera feed"
            />
            <canvas ref={d.overlayRef} width={960} height={540} className="ws-overlay" aria-hidden="true" />
            {d.camStatus !== "live" && (
              <div className="ws-idle">
                {d.camStatus === "requesting" ? <p>Requesting camera…</p>
                  : d.camStatus === "denied" || d.camStatus === "error" ? <p>{d.camError ?? "Camera unavailable."}</p>
                  : <p>The camera stays off until you start it.</p>}
              </div>
            )}
          </div>

          <div className="ws-controls">
            {d.running
              ? <button className="ws-btn ws-btn-ghost" onClick={d.stop}>Stop camera</button>
              : <button className="ws-btn ws-btn-solid" onClick={() => void d.start()}>Start camera</button>}
            <span className="ws-mono">
              {d.lmStatus === "loading" && "Loading landmark model…"}
              {d.lmStatus === "failed" && `Landmark model failed: ${d.lmError ?? "unknown"}`}
              {d.lmStatus === "ready" && d.readout.faceOk && "Face tracked · 20 mouth landmarks"}
              {d.lmStatus === "ready" && !d.readout.faceOk && "Looking for a face…"}
            </span>
          </div>

          <div className="ws-buffer">
            <div className="ws-buffer-head">
              <span className="ws-mono">Temporal frame buffer</span>
              <span className="ws-mono">{d.readout.frames} / {SEQUENCE.maxFrames} · min {SEQUENCE.minFrames}</span>
            </div>
            <canvas ref={d.bufferStripRef} width={SEQUENCE.maxFrames * 24} height={24} className="ws-buffer-strip" aria-label={`Buffer ${d.readout.frames} of ${SEQUENCE.maxFrames}`} />
          </div>
        </div>

        <aside className="ws-demo-panel">
          <div className="ws-panel-block">
            <span className="ws-mono">Predicted letter</span>
            <p className="ws-letter" aria-live="polite">
              {status !== "online" ? "—" : d.readout.detected ?? (d.predicting ? "…" : "—")}
            </p>
            <span className="ws-mono">
              {status === "online" ? "Connected" : status === "checking" ? "Checking…" : "Model connection unavailable"}
            </span>
          </div>

          <div className="ws-panel-block">
            <span className="ws-mono">Confidence</span>
            <p className="ws-value">{d.readout.confidence != null ? d.readout.confidence.toFixed(2) : "—"}</p>
          </div>

          <div className="ws-panel-block">
            <span className="ws-mono">Top 3</span>
            <ul className="ws-top3">
              {[0, 1, 2].map((i) => {
                const t = d.readout.top3?.[i]
                return (
                  <li key={i}>
                    <span>{t ? t.letter : "—"}</span>
                    <span className="ws-top3-bar" aria-hidden="true">
                      <i style={{ width: t ? `${Math.round(t.prob * 100)}%` : "0%" }} />
                    </span>
                    <span className="ws-mono">{t ? t.prob.toFixed(2) : "—"}</span>
                  </li>
                )
              })}
            </ul>
          </div>

          <div className="ws-streams">
            <div>
              <span className="ws-mono">Appearance</span>
              <canvas ref={d.appearanceRef} width={64} height={64} className="ws-appearance" aria-label="64×64 mouth crop" />
            </div>
            <div>
              <span className="ws-mono">Geometry</span>
              <canvas ref={d.geometryRef} width={120} height={40} className="ws-spark" aria-label="Mouth openness" />
            </div>
            <div>
              <span className="ws-mono">Motion</span>
              <canvas ref={d.motionRef} width={120} height={40} className="ws-spark" aria-label="Mouth motion energy" />
            </div>
          </div>

          <p className="ws-mono ws-fps">
            {d.running ? `Landmark ≈ ${d.readout.fps} Hz · local` : "Camera idle"}
          </p>
        </aside>
      </div>
    </div>
  )
}