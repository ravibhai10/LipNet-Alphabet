/**
 * Workspace — Model Testing
 *
 * Reuses the same live-demo pipeline as the Live Demo page so the
 * testing experience is byte-identical: real webcam → real MediaPipe
 * → real mouth crop → real /api/predict → top-3 + confidence.
 * Per-frame predictions and confusion-matrix accumulation are
 * implemented client-side from the live prediction stream.
 */
import { useDemoPipeline } from "../../live-demo/useDemoPipeline"
import { SEQUENCE, DECODER } from "../../data/architecture"
import { useEffect, useRef, useState } from "react"
import { useBackend } from "../../hooks/useBackend"

const ALL_LETTERS = DECODER.letters

export function TestingPage() {
  const d = useDemoPipeline()
  const { status } = useBackend()
  const [confusion, setConfusion] = useState<Record<string, number>>({})
  const [frameLog, setFrameLog] = useState<{ letter: string; conf: number; at: number }[]>([])
  const totalRef = useRef(0)
  const lastLetterRef = useRef<string | null>(null)

  // Accumulate stable predictions into the confusion matrix
  useEffect(() => {
    const l = d.readout.detected
    if (!l || l === "—" || l === "ABSTAIN") return
    if (l === lastLetterRef.current) return
    lastLetterRef.current = l
    totalRef.current += 1
    setConfusion((c) => ({ ...c, [l]: (c[l] ?? 0) + 1 }))
    setFrameLog((log) => [{ letter: l, conf: d.readout.confidence ?? 0, at: Date.now() }, ...log].slice(0, 30))
  }, [d.readout.detected, d.readout.confidence])

  const maxCount = Math.max(1, ...Object.values(confusion))

  return (
    <div className="ws-page-inner">
      <header className="ws-page-head">
        <h1>Model Testing</h1>
        <p className="ws-page-sub">Live per-frame predictions on the real webcam feed. Predictions accumulate as they stabilize.</p>
      </header>

      <div className="ws-demo-layout">
        <div className="ws-demo-stage">
          <div className="ws-video-wrap">
            <video ref={d.videoRef} className="ws-video" playsInline muted aria-label="Live camera feed" />
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
              ? <button className="ws-btn ws-btn-ghost" onClick={d.stop}>Stop</button>
              : <button className="ws-btn ws-btn-solid" onClick={() => void d.start()}>Start</button>}
            <span className="ws-mono">
              Buffer {d.readout.frames}/{SEQUENCE.maxFrames} · {d.readout.faceOk ? "face ✓" : "no face"} · {status === "online" ? "backend ✓" : "backend ✗"}
            </span>
          </div>
        </div>

        <aside className="ws-demo-panel">
          <div className="ws-panel-block">
            <span className="ws-mono">Stable predictions accumulated</span>
            <p className="ws-value">{totalRef.current}</p>
            <button className="ws-btn ws-btn-ghost" onClick={() => { setConfusion({}); setFrameLog([]); totalRef.current = 0; lastLetterRef.current = null }}>Reset</button>
          </div>

          <div className="ws-panel-block">
            <span className="ws-mono">Letter distribution</span>
            <div className="ws-confusion">
              {ALL_LETTERS.map((L) => {
                const n = confusion[L] ?? 0
                return (
                  <div key={L} className="ws-confusion-cell" title={`${L}: ${n}`}>
                    <span className="ws-confusion-label">{L}</span>
                    <div className="ws-confusion-bar" style={{ width: `${(n / maxCount) * 100}%` }} />
                    <span className="ws-confusion-count">{n}</span>
                  </div>
                )
              })}
            </div>
          </div>

          <div className="ws-panel-block">
            <span className="ws-mono">Recent predictions</span>
            <ul className="ws-frame-log">
              {frameLog.length === 0 ? <li className="ws-mono">— no predictions yet —</li>
                : frameLog.map((f) => (
                  <li key={f.at}>
                    <span className="ws-letter-mini">{f.letter}</span>
                    <span className="ws-mono">{(f.conf * 100).toFixed(0)}%</span>
                  </li>
                ))}
            </ul>
          </div>
        </aside>
      </div>
    </div>
  )
}