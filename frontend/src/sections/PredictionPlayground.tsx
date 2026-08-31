/**
 * PredictionPlayground.tsx — Frame-by-frame top-3 prediction playground.
 * Reuses the same useDemoPipeline as the workspace Live Demo page.
 */
import { useDemoPipeline } from "../live-demo/useDemoPipeline"
import { useState } from "react"
import { useBackend } from "../hooks/useBackend"
import { SEQUENCE } from "../data/architecture"

export function PredictionPlayground() {
  const d = useDemoPipeline()
  const { status } = useBackend()
  const [running, setRunning] = useState(false)
  const [history, setHistory] = useState<{ letter: string; conf: number }[]>([])

  const handleStart = async () => { setRunning(true); await d.start() }
  const handleStop = () => { d.stop(); setRunning(false) }

  const letter = d.readout.detected
  if (letter && letter !== "—" && letter !== "ABSTAIN" && running) {
    const last = history[0]
    if (!last || last.letter !== letter) {
      setHistory((h) => [{ letter, conf: d.readout.confidence ?? 0 }, ...h].slice(0, 30))
    }
  }
  const top3 = d.readout.top3 ?? []

  return (
    <section id="prediction-playground" style={{ width: "100%", background: "#08080c", padding: "64px 24px", display: "flex", flexDirection: "column", alignItems: "center", gap: 32 }}>
      <h2 style={{ fontFamily: "Fraunces, serif", fontSize: "clamp(24px,3vw,40px)", fontWeight: 700, color: "#fff", margin: 0 }}>Prediction Playground</h2>
      <p style={{ color: "#7070a0", fontSize: 15, textAlign: "center", maxWidth: 480 }}>Webcam feed with frame-by-frame top-3 confidence bars. Same engine as the workspace Live Demo.</p>
      <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
        {running
          ? <button onClick={handleStop} style={{ background: "rgba(228,76,76,0.15)", color: "#e44e4e", border: "1px solid #e44e4e", borderRadius: 8, padding: "8px 20px", cursor: "pointer", fontFamily: "JetBrains Mono, monospace", fontSize: 13 }}>Stop</button>
          : <button onClick={handleStart} style={{ background: "rgba(78,228,200,0.15)", color: "#4ee4c8", border: "1px solid #4ee4c8", borderRadius: 8, padding: "8px 20px", cursor: "pointer", fontFamily: "JetBrains Mono, monospace", fontSize: 13 }}>Start</button>}
        <span style={{ color: "#7070a0", fontFamily: "JetBrains Mono, monospace", fontSize: 13 }}>Buffer {d.readout.frames}/{SEQUENCE.maxFrames} · {d.readout.faceOk ? "face ✓" : "no face"} · {status === "online" ? "backend ✓" : "backend ✗"}</span>
      </div>
      <div style={{ display: "flex", gap: 24, width: "100%", maxWidth: 800, flexWrap: "wrap" }}>
        <div style={{ flex: 1, minWidth: 240, background: "#0f0f14", borderRadius: 12, padding: 20, border: "1px solid rgba(255,255,255,0.06)" }}>
          <p style={{ color: "#7070a0", fontFamily: "JetBrains Mono, monospace", fontSize: 12, margin: "0 0 12px" }}>TOP 3 PROBABILITIES</p>
          {top3.length === 0 ? <p style={{ color: "#404060", fontFamily: "JetBrains Mono, monospace", fontSize: 13 }}>—</p> : top3.map((t, i) => (
            <div key={i} style={{ marginBottom: 10 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ color: i === 0 ? "#00f0ff" : i === 1 ? "#7b61ff" : "#4ee4c8", fontFamily: "JetBrains Mono, monospace", fontSize: 15, fontWeight: 700 }}>{t.letter}</span>
                <span style={{ color: "#7070a0", fontFamily: "JetBrains Mono, monospace", fontSize: 13 }}>{(t.prob * 100).toFixed(1)}%</span>
              </div>
              <div style={{ height: 6, background: "#16161e", borderRadius: 3, overflow: "hidden" }}>
                <div style={{ height: "100%", width: `${(t.prob * 100).toFixed(1)}%`, background: i === 0 ? "linear-gradient(90deg,#00f0ff,#7b61ff)" : i === 1 ? "#7b61ff" : "#4ee4c8", borderRadius: 3, transition: "width 0.3s ease" }} />
              </div>
            </div>
          ))}
        </div>
        <div style={{ flex: 1, minWidth: 240, background: "#0f0f14", borderRadius: 12, padding: 20, border: "1px solid rgba(255,255,255,0.06)" }}>
          <p style={{ color: "#7070a0", fontFamily: "JetBrains Mono, monospace", fontSize: 12, margin: "0 0 12px" }}>PREDICTION HISTORY</p>
          {history.length === 0 ? <p style={{ color: "#404060", fontFamily: "JetBrains Mono, monospace", fontSize: 13 }}>—</p> : history.map((h, i) => (
            <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
              <span style={{ color: "#00f0ff", fontFamily: "JetBrains Mono, monospace", fontSize: 14 }}>{h.letter}</span>
              <span style={{ color: "#7070a0", fontFamily: "JetBrains Mono, monospace", fontSize: 12 }}>{(h.conf * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
