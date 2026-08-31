/**
 * AblationStudio.tsx — Toggle each representation stream to see its
 * estimated effect on overall accuracy.
 */
import { useState } from "react"
import { APPEARANCE, GEOMETRY, MOTION } from "../data/architecture"

const BASE_ACCURACY = 0.95

function estimateAccuracy(streams: { appearance: boolean; geometry: boolean; motion: boolean }) {
  let acc = 0
  if (streams.appearance) acc += APPEARANCE.contribution
  if (streams.geometry) acc += GEOMETRY.contribution
  if (streams.motion) acc += MOTION.contribution
  const total = APPEARANCE.contribution + GEOMETRY.contribution + MOTION.contribution
  return total > 0 ? (acc / total) * BASE_ACCURACY : 0
}

export function AblationStudio() {
  const [streams, setStreams] = useState({ appearance: true, geometry: true, motion: true })
  const acc = estimateAccuracy(streams)
  const color = acc > 0.7 ? "#4ee4a4" : acc > 0.4 ? "#e4c44e" : "#e44e4e"

  return (
    <section id="ablation-studio" style={{ width: "100%", background: "#08080c", padding: "64px 24px", display: "flex", flexDirection: "column", alignItems: "center", gap: 32 }}>
      <h2 style={{ fontFamily: "Fraunces, serif", fontSize: "clamp(24px,3vw,40px)", fontWeight: 700, color: "#fff", margin: 0 }}>Ablation Studio</h2>
      <p style={{ color: "#7070a0", fontSize: 15, textAlign: "center", maxWidth: 480 }}>Toggle each representation stream to see its effect on overall accuracy.</p>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", justifyContent: "center" }}>
        {(["appearance", "geometry", "motion"] as const).map((s) => (
          <button key={s} onClick={() => setStreams((p) => ({ ...p, [s]: !p[s] }))}
            style={{ background: streams[s] ? "rgba(78,228,200,0.15)" : "rgba(255,255,255,0.04)", color: streams[s] ? "#4ee4c8" : "#7070a0", border: `1px solid ${streams[s] ? "#4ee4c8" : "rgba(255,255,255,0.08)"}`, borderRadius: 8, padding: "10px 20px", cursor: "pointer", fontFamily: "JetBrains Mono, monospace", fontSize: 13, fontWeight: 600 }}>
            {streams[s] ? "✓" : "✗"} {s.charAt(0).toUpperCase() + s.slice(1)}
          </button>
        ))}
      </div>
      <div style={{ width: "100%", maxWidth: 480, background: "#0f0f14", borderRadius: 12, padding: 24, border: "1px solid rgba(255,255,255,0.06)", textAlign: "center" }}>
        <p style={{ color: "#7070a0", fontFamily: "JetBrains Mono, monospace", fontSize: 12, margin: 0 }}>ESTIMATED OVERALL ACCURACY</p>
        <p style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 56, fontWeight: 700, color, margin: "8px 0" }}>{(acc * 100).toFixed(1)}%</p>
        <p style={{ color: "#404060", fontFamily: "JetBrains Mono, monospace", fontSize: 11, margin: 0 }}>App: {APPEARANCE.contribution} · Geo: {GEOMETRY.contribution} · Motion: {MOTION.contribution}</p>
      </div>
    </section>
  )
}
