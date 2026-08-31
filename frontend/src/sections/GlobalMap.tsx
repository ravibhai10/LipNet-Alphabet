/**
 * GlobalMap.tsx — Optional 3D wireframe globe with pulsing prediction markers.
 */
import { GlobeMap } from "../components/3d/GlobeMap"

export function GlobalMap() {
  return (
    <section id="global-map" style={{ width: "100%", background: "linear-gradient(180deg,#08080c 0%,#0a0a18 100%)", padding: "64px 24px", display: "flex", flexDirection: "column", alignItems: "center", gap: 32 }}>
      <h2 style={{ fontFamily: "Fraunces, serif", fontSize: "clamp(24px,3vw,40px)", fontWeight: 700, color: "#fff", margin: 0 }}>Global Live Map</h2>
      <p style={{ color: "#7070a0", fontSize: 15, textAlign: "center", maxWidth: 480 }}>Live predictions pulsing around the world. Each dot is a recognition event.</p>
      <div style={{ width: "100%", maxWidth: 480, height: 320, borderRadius: 16, overflow: "hidden", border: "1px solid rgba(0,240,255,0.12)" }}>
        <GlobeMap markers={12} />
      </div>
    </section>
  )
}
