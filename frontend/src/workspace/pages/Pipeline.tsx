import { PIPELINE_STEPS } from "../../data/architecture"
import { PipelineFlow } from "../PipelineFlow"

type PipeStep = (typeof PIPELINE_STEPS)[number]

export function PipelinePage() {
  return (
    <div>
      <div className="ws-page-header">
        <p className="ws-kicker">07 — Architecture</p>
        <h1 className="ws-title">Architecture Pipeline</h1>
        <p className="ws-lede">Interactive breakdown of the full visual speech recognition pipeline from raw video to alphabet prediction.</p>
      </div>
      <div className="ws-card" style={{ marginBottom: "20px" }}>
        <div className="ws-card-title">Full Pipeline</div>
        <div style={{ marginTop: "16px" }}><PipelineFlow activeStage={6} /></div>
      </div>
      <div className="ws-grid-2">
        {PIPELINE_STEPS.map((step: PipeStep, i: number) => (
          <div key={step.id} className="ws-card">
            <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "12px" }}>
              <div style={{ width: "32px", height: "32px", borderRadius: "50%", background: "var(--ws-accent)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "12px", fontWeight: 700, color: "var(--ws-bg)", flexShrink: 0 }}>{i + 1}</div>
              <div>
                <div style={{ fontWeight: 700, fontSize: "15px" }}>{step.label}</div>
                <div style={{ fontSize: "11px", color: "var(--ws-text-dim)", fontFamily: "JetBrains Mono, monospace" }}>{step.id}</div>
              </div>
            </div>
            <p style={{ fontSize: "13px", color: "var(--ws-text-muted)", lineHeight: 1.5 }}>{step.note}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
